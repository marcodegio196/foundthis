"""Licensing export — assemble a delivery folder on local disk.

Stage 3's tagging pass already produced everything a stock marketplace asks for;
this turns it into files a human (or an upload tool) can work with: the clean
master, a per-clip JSON sidecar, and one CSV manifest for the whole batch.

Masters are **hard-linked** where the filesystem allows it and copied otherwise.
A 250GB archive that has already been rendered once should not be duplicated
again just to group it differently — a hard link is a second name for the same
bytes, costing nothing, and deleting the export never touches the render.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from .. import db, layout
from ..config import Config, config as default_config

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.csv"

# Column order for the CSV. Marketplaces differ on field names, so this is the
# superset worth having in front of you rather than any one platform's schema.
MANIFEST_FIELDS = (
    "shot_id", "file", "title", "description", "keywords",
    "country", "site", "captured_at", "gps_lat", "gps_lon",
    "drone_model", "width", "height", "fps", "duration_seconds",
    "license_tier", "aesthetic_score", "technical_score",
)


def keywords_for(row: db.Row) -> list[str]:
    """Search terms a buyer would actually type, de-duplicated, order kept."""
    candidates = [
        *(row["subject_tags"] or []),
        *(row["mood_tags"] or []),
        row["country"],
        row["site"],
        "aerial" if (row["drone_model"] or "") else None,
        "drone" if (row["drone_model"] or "") else None,
        "4k" if (row["source_width"] or 0) >= 3840 else None,
    ]
    seen: dict[str, None] = {}
    for value in candidates:
        if not value:
            continue
        term = str(value).replace("-", " ").strip().lower()
        if term:
            seen.setdefault(term, None)
    return list(seen)


def title_for(row: db.Row) -> str:
    """A short title derived from the description's first clause."""
    description = (row["description"] or "").strip()
    if description:
        title = description.split(".")[0].split(",")[0].strip()
        if title:
            return title[:120]
    place = " · ".join(p for p in (row["site"], row["country"]) if p)
    return (place or "Aerial footage").title()


def sidecar_for(row: db.Row) -> dict[str, Any]:
    """The per-clip metadata document written next to the master."""
    return {
        "shot_id": row["id"],
        "title": title_for(row),
        "description": row["description"],
        "keywords": keywords_for(row),
        "location": {
            "country": row["country"],
            "site": row["site"],
            "latitude": row["gps_lat"],
            "longitude": row["gps_lon"],
        },
        "capture": {
            "captured_at": row["captured_at"],
            "drone_model": row["drone_model"],
        },
        "technical": {
            "width": row["source_width"],
            "height": row["source_height"],
            "fps": row["source_fps"],
            "duration_seconds": round(row["duration"], 3),
            "aspect": row["source_aspect"],
        },
        "editorial": {
            "person_in_frame": bool(row["person_in_frame"]),
            "license_tier": row["license_tier"],
            "aesthetic_score": row["aesthetic_score"],
            "technical_score": row["technical_score"],
        },
        "provenance": {
            "source_file": row["source_relpath"],
            "in_point": row["in_point"],
            "out_point": row["out_point"],
        },
    }


def manifest_row(row: db.Row, relative_path: Path) -> dict[str, Any]:
    return {
        "shot_id": row["id"],
        "file": str(relative_path).replace(os.sep, "/"),
        "title": title_for(row),
        "description": row["description"] or "",
        "keywords": "; ".join(keywords_for(row)),
        "country": row["country"] or "",
        "site": row["site"] or "",
        "captured_at": row["captured_at"] or "",
        "gps_lat": row["gps_lat"] if row["gps_lat"] is not None else "",
        "gps_lon": row["gps_lon"] if row["gps_lon"] is not None else "",
        "drone_model": row["drone_model"] or "",
        "width": row["source_width"] or "",
        "height": row["source_height"] or "",
        "fps": row["source_fps"] or "",
        "duration_seconds": round(row["duration"], 3),
        "license_tier": row["license_tier"] or "",
        "aesthetic_score": row["aesthetic_score"] if row["aesthetic_score"] is not None else "",
        "technical_score": row["technical_score"] if row["technical_score"] is not None else "",
    }


def place_file(source: Path, target: Path) -> str:
    """Hard-link `source` to `target`, falling back to a copy. Returns the method.

    Cross-device links and filesystems without link support (some network and
    exFAT volumes) raise, which is the expected case on an external drive.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
        return "link"
    except (OSError, NotImplementedError):
        shutil.copy2(source, target)
        return "copy"


def run(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    tier: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Export every rendered, tagged, un-rejected shot into `export_root`.

    Filter by `license_tier` to assemble a batch for one marketplace; without
    it, everything licensable is exported.
    """
    query = (
        "SELECT * FROM shot_details WHERE rejected = 0 "
        "AND licensing_render_path IS NOT NULL AND tagged_at IS NOT NULL"
    )
    params: list[Any] = []
    if tier:
        query += " AND license_tier = ?"
        params.append(tier)
    query += " ORDER BY country, captured_at"
    if limit:
        query += f" LIMIT {int(limit)}"

    rows = conn.execute(query, params).fetchall()
    counts = {"exported": 0, "linked": 0, "copied": 0, "missing": 0}
    manifest: list[dict[str, Any]] = []

    root = cfg.export_root / tier if tier else cfg.export_root

    for row in rows:
        master = Path(row["licensing_render_path"])
        if not master.is_file():
            log.warning("shot %s: master missing at %s", row["id"], master)
            counts["missing"] += 1
            continue

        relative = layout.render_path(
            Path("."),
            shot_id=row["id"],
            source_relpath=row["source_relpath"],
            country=row["country"],
            captured_at=row["captured_at"],
            site=row["site"],
            suffix=master.suffix,
        ).relative_to(".")
        manifest.append(manifest_row(row, relative))

        if dry_run:
            counts["exported"] += 1
            continue

        target = root / relative
        method = place_file(master, target)
        counts["linked" if method == "link" else "copied"] += 1

        target.with_suffix(".json").write_text(
            json.dumps(sidecar_for(row), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        counts["exported"] += 1

    if manifest and not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        with (root / MANIFEST_NAME).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(manifest)
        log.info("wrote %s with %d rows", root / MANIFEST_NAME, len(manifest))

    return counts
