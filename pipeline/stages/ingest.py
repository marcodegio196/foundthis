"""Stage 1a — walk the archive and register every source file.

Idempotent: keyed on absolute path, and files whose size and mtime are unchanged
are skipped without re-probing, so re-running over 250GB costs a stat() per file.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterator

from .. import db, media
from ..config import Config, config as default_config

log = logging.getLogger(__name__)


def iter_video_files(root: Path, extensions: tuple[str, ...]) -> Iterator[Path]:
    suffixes = {ext.lower() for ext in extensions}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith("."):
            yield path


def country_from_path(path: Path, root: Path) -> str | None:
    """Folder convention: archive/<country>/<file>, any depth below that.

    Files sitting directly in the archive root have no country to infer.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    return relative.parts[0].replace("_", " ").lower() if len(relative.parts) > 1 else None


def is_unchanged(row: db.Row | None, stat_result) -> bool:
    return bool(
        row
        and row["size_bytes"] == stat_result.st_size
        and abs(row["mtime"] - stat_result.st_mtime) < 1e-6
        and row["duration"] is not None
    )


def run(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Register (or refresh) every video file under the archive root."""
    root = cfg.archive_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"archive root not found: {root}")

    known = {
        row["path"]: row for row in conn.execute("SELECT * FROM sources").fetchall()
    }
    counts = {"seen": 0, "added": 0, "updated": 0, "skipped": 0, "failed": 0}

    for path in iter_video_files(root, cfg.video_extensions):
        if limit and counts["seen"] >= limit:
            break
        counts["seen"] += 1
        existing = known.get(str(path))
        stat_result = path.stat()
        unchanged = is_unchanged(existing, stat_result)

        if not force and unchanged:
            counts["skipped"] += 1
            continue

        try:
            probed = media.probe(path)
        except media.MediaError as exc:
            log.warning("probe failed for %s: %s", path, exc)
            counts["failed"] += 1
            continue

        values = {
            "path": str(path),
            "relpath": str(path.relative_to(root)),
            "size_bytes": stat_result.st_size,
            "mtime": stat_result.st_mtime,
            "country": country_from_path(path, root),
            **probed,
        }
        with db.transaction(conn):
            source_id = db.upsert_source(conn, **values)
            # Only when the bytes actually changed do the old shot boundaries
            # describe a file that no longer exists. A --force re-probe of an
            # untouched file must not discard downstream stage state.
            if existing and not unchanged:
                conn.execute(
                    "UPDATE sources SET scene_detected_at = NULL WHERE id = ?",
                    (source_id,),
                )
        counts["updated" if existing else "added"] += 1
        log.info("%s %s", "updated" if existing else "added", values["relpath"])

    return counts
