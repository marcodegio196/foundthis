"""Stage 4 — select shots and render the two profiles.

Selection is semi-manual by design until the format is validated: `queue`
proposes shots under diversity rules, a human approves, and only approved shots
render. The rules are already tag-driven, so automating the approval later is
deleting the human step, not rewriting the stage.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from .. import db, media, render
from ..config import Config, config as default_config

log = logging.getLogger(__name__)

# How many recent posts a country or mood must sit outside of to come up again.
# The feed's whole premise is range; three coastlines in a row reads as one trip.
COUNTRY_COOLDOWN = 4
MOOD_COOLDOWN = 3

# Roughly one in eight posts should have you in frame — enough to make it a
# personal record rather than a stock reel, not so much that it becomes vlog.
PERSON_EVERY = 8


def _recent(conn: sqlite3.Connection, limit: int) -> list[db.Row]:
    return conn.execute(
        "SELECT * FROM shot_details WHERE posted_at IS NOT NULL "
        "ORDER BY posted_at DESC LIMIT ?",
        (limit,),
    ).fetchall()


def propose_queue(
    conn: sqlite3.Connection, count: int = 10
) -> list[tuple[db.Row, str]]:
    """Pick the next shots to review, with the reason each was chosen.

    Returns (shot, reason) so the approval step shows *why* a shot surfaced —
    a queue you can't interrogate is one you stop trusting and start ignoring.
    """
    recent = _recent(conn, max(COUNTRY_COOLDOWN, MOOD_COOLDOWN))
    recent_countries = {r["country"] for r in recent[:COUNTRY_COOLDOWN] if r["country"]}
    recent_moods = {m for r in recent[:MOOD_COOLDOWN] for m in (r["mood_tags"] or [])}

    posted = conn.execute(
        "SELECT count(*) FROM shots WHERE posted_at IS NOT NULL"
    ).fetchone()[0]
    want_person = posted > 0 and posted % PERSON_EVERY == 0

    candidates = db.selection_queue(conn)
    chosen: list[tuple[db.Row, str]] = []
    used_countries: set[str] = set()

    for row in candidates:
        if len(chosen) >= count:
            break
        moods = set(row["mood_tags"] or [])

        if want_person and not row["person_in_frame"]:
            continue
        if row["country"] and row["country"] in recent_countries | used_countries:
            continue
        if moods and moods <= recent_moods:
            continue

        reason = "you in frame" if want_person else "new country / mood"
        chosen.append((row, reason))
        if row["country"]:
            used_countries.add(row["country"])

    # Diversity rules can starve the queue on a thin archive; fall back to the
    # best-scoring survivors rather than returning nothing to review.
    if len(chosen) < count:
        picked = {row["id"] for row, _ in chosen}
        for row in candidates:
            if len(chosen) >= count:
                break
            if row["id"] not in picked:
                chosen.append((row, "top score (diversity rules exhausted)"))

    return chosen


def stage_queue(conn: sqlite3.Connection, count: int = 10) -> list[db.Row]:
    """Mark the proposed shots as queued so the approval step has a stable list."""
    proposed = propose_queue(conn, count)
    with db.transaction(conn):
        for row, _ in proposed:
            db.update_shot(conn, row["id"], selection_state="queued")
    return [row for row, _ in proposed]


def approve(conn: sqlite3.Connection, shot_ids: list[int]) -> int:
    with db.transaction(conn):
        for shot_id in shot_ids:
            db.update_shot(
                conn, shot_id, selection_state="approved", approved_at=db.now()
            )
    return len(shot_ids)


def overlay_for(row: db.Row, cfg: Config = default_config) -> list[str]:
    year = (row["captured_at"] or "")[:4] or None
    return render.overlay_lines(year, row["country"], row["site"], cfg.overlay_text)


def render_shot(
    conn: sqlite3.Connection,
    row: db.Row,
    cfg: Config = default_config,
    *,
    profiles: tuple[str, ...] = ("social", "licensing"),
) -> dict[str, str]:
    """Render one approved shot into the requested profiles."""
    written: dict[str, str] = {}
    stem = f"{row['id']:06d}_{Path(row['source_relpath']).stem}"

    if "social" in profiles:
        target = cfg.social_render_dir / f"{stem}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = overlay_for(row, cfg)
        media.run(
            render.build_social_command(
                row["source_path"], target,
                in_point=row["in_point"], out_point=row["out_point"],
                width=row["source_width"] or 0, height=row["source_height"] or 0,
                lines=lines,
                font_file=cfg.overlay_font,
            ),
            capture=True,
        )
        written["social"] = str(target)

    if "licensing" in profiles:
        target = cfg.licensing_render_dir / f"{stem}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        media.run(
            render.build_licensing_command(
                row["source_path"], target,
                in_point=row["in_point"], out_point=row["out_point"],
                codec=cfg.licensing_codec,
            ),
            capture=True,
        )
        written["licensing"] = str(target)

    with db.transaction(conn):
        db.update_shot(
            conn,
            row["id"],
            social_render_path=written.get("social"),
            licensing_render_path=written.get("licensing"),
            overlay_text=" / ".join(overlay_for(row, cfg)),
            rendered_at=db.now(),
        )
    return written


def run(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
    profiles: tuple[str, ...] = ("social", "licensing"),
) -> dict[str, int]:
    """Render every approved shot that has not been rendered yet."""
    rows = conn.execute(
        "SELECT * FROM shot_details WHERE selection_state = 'approved' "
        "AND rendered_at IS NULL ORDER BY approved_at"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()

    counts = {"rendered": 0, "failed": 0}
    for row in rows:
        try:
            written = render_shot(conn, row, cfg, profiles=profiles)
        except media.MediaError as exc:
            log.warning("render failed for shot %s: %s", row["id"], exc)
            counts["failed"] += 1
            continue
        counts["rendered"] += 1
        log.info("shot %s -> %s", row["id"], ", ".join(written.values()))

    return counts
