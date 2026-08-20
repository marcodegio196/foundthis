"""Stage 4 — select shots and render the two profiles.

Selection is semi-manual by design until the format is validated: `queue`
proposes shots under diversity rules, a human approves, and only approved shots
render. The rules are already tag-driven, so automating the approval later is
deleting the human step, not rewriting the stage.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from pathlib import Path

from .. import db, layout, media, motion, render
from ..config import Config, config as default_config

log = logging.getLogger(__name__)

# How many recent posts a country, year, or mood must sit outside of to come up
# again. The feed's whole premise is range; three coastlines in a row reads as
# one trip, and a run of clips from the same summer reads as one holiday even
# when the countries differ.
COUNTRY_COOLDOWN = 4
YEAR_COOLDOWN = 3
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
    conn: sqlite3.Connection,
    count: int = 10,
    *,
    rng: random.Random | None = None,
) -> list[tuple[db.Row, str]]:
    """Pick the next shots to review, with the reason each was chosen.

    Returns (shot, reason) so the approval step shows *why* a shot surfaced —
    a queue you can't interrogate is one you stop trusting and start ignoring.

    Order is random, not ranked. Ranking by score walks the archive in quality
    order, which on a 250GB library means the feed works through the best
    country first and only reaches the rest months later; shuffling spreads the
    whole archive across the calendar instead. Diversity is then applied on top
    of the shuffle as a filter, so what comes out is random *within* the range
    rules rather than a fixed rotation.

    The rules relax in tiers when the archive is too thin to satisfy them, so a
    small library still produces a queue — but the reason attached to each pick
    says which tier it came from, so a feed that has quietly stopped varying is
    visible rather than silent.
    """
    rng = rng or random.Random(default_config.selection_seed)

    recent = _recent(conn, max(COUNTRY_COOLDOWN, YEAR_COOLDOWN, MOOD_COOLDOWN))
    recent_countries = {r["country"] for r in recent[:COUNTRY_COOLDOWN] if r["country"]}
    recent_years = {
        layout.year_of(r["captured_at"]) for r in recent[:YEAR_COOLDOWN]
    }
    recent_moods = {m for r in recent[:MOOD_COOLDOWN] for m in (r["mood_tags"] or [])}

    posted = conn.execute(
        "SELECT count(*) FROM shots WHERE posted_at IS NOT NULL"
    ).fetchone()[0]
    want_person = posted > 0 and posted % PERSON_EVERY == 0

    candidates = list(db.selection_queue(conn))
    rng.shuffle(candidates)

    # `person_in_frame` is only known if the tagging pass ran. Without it the
    # slot would match nothing and this post would return an empty queue rather
    # than a shot, stalling the feed every PERSON_EVERY posts.
    if want_person and not any(row["person_in_frame"] for row in candidates):
        want_person = False

    chosen: list[tuple[db.Row, str]] = []
    picked: set[int] = set()
    used_countries: set[str] = set()
    used_years: set[str] = set()

    def take(row: db.Row, reason: str) -> None:
        chosen.append((row, reason))
        picked.add(row["id"])
        if row["country"]:
            used_countries.add(row["country"])
        used_years.add(layout.year_of(row["captured_at"]))

    # Each tier drops one rule, weakest first. Mood goes before year because it
    # is the softest signal and is absent entirely unless the optional tagging
    # pass ran — leaving it in an early tier let it knock every candidate down
    # to the country-only tier, where year is not checked at all, which quietly
    # disabled the year rule. Country outlives year because two clips from the
    # same place read as repetition more strongly than two from the same summer.
    tiers = (
        ("new country / year / mood", True, True, True),
        ("new country / year", True, True, False),
        ("new country", True, False, False),
        ("no rules left to apply", False, False, False),
    )

    for reason, by_country, by_year, by_mood in tiers:
        for row in candidates:
            if len(chosen) >= count:
                return chosen
            if row["id"] in picked:
                continue
            if want_person and not row["person_in_frame"]:
                continue
            if by_country and row["country"] and (
                row["country"] in recent_countries | used_countries
            ):
                continue
            if by_year and layout.year_of(row["captured_at"]) in (
                recent_years | used_years
            ):
                continue
            if by_mood:
                moods = set(row["mood_tags"] or [])
                if moods and moods <= recent_moods:
                    continue
            take(row, "you in frame" if want_person else reason)

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


def clip_bounds(row: db.Row, cfg: Config = default_config) -> tuple[float, float]:
    """In and out points for a render, after applying the length ceiling.

    Trimming rather than skipping is deliberate: a long held shot is usually
    good footage that simply outruns the format, and the opening seconds are the
    composed ones — the tail is where the drone starts drifting off.
    """
    in_point, out_point = row["in_point"], row["out_point"]
    if cfg.max_render_seconds > 0:
        out_point = min(out_point, in_point + cfg.max_render_seconds)
    return in_point, out_point


def render_shot(
    conn: sqlite3.Connection,
    row: db.Row,
    cfg: Config = default_config,
    *,
    profiles: tuple[str, ...] = ("social", "licensing"),
) -> dict[str, str]:
    """Render one approved shot into the requested profiles."""
    written: dict[str, str] = {}
    in_point, out_point = clip_bounds(row, cfg)

    def target_for(root: Path) -> Path:
        return layout.render_path(
            root,
            shot_id=row["id"],
            source_relpath=row["source_relpath"],
            country=row["country"],
            captured_at=row["captured_at"],
            site=row["site"],
        )

    if "social" in profiles:
        target = target_for(cfg.social_render_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = overlay_for(row, cfg)
        media.run(
            render.build_social_command(
                row["source_path"], target,
                in_point=in_point, out_point=out_point,
                width=row["source_width"] or 0, height=row["source_height"] or 0,
                lines=lines,
                font_file=cfg.overlay_font,
            ),
            capture=True,
        )
        written["social"] = str(target)

    if "licensing" in profiles:
        target = target_for(cfg.licensing_render_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        media.run(
            render.build_licensing_command(
                row["source_path"], target,
                in_point=in_point, out_point=out_point,
                codec=cfg.licensing_codec,
            ),
            capture=True,
        )
        written["licensing"] = str(target)

    # Only record the profiles this run actually wrote. Passing None for the
    # other one would clear a path that is still on disk, so `render --profiles
    # social` would silently lose the licensing master it never touched.
    updates: dict[str, object] = {
        "overlay_text": " / ".join(overlay_for(row, cfg)),
        "rendered_at": db.now(),
    }
    if "social" in written:
        updates["social_render_path"] = written["social"]
    if "licensing" in written:
        updates["licensing_render_path"] = written["licensing"]

    with db.transaction(conn):
        db.update_shot(conn, row["id"], **updates)
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

    counts = {
        "rendered": 0, "skipped_short": 0, "skipped_jerk": 0,
        "skipped_rejected": 0, "failed": 0,
    }
    for row in rows:
        # Repositioning never renders, however it was approved. A jerk that
        # reached the queue is an upstream mistake, not something to burn an
        # overlay onto and post.
        if row["motion_class"] == motion.REPOSITION:
            log.info("shot %s skipped: repositioning", row["id"])
            counts["skipped_jerk"] += 1
            continue

        # A rejected shot is not publishable, and approving one individually is
        # not the way to overrule that — `score --rejections-only` re-draws the
        # bar for everything at once, which is the honest way to change it.
        if row["rejected"]:
            log.info("shot %s skipped: %s", row["id"], row["reject_reason"] or "rejected")
            counts["skipped_rejected"] += 1
            continue

        in_point, out_point = clip_bounds(row, cfg)
        if cfg.min_render_seconds > 0 and (out_point - in_point) < cfg.min_render_seconds:
            log.info(
                "shot %s skipped: %.2fs is under MIN_RENDER_SECONDS=%.2f",
                row["id"], out_point - in_point, cfg.min_render_seconds,
            )
            counts["skipped_short"] += 1
            continue

        try:
            written = render_shot(conn, row, cfg, profiles=profiles)
        except media.MediaError as exc:
            log.warning("render failed for shot %s: %s", row["id"], exc)
            counts["failed"] += 1
            continue
        counts["rendered"] += 1
        log.info("shot %s -> %s", row["id"], ", ".join(written.values()))

    return counts
