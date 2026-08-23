"""Stage 5 — publish rendered clips and track what happened to them.

The Zernio protocol lives in `pipeline/zernio.py`, ported from the working
integration in the `new-visu` repo. This module is the pipeline half: which
shots are ready, what the caption says, and what gets written back.

One correction worth stating plainly: Zernio reports **delivery status**
(queued / published / failed, per platform), not performance. The feedback loop
that reweights Stage 4 selection needs view and engagement counts, which come
from somewhere else — so those land in `metrics`, and delivery status lands in
`delivery_status`, and the two are never conflated.

Licensing upload stays manual: it isn't worth automating until a marketplace
proves it converts.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from .. import db
from ..config import Config, config as default_config
from ..zernio import ZernioClient, ZernioError, summarise_delivery

log = logging.getLogger(__name__)

# Vertical renders suit reels and TikTok; Facebook is left out unless asked for.
DEFAULT_PLATFORMS = ("instagram", "tiktok")

# Same shape as db.now() — naive text, UTC implied — so scheduled_at compares
# correctly against it with plain string ordering.
_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _format_ts(when: datetime) -> str:
    return when.strftime(_TS_FORMAT)


def _parse_ts(text: str) -> datetime:
    return datetime.strptime(text, _TS_FORMAT).replace(tzinfo=timezone.utc)


# Instagram shows roughly the first 125 characters and hides the rest behind
# "more". The caption is built to use that: the headline alone above the fold,
# the place and year only for someone who taps.
#
# The gap cannot be blank lines. Instagram strips trailing whitespace and
# collapses runs of empty lines, so a stack of newlines arrives as a single
# break and the reveal never happens. U+2800 BRAILLE PATTERN BLANK renders as
# nothing but is not whitespace, so it survives that normalisation.
GAP_LINE = "\u2800"
GAP_LINES = 10


def build_caption(row: db.Row, cfg: Config = default_config) -> str:
    """Headline, a long gap, then where and when, read after tapping "more".

    Everything here comes from the file and the folder holding it: the country
    from `input/<country>/`, the year from the capture timestamp. No tagging
    pass, so nothing costs a model call per shot or goes stale.
    """
    year = (row["captured_at"] or "")[:4]
    country = (row["country"] or "").title()
    place = " \u00b7 ".join(part for part in (country, year) if part)
    if not place:
        return cfg.overlay_text

    gap = "\n".join([""] + [GAP_LINE] * GAP_LINES + [""])
    return f"{cfg.overlay_text}{gap}\n{place}"


class PublishClient(Protocol):
    def publish(
        self, video_path: Path, caption: str, *, platforms: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Upload a render and publish it now; return the Zernio response."""

    def upload_media(self, video_path: Path) -> str:
        """Upload a render, return its hosted URL, without posting."""

    def publish_url(
        self,
        media_url: str,
        caption: str,
        *,
        media_type: str = "video",
        platforms: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Publish media that's already hosted — no local file needed."""

    def fetch_post(self, post_id: str) -> dict[str, Any] | None:
        """Fetch a published post's current state."""


class DryRunClient:
    """Records what would be published. The default until an API key is set."""

    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self.uploaded: list[Path] = []

    def upload_media(self, video_path):
        self.uploaded.append(Path(video_path))
        log.info("[dry run] would upload %s", video_path)
        return f"dry-run://{Path(video_path).name}"

    def publish_url(self, media_url, caption, *, media_type="video", platforms=None):
        self.published.append((media_url, caption))
        log.info("[dry run] would publish %s to %s", media_url, platforms or "all")
        return {}

    def publish(self, video_path, caption, *, platforms=None):
        return self.publish_url(self.upload_media(video_path), caption, platforms=platforms)

    def fetch_post(self, post_id: str):
        return None


def build_client(cfg: Config = default_config) -> PublishClient:
    try:
        return ZernioClient()
    except ZernioError as exc:
        log.warning("%s — running in dry-run mode", exc)
        return DryRunClient()


def schedule_pending(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    count: int | None = None,
    interval_hours: float | None = None,
    start_at: datetime | None = None,
) -> list[tuple[db.Row, datetime]]:
    """Give rendered, un-posted, un-scheduled shots a future posting time.

    This is a queue inside the pipeline, not a Zernio feature — `publish`
    still sends each post as `publishNow` once its slot arrives (see
    `publish_pending`). That keeps a blocked or skipped run harmless: nothing
    posts early, and nothing is lost, because a shot just waits at
    `posted_at IS NULL` until a `publish` run actually reaches its time.

    Slots are spaced `interval_hours` apart, starting after whatever is
    already scheduled — so calling this again to add more shots extends the
    queue instead of double-booking an existing slot. Only touches shots with
    no `scheduled_at` yet; to change an existing slot, clear it first.
    """
    interval = interval_hours if interval_hours is not None else cfg.post_interval_hours
    rows = conn.execute(
        "SELECT * FROM shot_details WHERE selection_state = 'approved' "
        "AND rendered_at IS NOT NULL AND posted_at IS NULL AND scheduled_at IS NULL "
        "ORDER BY approved_at"
        + (f" LIMIT {int(count)}" if count else "")
    ).fetchall()
    if not rows:
        return []

    if start_at is not None:
        next_slot = start_at
    else:
        latest = conn.execute(
            "SELECT MAX(scheduled_at) FROM shots "
            "WHERE scheduled_at IS NOT NULL AND posted_at IS NULL"
        ).fetchone()[0]
        next_slot = (
            _parse_ts(latest) + timedelta(hours=interval)
            if latest
            else datetime.now(timezone.utc)
        )

    scheduled: list[tuple[db.Row, datetime]] = []
    with db.transaction(conn):
        for row in rows:
            db.update_shot(conn, row["id"], scheduled_at=_format_ts(next_slot))
            scheduled.append((row, next_slot))
            next_slot = next_slot + timedelta(hours=interval)
    return scheduled


def upload_pending(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
    client: PublishClient | None = None,
) -> dict[str, int]:
    """Host every rendered, un-uploaded render on Zernio, ahead of posting it.

    This is the step that needs the actual file and a real network upload —
    meant to run wherever the render lives (next to the archive, with
    ffmpeg). Once `social_media_url` is set, `publish_pending` never touches
    the file again, so a runner with only the database (a cloud publish job,
    say) can still post it.
    """
    client = client if client is not None else build_client(cfg)
    rows = conn.execute(
        "SELECT * FROM shot_details WHERE social_render_path IS NOT NULL "
        "AND social_media_url IS NULL AND posted_at IS NULL "
        "ORDER BY rendered_at"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()

    counts = {"uploaded": 0, "failed": 0}
    for row in rows:
        try:
            media_url = client.upload_media(Path(row["social_render_path"]))
        except Exception as exc:
            log.warning("upload failed for shot %s: %s", row["id"], exc)
            counts["failed"] += 1
            continue
        with db.transaction(conn):
            db.update_shot(
                conn, row["id"], social_media_url=media_url, media_uploaded_at=db.now()
            )
        counts["uploaded"] += 1
        log.info("shot %s uploaded to %s", row["id"], media_url)

    return counts


def publish_pending(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
    platforms: Sequence[str] = DEFAULT_PLATFORMS,
    client: PublishClient | None = None,
) -> dict[str, int]:
    """Post every rendered, approved, un-posted shot whose time has come.

    A shot with no `scheduled_at` posts as soon as this runs, same as before
    scheduling existed — `schedule` is opt-in per shot, not a mode switch.

    A shot already uploaded (see `upload_pending`) posts from its stored URL
    and never touches `social_render_path` — the only case that needs the
    file on disk is one nobody uploaded ahead of time, which is the normal
    local flow (render then publish in the same run).
    """
    client = client if client is not None else build_client(cfg)
    rows = conn.execute(
        "SELECT * FROM shot_details WHERE selection_state = 'approved' "
        "AND social_render_path IS NOT NULL AND posted_at IS NULL "
        "AND (scheduled_at IS NULL OR scheduled_at <= ?) "
        "ORDER BY approved_at"
        + (f" LIMIT {int(limit)}" if limit else ""),
        (db.now(),),
    ).fetchall()

    counts = {"posted": 0, "failed": 0}
    for row in rows:
        caption = build_caption(row, cfg)
        try:
            if row["social_media_url"]:
                response = client.publish_url(
                    row["social_media_url"], caption, platforms=platforms
                )
            else:
                response = client.publish(
                    Path(row["social_render_path"]), caption, platforms=platforms
                )
        except Exception as exc:
            log.warning("publish failed for shot %s: %s", row["id"], exc)
            counts["failed"] += 1
            continue

        post = response.get("post") if isinstance(response, dict) else None
        zernio_post_id = (post or {}).get("_id") if isinstance(post, dict) else None

        # A dry run returns nothing; leaving posted_at unset keeps the shot in
        # the queue rather than silently marking it published.
        if not zernio_post_id:
            continue

        with db.transaction(conn):
            db.update_shot(
                conn,
                row["id"],
                caption=caption,
                zernio_post_id=str(zernio_post_id),
                platform_ids={
                    p["platform"]: p.get("accountId", "")
                    for p in (post.get("platforms") or [])
                    if isinstance(p, dict) and p.get("platform")
                },
                delivery_status=summarise_delivery(post),
                delivery_checked_at=db.now(),
                posted_at=db.now(),
            )
        counts["posted"] += 1
        log.info("shot %s posted as %s", row["id"], zernio_post_id)

    return counts


def sync_delivery(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    client: PublishClient | None = None,
) -> dict[str, int]:
    """Re-check what Zernio did with each posted shot.

    A post accepted by the API can still fail at the platform minutes later,
    so "posted" is not the end of the story — this is what turns a silent
    Instagram rejection into something visible in `stats`.
    """
    client = client if client is not None else build_client(cfg)
    rows = conn.execute(
        "SELECT id, zernio_post_id FROM shots WHERE zernio_post_id IS NOT NULL"
    ).fetchall()

    counts = {"checked": 0, "failed_delivery": 0}
    for row in rows:
        post = client.fetch_post(row["zernio_post_id"])
        if not post:
            continue
        statuses = summarise_delivery(post)
        with db.transaction(conn):
            db.update_shot(
                conn, row["id"],
                delivery_status=statuses,
                delivery_checked_at=db.now(),
            )
        counts["checked"] += 1
        if any(s.startswith(("failed", "error")) for s in statuses.values()):
            counts["failed_delivery"] += 1
            log.warning("shot %s delivery problem: %s", row["id"], statuses)

    return counts


def run(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
    platforms: Sequence[str] = DEFAULT_PLATFORMS,
    client: PublishClient | None = None,
) -> dict[str, int]:
    client = client if client is not None else build_client(cfg)
    published = publish_pending(
        conn, cfg, limit=limit, platforms=platforms, client=client
    )
    return {**published, **sync_delivery(conn, cfg, client=client)}
