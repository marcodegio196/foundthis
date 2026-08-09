"""Stage 5 — publish rendered clips and feed results back into selection.

The Zernio client here is an adapter written against the shape its API is
expected to have (bearer token, POST a post, GET metrics by post id). The
endpoint paths are configuration, not constants, so pointing it at the real API
is an env change rather than a rewrite — and `DryRunClient` lets the rest of
the stage be exercised end to end before any of that is settled.

Licensing upload stays deliberately manual: uploading a master to a marketplace
is not worth automating until one proves it converts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .. import db
from ..config import Config, config as default_config

log = logging.getLogger(__name__)


def build_caption(row: db.Row, cfg: Config = default_config) -> str:
    """Caption for the post: the headline, the place, then mood as hashtags."""
    location = " · ".join(part for part in (row["site"], row["country"]) if part)
    parts = [cfg.overlay_text]
    if location:
        parts.append(location.title())
    if row["description"]:
        parts.append(row["description"])

    tags = list(row["mood_tags"] or []) + list(row["subject_tags"] or [])
    body = "\n\n".join(parts)
    if tags:
        body += "\n\n" + " ".join(f"#{tag.replace('-', '')}" for tag in tags[:8])
    return body


class PublishClient(Protocol):
    def publish(self, video_path: Path, caption: str) -> dict[str, str]:
        """Publish a clip; return {platform: post_id}."""

    def metrics(self, platform_ids: dict[str, str]) -> dict[str, Any]:
        """Fetch current performance metrics for previously published posts."""


class DryRunClient:
    """Records what would be published. The default until a token is set."""

    def __init__(self) -> None:
        self.published: list[tuple[Path, str]] = []

    def publish(self, video_path: Path, caption: str) -> dict[str, str]:
        self.published.append((video_path, caption))
        log.info("[dry run] would publish %s", video_path)
        return {}

    def metrics(self, platform_ids: dict[str, str]) -> dict[str, Any]:
        return {}


class ZernioClient:
    """Minimal HTTP adapter — stdlib only, so Stage 5 adds no dependency."""

    def __init__(self, base_url: str, token: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"zernio {method} {path} failed: {exc.code} {exc.read()[:200]!r}"
            ) from exc

    def publish(self, video_path: Path, caption: str) -> dict[str, str]:
        # The upload itself is a separate call from the post so a failed publish
        # doesn't mean re-uploading a 40MB file.
        media_id = self._request(
            "POST", "/v1/media", {"filename": video_path.name}
        )["id"]
        self._upload(media_id, video_path)
        result = self._request(
            "POST", "/v1/posts", {"media_id": media_id, "caption": caption}
        )
        return {p["platform"]: p["id"] for p in result.get("platforms", [])}

    def _upload(self, media_id: str, video_path: Path) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/v1/media/{media_id}/content",
            data=video_path.read_bytes(),
            method="PUT",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "video/mp4",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout):
            pass

    def metrics(self, platform_ids: dict[str, str]) -> dict[str, Any]:
        collected: dict[str, Any] = {}
        for platform, post_id in platform_ids.items():
            try:
                collected[platform] = self._request("GET", f"/v1/posts/{post_id}/metrics")
            except RuntimeError as exc:
                log.warning("metrics fetch failed for %s/%s: %s", platform, post_id, exc)
        return collected


def build_client(cfg: Config = default_config) -> PublishClient:
    if not cfg.zernio_token:
        log.warning("ZERNIO_TOKEN unset — running in dry-run mode")
        return DryRunClient()
    return ZernioClient(cfg.zernio_base_url, cfg.zernio_token)


def publish_pending(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
    client: PublishClient | None = None,
) -> dict[str, int]:
    """Post every rendered, approved, un-posted shot."""
    client = client if client is not None else build_client(cfg)
    rows = conn.execute(
        "SELECT * FROM shot_details WHERE selection_state = 'approved' "
        "AND social_render_path IS NOT NULL AND posted_at IS NULL "
        "ORDER BY approved_at"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()

    counts = {"posted": 0, "failed": 0}
    for row in rows:
        caption = build_caption(row, cfg)
        try:
            platform_ids = client.publish(Path(row["social_render_path"]), caption)
        except Exception as exc:
            log.warning("publish failed for shot %s: %s", row["id"], exc)
            counts["failed"] += 1
            continue

        # A dry run produces no platform ids; leaving posted_at unset keeps the
        # shot in the queue rather than silently marking it published.
        if not platform_ids:
            continue

        with db.transaction(conn):
            db.update_shot(
                conn,
                row["id"],
                caption=caption,
                platform_ids=platform_ids,
                posted_at=db.now(),
            )
        counts["posted"] += 1
        log.info("shot %s posted to %s", row["id"], ", ".join(platform_ids))

    return counts


def refresh_metrics(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    client: PublishClient | None = None,
) -> dict[str, int]:
    """Pull performance metrics for posted shots and store them on the row.

    This is what eventually reweights Stage 4 selection — which moods and
    countries actually perform — so it is worth running long before there is
    enough volume to draw conclusions from.
    """
    client = client if client is not None else build_client(cfg)
    rows = conn.execute(
        "SELECT * FROM shots WHERE posted_at IS NOT NULL AND platform_ids IS NOT NULL"
    ).fetchall()

    counts = {"updated": 0, "failed": 0}
    for row in rows:
        try:
            metrics = client.metrics(row["platform_ids"])
        except Exception as exc:
            log.warning("metrics failed for shot %s: %s", row["id"], exc)
            counts["failed"] += 1
            continue
        if not metrics:
            continue
        with db.transaction(conn):
            db.update_shot(
                conn, row["id"], metrics=metrics, metrics_updated_at=db.now()
            )
        counts["updated"] += 1

    return counts


def run(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
    client: PublishClient | None = None,
) -> dict[str, int]:
    client = client if client is not None else build_client(cfg)
    published = publish_pending(conn, cfg, limit=limit, client=client)
    return {**published, **refresh_metrics(conn, cfg, client=client)}
