"""Zernio API client.

Ported from the working integration in the `new-visu` repo
(`new_visu/customer/social/late_service.py`) rather than written against a
guess. The protocol details below — presigned upload, the `platforms` array
keyed by Zernio account id, `post._id` in the response — all come from there.

Two things that repo learned the hard way and are reproduced here:

- Instagram and TikTok reject a post with no media. Every social render this
  pipeline produces has video, so that is satisfied by construction, but the
  check stays because a missing render is otherwise a confusing API error.
- Vertical video posts to Instagram as a reel only if `platformSpecificData`
  says so. Everything this pipeline renders for social is 9:16, so reels is
  the default here rather than a per-post flag.

Deliberately stdlib-only: `new-visu` uses httpx inside a Flask app, but this
pipeline has no web stack and adding one for four calls isn't worth it.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://zernio.com/api/v1"
TIMEOUT = 30.0
UPLOAD_TIMEOUT = 120.0

# sk_ + 64 hex. Wrong-length keys are the most common setup mistake, and the
# resulting 401 doesn't say so.
API_KEY_PREFIX = "sk_"
API_KEY_LENGTH = 67

# Platforms that reject a post without media.
PLATFORMS_REQUIRING_MEDIA = frozenset({"instagram", "tiktok"})

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class ZernioError(RuntimeError):
    def __init__(self, message: str, status_code: int = 0, detail: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail

    @property
    def retryable(self) -> bool:
        return self.status_code in RETRYABLE_STATUS


def content_type_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    guessed = CONTENT_TYPES.get(ext) or mimetypes.guess_type(filename)[0]
    if not guessed:
        raise ZernioError(f"unsupported upload extension: {ext or '(none)'}")
    return guessed


def media_item_type(filename: str) -> str:
    return "video" if content_type_for(filename).startswith("video/") else "image"


def read_api_key(env: dict[str, str] | None = None) -> str:
    """Read and sanity-check ZERNIO_API_KEY.

    Strips the BOM and stray quotes a .env file picks up on Windows — the
    failure mode otherwise is a 401 that looks like a revoked key.
    """
    env = env if env is not None else dict(os.environ)
    key = (env.get("ZERNIO_API_KEY") or "").strip().lstrip("﻿").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = key[1:-1].strip()
    if not key:
        raise ZernioError("ZERNIO_API_KEY is not set")
    if not key.startswith(API_KEY_PREFIX):
        log.warning("ZERNIO_API_KEY does not start with %r — check .env", API_KEY_PREFIX)
    elif len(key) != API_KEY_LENGTH:
        log.warning(
            "ZERNIO_API_KEY is %d characters (expected %d) — likely truncated",
            len(key), API_KEY_LENGTH,
        )
    return key


def parse_error(status: int, body: bytes) -> ZernioError:
    """Turn a Zernio error body into a message worth reading.

    Validation failures arrive as an `issues` array; without unpacking it the
    message is just "Bad Request" and you're guessing which field was wrong.
    """
    try:
        data = json.loads(body or b"null") or {}
    except json.JSONDecodeError:
        snippet = body[:400].decode("utf-8", "replace").replace("\n", " ")
        return ZernioError(f"invalid response from Zernio (HTTP {status}): {snippet}", status)

    message = data.get("message") or data.get("error") or str(data)
    if isinstance(message, dict):
        message = message.get("message") or str(message)

    issues = data.get("issues")
    if isinstance(issues, list) and issues:
        parts = []
        for item in issues[:5]:
            if isinstance(item, dict):
                path = ".".join(str(p) for p in (item.get("path") or []))
                detail = item.get("message") or item.get("code") or item
                parts.append(f"{path}: {detail}" if path else str(detail))
            else:
                parts.append(str(item))
        message = f"{message} ({'; '.join(parts)})"

    if status == 401:
        message += " — check ZERNIO_API_KEY (no stray quotes or whitespace) and that it is still valid"

    return ZernioError(f"Zernio error: {message}", status, str(message))


def build_post_body(
    *,
    caption: str,
    accounts: dict[str, str],
    media_url: str | None,
    media_type: str = "video",
    profile_id: str | None = None,
    as_reel: bool = True,
    cover_frame_ms: int | None = None,
) -> dict[str, Any]:
    """Assemble the POST /posts payload.

    Pure, so the payload shape is asserted on in tests rather than discovered
    from a 400 halfway through a posting run.
    """
    platforms: list[dict[str, Any]] = []
    for platform, account_id in accounts.items():
        entry: dict[str, Any] = {"platform": platform, "accountId": account_id}
        if platform == "instagram" and as_reel:
            specific: dict[str, Any] = {"contentType": "reels", "shareToFeed": True}
            if cover_frame_ms is not None:
                specific["thumbOffset"] = int(cover_frame_ms) * 1000
            entry["platformSpecificData"] = specific
        platforms.append(entry)

    if not platforms:
        raise ZernioError("no connected Zernio accounts for the requested platforms")

    missing_media = set(accounts) & PLATFORMS_REQUIRING_MEDIA
    if missing_media and not media_url:
        raise ZernioError(
            f"{', '.join(sorted(missing_media))} require media; the shot has no render"
        )

    # Zernio rejects empty content.
    body: dict[str, Any] = {
        "content": caption.strip() or " ",
        "platforms": platforms,
        "publishNow": True,
    }
    if profile_id:
        body["profileId"] = profile_id
    if media_url:
        body["mediaItems"] = [{"url": media_url, "type": media_type}]
    return body


def summarise_delivery(post: dict[str, Any]) -> dict[str, str]:
    """Flatten a Zernio post's per-platform rows into {platform: status}."""
    statuses: dict[str, str] = {}
    for row in post.get("platforms") or []:
        if isinstance(row, dict):
            platform = str(row.get("platform") or "?")
            status = str(row.get("status") or "").lower() or "unknown"
            error = str(row.get("error") or "").strip()
            statuses[platform] = f"{status}: {error}" if error else status
    return statuses


class ZernioClient:
    """Talks to Zernio over the four endpoints this pipeline needs."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        profile_id: str | None = None,
    ):
        self.api_key = api_key if api_key is not None else read_api_key()
        self.base_url = (
            base_url or os.environ.get("ZERNIO_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.profile_id = profile_id or os.environ.get("ZERNIO_PROFILE_ID") or None
        self._accounts: dict[str, str] | None = None

    # -- transport ---------------------------------------------------------

    def _call(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        timeout: float = TIMEOUT,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode() if payload is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read() or b"null") or {}
        except urllib.error.HTTPError as exc:
            raise parse_error(exc.code, exc.read()) from exc
        except urllib.error.URLError as exc:
            raise ZernioError(f"Zernio unreachable: {exc.reason}") from exc

    # -- accounts ----------------------------------------------------------

    def accounts(self, refresh: bool = False) -> dict[str, str]:
        """{platform: zernio_account_id} for the connected accounts.

        Connecting an account is an interactive OAuth redirect
        (`GET /connect/{platform}`), so it happens in the web app, not here —
        this pipeline only reads what is already connected.
        """
        if self._accounts is None or refresh:
            data = self._call("GET", "/accounts")
            rows = data.get("accounts") if isinstance(data, dict) else data
            resolved: dict[str, str] = {}
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                platform = str(row.get("platform") or "").strip().lower()
                account_id = row.get("_id") or row.get("id") or row.get("accountId")
                if platform and account_id:
                    resolved[platform] = str(account_id)
            self._accounts = resolved
        return dict(self._accounts)

    # -- media -------------------------------------------------------------

    def upload_media(self, file_path: str | Path) -> str:
        """Presign, PUT the bytes, return the public URL for `mediaItems`."""
        file_path = Path(file_path)
        content_type = content_type_for(file_path.name)

        presigned = self._call(
            "POST", "/media/presign",
            payload={"filename": file_path.name, "contentType": content_type},
        )
        upload_url = presigned.get("uploadUrl") or presigned.get("upload_url")
        public_url = presigned.get("publicUrl") or presigned.get("public_url")
        if not upload_url or not public_url:
            raise ZernioError("Zernio presign response missing uploadUrl or publicUrl")

        # The PUT goes to signed object storage, not the API — no auth header.
        request = urllib.request.Request(
            upload_url,
            data=file_path.read_bytes(),
            method="PUT",
            headers={"Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(request, timeout=UPLOAD_TIMEOUT) as response:
                if response.status >= 400:
                    raise ZernioError(f"upload failed (HTTP {response.status})")
        except urllib.error.HTTPError as exc:
            raise ZernioError(
                f"upload to Zernio storage failed (HTTP {exc.code})",
                exc.code, exc.read()[:500].decode("utf-8", "replace"),
            ) from exc

        return str(public_url)

    # -- posts -------------------------------------------------------------

    def publish(
        self,
        video_path: str | Path,
        caption: str,
        *,
        platforms: Sequence[str] | None = None,
        cover_frame_ms: int | None = None,
    ) -> dict[str, Any]:
        """Upload the render and publish it now. Returns the Zernio response."""
        video_path = Path(video_path)
        connected = self.accounts()
        wanted = (
            {p: connected[p] for p in platforms if p in connected}
            if platforms
            else connected
        )
        if not wanted:
            raise ZernioError(
                "no connected Zernio accounts"
                + (f" for {', '.join(platforms)}" if platforms else "")
            )

        media_url = self.upload_media(video_path)
        body = build_post_body(
            caption=caption,
            accounts=wanted,
            media_url=media_url,
            media_type=media_item_type(video_path.name),
            profile_id=self.profile_id,
            cover_frame_ms=cover_frame_ms,
        )
        return self._call("POST", "/posts", payload=body, timeout=UPLOAD_TIMEOUT)

    def fetch_post(self, post_id: str) -> dict[str, Any] | None:
        """GET /posts/{id}, unwrapping the `post` envelope Zernio uses."""
        try:
            data = self._call("GET", f"/posts/{urllib.parse.quote(post_id)}")
        except ZernioError as exc:
            log.warning("fetching Zernio post %s failed: %s", post_id, exc)
            return None
        post = data.get("post")
        if isinstance(post, dict):
            return post
        return data if data.get("_id") else None

    def post_logs(self, post_id: str) -> list[dict[str, Any]]:
        """Publishing logs — the fallback when GET /posts is unavailable."""
        try:
            data = self._call("GET", "/logs", params={"postId": post_id})
        except ZernioError as exc:
            log.warning("fetching Zernio logs for %s failed: %s", post_id, exc)
            return []
        return data.get("logs") or []
