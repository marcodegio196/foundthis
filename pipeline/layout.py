"""Where files land on disk.

Everything the pipeline writes goes into a folder tree that mirrors the archive
convention already in use (`archive/<country>/<file>`), so renders and exports
are browsable in Finder or Explorer without the database. A flat output folder
would not survive 250GB across ten countries.

Raw footage is never written to, moved, or deleted — these paths only ever
describe *new* files under `render_root` and `export_root`.
"""

from __future__ import annotations

import re
from pathlib import Path

UNKNOWN_COUNTRY = "unknown-country"
UNDATED = "undated"

# Windows forbids these outright; they also make paths miserable to type on a
# shell. Applied to every path component derived from metadata.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def slugify(value: str | None, fallback: str) -> str:
    """Make one path component safe on every platform we might run on."""
    text = _UNSAFE.sub("", (value or "").strip()).strip(". ")
    text = re.sub(r"\s+", "-", text).strip("-").lower()
    if not text:
        return fallback
    if text in _WINDOWS_RESERVED:
        return f"{text}-"
    return text[:64]


def year_of(captured_at: str | None) -> str:
    """Capture year from an ISO-ish timestamp, or 'undated'.

    Undated footage sorts into its own folder rather than a guessed year — a
    licensing buyer filtering by date is better served by an honest gap.
    """
    prefix = (captured_at or "")[:4]
    return prefix if prefix.isdigit() else UNDATED


def shot_dir(country: str | None, captured_at: str | None) -> Path:
    """`<country>/<year>` — the shared shape for renders and exports."""
    return Path(slugify(country, UNKNOWN_COUNTRY)) / year_of(captured_at)


def shot_stem(shot_id: int, source_relpath: str, site: str | None = None) -> str:
    """Filename stem: zero-padded id, source name, and site when known.

    The id leads so the filename sorts stably and maps back to a database row;
    the source name is kept so a file found on disk can be traced to the
    original clip without a query.
    """
    source = slugify(Path(source_relpath).stem, "clip")
    parts = [f"{shot_id:06d}", source]
    if site:
        parts.append(slugify(site, ""))
    return "_".join(part for part in parts if part)


def render_path(
    root: Path,
    *,
    shot_id: int,
    source_relpath: str,
    country: str | None,
    captured_at: str | None,
    site: str | None = None,
    suffix: str = ".mp4",
) -> Path:
    """Full output path for one rendered shot."""
    return root / shot_dir(country, captured_at) / (
        shot_stem(shot_id, source_relpath, site) + suffix
    )
