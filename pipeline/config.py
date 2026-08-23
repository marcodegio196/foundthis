"""Pipeline configuration.

Everything is overridable by environment variable so the same code runs against
the real 250GB archive and a small test folder without editing source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: the DB layer and CLI work with no third-party deps installed
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> None:
        return None

load_dotenv()


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    # Stage 1
    # Raw footage in, renders out. `input/<country>/<file>` — the country folder
    # is what stage 1 reads the country from, so files dropped in the root sort
    # to "unknown-country".
    archive_root: Path = _path("ARCHIVE_ROOT", "./input")
    db_path: Path = _path("PIPELINE_DB", "./pipeline.db")
    video_extensions: tuple[str, ...] = (".mp4", ".mov", ".mxf", ".insv")

    # Stage 1b — content-aware detection. A drone push-in is one continuous
    # move, so the threshold is deliberately high: we would rather keep a file
    # whole than chop a slow reveal into pieces.
    scene_threshold: float = _float("SCENE_THRESHOLD", 32.0)
    min_shot_seconds: float = _float("MIN_SHOT_SECONDS", 3.0)

    # Split each scene by what the camera is doing, so a file holding two good
    # moves either side of a reposition becomes two usable shots rather than one
    # averaged-out shot. Set MOTION_SEGMENTATION=0 to cut only on hard cuts.
    motion_segmentation: bool = os.environ.get("MOTION_SEGMENTATION", "1") not in ("0", "false", "")
    # Sampling rate for the profile. Higher tracks faster movement but costs a
    # decode per sample across the whole archive.
    motion_sample_fps: float = _float("MOTION_SAMPLE_FPS", 4.0)
    motion_max_frames: int = _int("MOTION_MAX_FRAMES", 600)
    # A reposition only has to last about a second to be worth cutting out;
    # judging it by min_shot_seconds would fold it back into the shot beside it.
    min_reposition_seconds: float = _float("MIN_REPOSITION_SECONDS", 1.0)
    # Speed ceiling in frame widths per second: above this a move is too fast
    # to hold a composition even if it runs perfectly straight. Generous by
    # default — the jitter test does most of the work.
    motion_max_rate: float = _float("MOTION_MAX_RATE", 1.0)

    # Sources segmented in parallel. Each worker drives ffmpeg and OpenCV, so
    # the work happens outside the interpreter lock; the limit is cores and how
    # fast the drive feeds them. Lower it if the archive lives on a spinning or
    # network disk, where the decodes contend rather than overlap.
    segment_workers: int = _int("SEGMENT_WORKERS", min(8, (os.cpu_count() or 2)))

    # Stage 2 — bottom percentile is flagged, never deleted.
    reject_percentile: float = _float("REJECT_PERCENTILE", 0.35)
    sample_fps: float = _float("SAMPLE_FPS", 1.0)
    # Shots scored in parallel. Each worker runs an ffmpeg subprocess and an
    # optical-flow pass, so this is bounded by cores and by how fast the drive
    # holding the archive can feed them — past ~8 an external disk is usually
    # the limit, not the CPU.
    score_workers: int = _int("SCORE_WORKERS", min(8, (os.cpu_count() or 2)))

    # Stage 4 — bounds on the length of a rendered clip, in seconds. 0 disables
    # either end. A shot shorter than the minimum is not rendered at all; one
    # longer than the maximum is trimmed to it from the in-point, so a good
    # 40-second hold still yields a postable clip rather than being skipped.
    min_render_seconds: float = _float("MIN_RENDER_SECONDS", 0.0)
    max_render_seconds: float = _float("MAX_RENDER_SECONDS", 0.0)

    # Stage 1b — carve each usable run into the stretches that actually hold a
    # steady speed, rather than taking whatever sits at the front of it. A long
    # take can hold more than one, and each becomes its own shot; the leftovers
    # around them stay addressable instead of being deleted.
    # Set WINDOW_SELECTION=0 to keep runs whole.
    window_selection: bool = os.environ.get("WINDOW_SELECTION", "1") not in ("0", "false", "")
    # Widest ratio between the fastest and slowest second inside a window. A
    # reveal that starts from a hover is exempt — it necessarily has a huge
    # ratio, and it is the shape of a deliberate reveal, not a lurch.
    window_max_spread: float = _float("WINDOW_MAX_SPREAD", 3.0)

    render_root: Path = _path("RENDER_ROOT", "./renders")
    overlay_text: str = os.environ.get("OVERLAY_TEXT", "Found this.")
    # ffmpeg's drawtext needs an explicit font file on most Linux builds; on
    # macOS it falls back to the system font when this is unset.
    overlay_font: str | None = os.environ.get("OVERLAY_FONT") or None
    licensing_codec: str = os.environ.get("LICENSING_CODEC", "libx264")

    # Stage 4 — the selection queue is shuffled rather than ranked, so the feed
    # spreads the whole archive across the calendar instead of working through
    # the best country first. Set SELECTION_SEED to make that shuffle repeatable
    # when comparing two runs; leave it unset for a different order every time.
    selection_seed: int | None = (
        int(os.environ["SELECTION_SEED"]) if os.environ.get("SELECTION_SEED") else None
    )

    # Stage 5 — Zernio credentials are read by pipeline/zernio.py from
    # ZERNIO_API_KEY / ZERNIO_BASE_URL / ZERNIO_PROFILE_ID, matching the names
    # the new-visu app already uses so one .env serves both.
    social_platforms: tuple[str, ...] = tuple(
        p.strip() for p in os.environ.get("SOCIAL_PLATFORMS", "instagram,tiktok").split(",") if p.strip()
    )
    # Default spacing between scheduled posts. `schedule` uses this when no
    # --interval-hours is given on the CLI.
    post_interval_hours: float = _float("POST_INTERVAL_HOURS", 24.0)

    # Stage 4 outputs and stage 6 exports all stay on local disk. Nothing in the
    # pipeline ever deletes or moves a file under archive_root.
    export_root: Path = _path("EXPORT_ROOT", "./exports")

    @property
    def social_render_dir(self) -> Path:
        return self.render_root / "social"

    @property
    def licensing_render_dir(self) -> Path:
        return self.render_root / "licensing"


config = Config()
