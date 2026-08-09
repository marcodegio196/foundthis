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


@dataclass(frozen=True)
class Config:
    # Stage 1
    archive_root: Path = _path("ARCHIVE_ROOT", "./archive")
    db_path: Path = _path("PIPELINE_DB", "./pipeline.db")
    video_extensions: tuple[str, ...] = (".mp4", ".mov", ".mxf", ".insv")

    # Stage 1b — content-aware detection. A drone push-in is one continuous
    # move, so the threshold is deliberately high: we would rather keep a file
    # whole than chop a slow reveal into pieces.
    scene_threshold: float = _float("SCENE_THRESHOLD", 32.0)
    min_shot_seconds: float = _float("MIN_SHOT_SECONDS", 3.0)

    # Stage 2 — bottom percentile is flagged, never deleted.
    reject_percentile: float = _float("REJECT_PERCENTILE", 0.35)
    sample_fps: float = _float("SAMPLE_FPS", 1.0)

    # Stage 4
    render_root: Path = _path("RENDER_ROOT", "./renders")
    overlay_text: str = os.environ.get("OVERLAY_TEXT", "Found this.")

    @property
    def social_render_dir(self) -> Path:
        return self.render_root / "social"

    @property
    def licensing_render_dir(self) -> Path:
        return self.render_root / "licensing"


config = Config()
