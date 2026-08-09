"""Stage 1b — split each source into shots.

Drone footage is mostly one continuous move per file, so the bias here is
towards keeping a file whole: the detector threshold is high, sub-minimum
segments are merged into their neighbours rather than kept, and a file with no
detected cut becomes a single shot flagged `continuous_move`.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from .. import db
from ..config import Config, config as default_config

log = logging.getLogger(__name__)

Segment = tuple[float, float]


def merge_short_segments(segments: list[Segment], min_seconds: float) -> list[Segment]:
    """Absorb segments below the minimum into the neighbour they came from.

    A 1-second sliver is a detector artifact, not a shot. Merging preserves the
    file's full timeline — every second stays addressable in exactly one shot.
    """
    if not segments:
        return []

    merged: list[Segment] = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if (prev_end - prev_start) < min_seconds:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # The tail can still be short after the sweep; fold it back if so.
    if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_seconds:
        start, end = merged.pop()
        merged[-1] = (merged[-1][0], end)
    return merged


def detect_segments(
    path: str | Path, duration: float, cfg: Config = default_config
) -> list[Segment]:
    """Content-aware cut list, or one whole-file segment when nothing is found.

    Falls back to a single segment if PySceneDetect is not installed, so Stage 1
    still produces addressable shots on a bare checkout.
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError:
        log.warning("scenedetect not installed — treating %s as one shot", path)
        return [(0.0, duration)]

    video = open_video(str(path))
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(
            threshold=cfg.scene_threshold,
            min_scene_len=int(cfg.min_shot_seconds * video.frame_rate),
        )
    )
    manager.detect_scenes(video, show_progress=False)
    scenes = manager.get_scene_list()

    if not scenes:
        return [(0.0, duration)]

    segments = [(start.get_seconds(), end.get_seconds()) for start, end in scenes]
    # Detection can stop short of the container duration; keep the tail.
    if duration - segments[-1][1] > 0.5:
        segments[-1] = (segments[-1][0], duration)
    return merge_short_segments(segments, cfg.min_shot_seconds)


def run(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
) -> dict[str, int]:
    """Segment every source that has not been through detection yet."""
    counts = {"sources": 0, "shots": 0, "continuous": 0, "failed": 0}

    pending = db.sources_pending_scene_detection(conn)
    for source in pending[:limit] if limit else pending:
        if not source["duration"]:
            log.warning("skipping %s: no duration probed", source["relpath"])
            counts["failed"] += 1
            continue
        try:
            segments = detect_segments(source["path"], source["duration"], cfg)
        except Exception as exc:  # detector failures must not stall the archive
            log.warning("scene detection failed for %s: %s", source["relpath"], exc)
            counts["failed"] += 1
            continue

        db.replace_shots(conn, source["id"], segments)
        counts["sources"] += 1
        counts["shots"] += len(segments)
        counts["continuous"] += 1 if len(segments) == 1 else 0
        log.info("%s -> %d shot(s)", source["relpath"], len(segments))

    return counts
