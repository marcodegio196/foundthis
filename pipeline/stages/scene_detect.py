"""Stage 1b — split each source into shots.

Two passes, because a drone file has two kinds of boundary.

**Cuts.** Content-aware detection finds hard transitions — separate recordings
in one file. The threshold is deliberately high and sub-minimum segments are
merged away: a slow reveal is one shot, not a dozen.

**Camera behaviour.** Most drone files contain no cuts at all. One continuous
recording holds a good move, a fast reposition to find the next composition,
then another good move. Those are not cuts, so the detector cannot see them,
and a file left whole is scored as the average of its best and worst parts —
which usually puts it below the bar and throws the good footage away with the
bad. `pipeline/motion.py` reads what the camera is doing and splits on it, so
each usable stretch becomes its own shot.

Repositioning is flagged rejected as it is created, never deleted. The footage
stays addressable in case the bar moves later.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from pathlib import Path

from .. import db, media, motion
from ..config import Config, config as default_config

log = logging.getLogger(__name__)

Segment = tuple[float, float]


def _seconds(timecode) -> float:
    """Read a PySceneDetect timecode across versions.

    0.7 exposes a `seconds` property and deprecates `get_seconds()`; 0.6 only
    has the method.
    """
    value = getattr(timecode, "seconds", None)
    return float(value) if value is not None else float(timecode.get_seconds())


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

    segments = [(_seconds(start), _seconds(end)) for start, end in scenes]
    # Detection can stop short of the container duration; keep the tail.
    if duration - segments[-1][1] > 0.5:
        segments[-1] = (segments[-1][0], duration)
    return merge_short_segments(segments, cfg.min_shot_seconds)


def profile_motion(
    path: str | Path, start: float, end: float, cfg: Config = default_config
) -> tuple[list[motion.Sample], int]:
    """Sample a stretch of a file and measure what the camera is doing."""
    timestamps = media.sample_timestamps(
        start, end, cfg.motion_sample_fps, max_frames=cfg.motion_max_frames
    )
    if len(timestamps) < 2:
        return [], 0
    with tempfile.TemporaryDirectory(prefix="motion-") as tmp:
        # 256px, matching `media.iter_gray_frames`. Phase correlation needs a
        # clean peak, and specular sparkle on water is high-frequency noise that
        # buries it: measured over a sunlit sea, confidence at 512px was 0.09 —
        # under MIN_RESPONSE — so a steady move (jitter 0.03) was read as
        # "too chaotic to correlate" and rejected as repositioning. Decimating
        # to 256px is a low-pass that removes the sparkle and not the features:
        # confidence 0.09 -> 0.21 on water, 0.30 -> 0.54 on rock, with the
        # measured shift unchanged. Gaussian blur does not work here — it smears
        # the structure the correlation needs and inflates jitter to 0.68.
        frames = media.extract_frames(path, timestamps, tmp, width=256)
        # A frame that failed to extract shifts every later timestamp, so only
        # measure when the two line up.
        if len(frames) != len(timestamps):
            timestamps = timestamps[: len(frames)]
        return motion.measure(frames, timestamps)


def carve_windows(
    run: motion.Run, samples: list[motion.Sample], frame_width: int, cfg: Config
) -> list[tuple[float, float, str, list[float], bool]]:
    """Split one usable run into its steady stretches, keeping the leftovers.

    Stage 4 used to render from a run's in-point, which on a long take ships
    whatever is at the front. A 21-second take began with a stop-and-go and
    ended calm; anchored rendering kept the stop-and-go and dropped the calm.
    Carving here means each steady stretch becomes its own shot, so a take
    holding two good sections yields two clips instead of one compromised one.

    Leftovers are emitted as their own segments and marked `is_window=False` —
    the seconds stay addressable if the window rules change later, but they do
    not reach a render. A run that contains no steady window at all is kept
    whole and marked the same way: better to publish nothing from it than to
    ship the lurch because the run happened to be long enough.

    Returns `(start, end, motion_class, speed_profile, is_window)`.
    """
    profile = motion.speed_buckets(samples, cfg.motion_sample_fps, frame_width)
    # A run too short to hold a window is not "outside" one — it is good
    # footage that does not fit the target length, which is stock inventory,
    # not a reject. Only runs that could have held a window are judged on
    # whether they do.
    carving = bool(
        cfg.window_selection
        and profile
        and cfg.min_render_seconds > 0
        and run.duration >= cfg.min_render_seconds
    )
    # With carving off there is nothing to be outside of, so the run stays
    # renderable and behaves as it did before windows existed.
    whole = (run.start, run.end, run.motion_class, profile, not carving)
    if not carving:
        return [whole]

    windows = motion.find_windows(
        profile,
        start=run.start,
        min_seconds=cfg.min_render_seconds,
        max_seconds=cfg.max_render_seconds or run.duration,
        max_spread=cfg.window_max_spread,
    )
    if not windows:
        return [whole]

    def slice_profile(start: float, end: float) -> list[float]:
        lo = int(round((start - run.start) / motion.SPEED_BUCKET_SECONDS))
        hi = int(round((end - run.start) / motion.SPEED_BUCKET_SECONDS))
        return profile[max(0, lo) : max(0, hi)]

    # Buckets are whole seconds, so a window always stops on a second boundary
    # and can leave up to a bucket of the run unclaimed. Anything under one
    # bucket is absorbed into the window rather than emitted on its own: it is
    # too short to be a shot, and across an archive it would otherwise leave
    # thousands of sub-second rows behind.
    sliver = motion.SPEED_BUCKET_SECONDS

    out: list[tuple[float, float, str, list[float], bool]] = []
    cursor = run.start
    for win in windows:
        if win.start - cursor > sliver:
            out.append(
                (cursor, win.start, run.motion_class, slice_profile(cursor, win.start), False)
            )
            cursor = win.start
        end = min(win.end, run.end)
        # Absorb a short tail into this window rather than emitting it alone.
        if run.end - end <= sliver:
            end = run.end
        out.append((cursor, end, run.motion_class, slice_profile(cursor, end), True))
        cursor = end
    if run.end - cursor > sliver:
        out.append(
            (cursor, run.end, run.motion_class, slice_profile(cursor, run.end), False)
        )
    return out


def split_by_motion(
    path: str | Path, segments: list[Segment], cfg: Config = default_config
) -> list[tuple[float, float, str, list[float]]]:
    """Sub-divide each scene by what the camera is doing inside it.

    This is what makes a file containing two good moves separated by a
    reposition become two usable shots plus a flagged one, rather than a single
    shot whose score is the average of all three.

    Each usable run is then carved into the stretches that hold a steady speed,
    so a take with a lurch at the front does not have to be shipped with it.
    Every segment carries the speed curve it was measured on.
    """
    out: list[tuple[float, float, str, list[float], bool]] = []
    for start, end in segments:
        try:
            samples, width = profile_motion(path, start, end, cfg)
        except Exception as exc:
            log.warning("motion profiling failed for %s (%.1f-%.1f): %s", path, start, end, exc)
            samples, width = [], 0

        runs = motion.segment(
            samples,
            width,
            cfg.motion_sample_fps,
            cfg.min_shot_seconds,
            min_reposition_seconds=cfg.min_reposition_seconds,
            max_rate=cfg.motion_max_rate,
        )
        if not runs:
            # Nothing measurable — keep the scene whole rather than dropping it.
            out.append((start, end, motion.MOVE, [], True))
            continue

        # Sampling starts half an interval in, so stretch the ends back out to
        # the scene boundaries; every second must land in exactly one shot.
        runs[0] = motion.Run(start, runs[0].end, runs[0].motion_class)
        runs[-1] = motion.Run(runs[-1].start, end, runs[-1].motion_class)

        for run in runs:
            inside = [s for s in samples if run.start <= s.midpoint <= run.end]
            if run.motion_class == motion.REPOSITION:
                # Repositioning is discarded whole; carving it would only
                # produce steadier-looking pieces of a hunt.
                out.append(
                    (run.start, run.end, run.motion_class,
                     motion.speed_buckets(inside, cfg.motion_sample_fps, width), True)
                )
                continue
            out.extend(carve_windows(run, inside, width, cfg))
    return out


def run(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
    profile: bool | None = None,
) -> dict[str, int]:
    """Segment every source that has not been through detection yet."""
    profile = cfg.motion_segmentation if profile is None else profile
    counts = {
        "sources": 0, "shots": 0, "usable": 0, "reposition": 0,
        "continuous": 0, "failed": 0,
    }

    pending = db.sources_pending_scene_detection(conn)
    for source in pending[:limit] if limit else pending:
        if not source["duration"]:
            log.warning("skipping %s: no duration probed", source["relpath"])
            counts["failed"] += 1
            continue
        try:
            scenes = detect_segments(source["path"], source["duration"], cfg)
            if profile:
                segments = split_by_motion(source["path"], scenes, cfg)
            else:
                segments = [(a, b, motion.MOVE) for a, b in scenes]
        except Exception as exc:  # detector failures must not stall the archive
            log.warning("scene detection failed for %s: %s", source["relpath"], exc)
            counts["failed"] += 1
            continue

        shot_ids = db.replace_shots(conn, source["id"], segments)

        # Repositioning is flagged, never deleted — the footage stays addressable
        # in case the bar moves or a frame of it turns out to be worth keeping.
        rejects = [
            shot_id
            for shot_id, segment in zip(shot_ids, segments)
            if segment[2] == motion.REPOSITION
        ]
        if rejects:
            db.reject(conn, rejects, "repositioning: camera searching, not composed")

        # Everything outside a steady window is flagged the same way: kept on
        # disk and addressable, but not offered up for a render. A long run
        # whose speed never settles produces nothing rather than being shipped
        # because it cleared the length floor.
        outside = [
            shot_id
            for shot_id, segment in zip(shot_ids, segments)
            if segment[2] != motion.REPOSITION and len(segment) > 4 and not segment[4]
        ]
        if outside:
            db.reject(
                conn, outside,
                f"outside a steady {cfg.min_render_seconds:g}s window",
            )

        counts["sources"] += 1
        counts["shots"] += len(segments)
        counts["reposition"] += len(rejects)
        counts["usable"] += len(segments) - len(rejects)
        counts["continuous"] += 1 if len(segments) == 1 else 0
        log.info(
            "%s -> %d shot(s), %d usable",
            source["relpath"], len(segments), len(segments) - len(rejects),
        )

    return counts
