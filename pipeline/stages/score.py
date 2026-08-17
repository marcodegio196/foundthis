"""Stage 2 — score every shot, then cut the bottom percentile.

Two passes, deliberately separate:

1. `score_pending` measures each shot on its own and writes the numbers.
2. `apply_rejections` compares shots against each other and flags the bottom.

Splitting them means the bar can be re-drawn at any time (a different
percentile, a different weighting) without re-decoding a frame — and re-drawing
it is expected, since "the usable 30-40%" is a judgement, not a constant.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import db, media, motion, scoring
from ..aesthetic import AestheticScorer, load_scorer, null_scorer
from ..config import Config, config as default_config

log = logging.getLogger(__name__)

# How often to report progress on a long run. Scoring a full archive is an
# hours-long job; silence for that long is indistinguishable from a hang.
PROGRESS_EVERY = 25


def analyse_frames(frame_paths: list[Path]) -> tuple[list[scoring.FrameFlow], list[float], list[float], int]:
    """Optical flow, sharpness, and clipping over a shot's sampled frames.

    Returns (flows, sharpness per frame, clipped fraction per frame, frame width).
    """
    import cv2
    import numpy as np

    flows: list[scoring.FrameFlow] = []
    sharpness: list[float] = []
    clipped: list[float] = []
    previous = None
    width = 0

    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        width = frame.shape[1]

        sharpness.append(float(cv2.Laplacian(frame, cv2.CV_64F).var()))
        crushed = float(np.count_nonzero(frame <= 4) + np.count_nonzero(frame >= 251))
        clipped.append(crushed / frame.size)

        if previous is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous, frame, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flows.append(
                scoring.FrameFlow(
                    dx=float(flow[..., 0].mean()),
                    dy=float(flow[..., 1].mean()),
                    magnitude=float(magnitude.mean()),
                )
            )
        previous = frame

    return flows, sharpness, clipped, width


def score_shot(
    row: db.Row,
    cfg: Config = default_config,
    aesthetic: AestheticScorer = null_scorer,
) -> scoring.Scores:
    """Measure one shot. Samples frames once and reuses them for every component."""
    timestamps = media.sample_timestamps(
        row["in_point"], row["out_point"], cfg.sample_fps
    )
    if not timestamps:
        return scoring.Scores()

    with tempfile.TemporaryDirectory(prefix="shot-") as tmp:
        # Frames are extracted at a width the flow analysis is happy with; the
        # aesthetic model re-reads the same JPEGs and does its own resizing.
        frame_paths = media.extract_frames(row["source_path"], timestamps, tmp, width=512)
        if not frame_paths:
            return scoring.Scores()

        flows, sharpness, clipped, width = analyse_frames(frame_paths)
        aesthetic_value = aesthetic(frame_paths)

    return scoring.Scores(
        motion=scoring.motion_score(flows, width),
        stability=scoring.stability_score(flows, width),
        technical=scoring.technical_score(sharpness, clipped),
        aesthetic=aesthetic_value,
    )


def _persist(conn: sqlite3.Connection, shot_id: int, scores: scoring.Scores) -> None:
    with db.transaction(conn):
        db.update_shot(
            conn,
            shot_id,
            motion_score=scores.motion,
            stability_score=scores.stability,
            technical_score=scores.technical,
            aesthetic_score=scores.aesthetic,
            scored_at=db.now(),
        )


def score_pending(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
    aesthetic: AestheticScorer | None = None,
    workers: int | None = None,
) -> dict[str, int]:
    """Pass 1 — measure every shot that has no scores yet.

    Each shot is independent, and the expensive parts (an ffmpeg subprocess per
    shot, then optical flow inside OpenCV) both spend their time outside the
    interpreter lock, so threads give real parallelism here without the cost of
    handing rows to separate processes.

    Measurement happens on the workers; **every database write happens on this
    thread**. SQLite connections are not shared safely across threads, and
    serialising the writes costs nothing next to decoding 4K frames.
    """
    aesthetic = aesthetic if aesthetic is not None else load_scorer()
    workers = workers if workers is not None else cfg.score_workers
    rows = db.shots_pending_scoring(conn, limit=limit)
    counts = {"scored": 0, "failed": 0}
    if not rows:
        return counts

    total = len(rows)
    started = time.monotonic()

    def record(row: db.Row, scores: scoring.Scores | None, error: Exception | None) -> None:
        if error is not None:  # one unreadable file must not stall the pass
            log.warning("scoring failed for shot %s: %s", row["id"], error)
            counts["failed"] += 1
            return
        _persist(conn, row["id"], scores)
        counts["scored"] += 1
        log.info(
            "shot %s: motion=%s stability=%s technical=%s aesthetic=%s",
            row["id"], scores.motion, scores.stability, scores.technical, scores.aesthetic,
        )
        done = counts["scored"] + counts["failed"]
        if done % PROGRESS_EVERY == 0 or done == total:
            rate = done / max(time.monotonic() - started, 1e-6)
            remaining = (total - done) / rate if rate else 0
            log.warning(  # deliberately at warning so it shows without -v
                "scored %d/%d shots (%.1f/min, ~%d min left)",
                done, total, rate * 60, round(remaining / 60),
            )

    if workers <= 1:
        for row in rows:
            try:
                record(row, score_shot(row, cfg, aesthetic), None)
            except Exception as exc:
                record(row, None, exc)
        return counts

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(score_shot, row, cfg, aesthetic): row for row in rows}
        for future in as_completed(pending):
            row = pending[future]
            try:
                record(row, future.result(), None)
            except Exception as exc:
                record(row, None, exc)

    return counts


def apply_rejections(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Pass 2 — flag hard-floor failures and the bottom percentile.

    Recomputed from scratch every run: previously rejected shots are cleared
    first, so lowering the bar genuinely brings footage back rather than
    requiring a rescore.

    Repositioning is excluded throughout. That rejection comes from stage 1b and
    describes what the camera was doing, not how the footage scored, so clearing
    it here would quietly hand the jerk between two good moves back to the
    selection queue. Excluding it from the population matters too: a reposition
    has no motion or stability score, so it would otherwise be ranked on
    technical quality alone — and a sharp, well-exposed frame of the camera
    searching scores *above* the bar.
    """
    rows = conn.execute(
        "SELECT id, motion_score, stability_score, technical_score, aesthetic_score "
        "FROM shots WHERE scored_at IS NOT NULL "
        "AND (motion_class IS NULL OR motion_class != ?)",
        (motion.REPOSITION,),
    ).fetchall()
    if not rows:
        return {"considered": 0, "rejected": 0, "threshold": None}

    combined: dict[int, float] = {}
    floor_failures: dict[int, str] = {}
    for row in rows:
        scores = scoring.Scores(
            motion=row["motion_score"],
            stability=row["stability_score"],
            technical=row["technical_score"],
            aesthetic=row["aesthetic_score"],
        )
        reason = scoring.fails_hard_floor(scores)
        if reason:
            floor_failures[row["id"]] = reason
            continue
        value = scoring.combine(scores)
        if value is not None:
            combined[row["id"]] = value

    threshold = scoring.percentile_threshold(
        list(combined.values()), cfg.reject_percentile
    )
    below = (
        {shot_id for shot_id, value in combined.items() if value < threshold}
        if threshold is not None
        else set()
    )

    result = {
        "considered": len(rows),
        "rejected": len(below) + len(floor_failures),
        "threshold": round(threshold, 4) if threshold is not None else None,
        "floor_failures": len(floor_failures),
    }
    if dry_run:
        return result

    with db.transaction(conn):
        conn.execute(
            "UPDATE shots SET rejected = 0, reject_reason = NULL "
            "WHERE scored_at IS NOT NULL AND (motion_class IS NULL OR motion_class != ?)",
            (motion.REPOSITION,),
        )
    db.reject(conn, below, f"bottom {cfg.reject_percentile:.0%} (< {threshold:.3f})")
    for shot_id, reason in floor_failures.items():
        db.reject(conn, [shot_id], reason)

    log.info(
        "rejected %d of %d scored shots (threshold %.3f)",
        result["rejected"], len(rows), threshold or 0.0,
    )
    return result


def run(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
    aesthetic: AestheticScorer | None = None,
    workers: int | None = None,
) -> dict[str, object]:
    scored = score_pending(
        conn, cfg, limit=limit, aesthetic=aesthetic, workers=workers
    )
    return {**scored, **apply_rejections(conn, cfg)}
