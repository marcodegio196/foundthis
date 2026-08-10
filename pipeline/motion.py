"""Motion profiling — what the *camera* is doing, second by second.

Scene detection finds cuts. A drone file usually has none: it is one continuous
recording holding a good move, then a fast reposition to find the next
composition, then another good move. Those transitions are not cuts, so the
detector cannot see them, and treating the file as one shot averages the good
and the bad together — the whole thing then scores below either half and gets
thrown away.

This reads camera behaviour directly and splits on it, so each usable move
becomes its own shot and the searching between them is flagged.

Measurement is **phase correlation**, which finds the single dominant shift
between two frames. That choice matters twice over:

*It measures the camera, not the contents.* Mean optical flow is pulled around
by anything moving inside the frame, so a locked-off shot over water or foliage
reads as chaotic and gets discarded. Phase correlation locks onto the global
shift and ignores local motion. Measured over a static camera with a subject
crossing frame, and again over moving water: 0.00 and 0.02 px of camera
movement, correctly still in both cases.

*It stays accurate when the camera moves fast.* Optical flow silently
under-reports displacements it cannot track, so a frantic 1200 px/s reposition
measured a *lower* apparent speed than a 40 px/s pan — classifying on it would
rank the worst footage as the calmest. Phase correlation recovered 8.0 / 30.0 /
80.1 px per sample for 40 / 150 / 400 px/s pans: exact, and monotonic.

Two numbers come out of a window of samples:

    shift   how far the frame moves per sample, averaged
    drift   the length of the *summed* movement over the window

A deliberate move goes one way, so those are equal. Searching doubles back on
itself, so the sum cancels while the per-step distance keeps accumulating.
`jitter = (shift - drift) / shift` is that gap, and it is what separates a pan
from a hunt:

    pan, any speed        jitter 0.00
    reposition            jitter 0.43
    erratic oscillation   jitter 0.59
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

log = logging.getLogger(__name__)

HELD = "held"
MOVE = "move"
REPOSITION = "reposition"

USABLE = frozenset({HELD, MOVE})

# Camera speed, as a fraction of frame width per second, below which the camera
# is parked. A true hold and a hold over moving water both measured under
# 0.0002; the slowest deliberate pan measured 0.06.
HELD_RATE = 0.01

# How much of the movement cancels itself out before it is a hunt rather than a
# move. Deliberate pans measured 0.00 at every speed tested; repositioning 0.43.
JITTER = 0.25

# Speed ceiling, in frame widths per second. Above this a move is too fast to
# hold a composition even if it runs perfectly straight. Generous by default —
# jitter does most of the work, and a fast reveal is a legitimate shot.
MAX_RATE = 1.0

# Phase correlation returns a confidence. Featureless frames (fog, blown sky,
# a black frame) produce a meaningless shift with a low score; treating those
# as held is the safe reading, since there is no evidence of movement.
MIN_RESPONSE = 0.15


@dataclass(frozen=True)
class Sample:
    """The camera's movement between one pair of sampled frames."""

    start: float
    end: float
    dx: float
    dy: float
    response: float

    @property
    def distance(self) -> float:
        return math.hypot(self.dx, self.dy)

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True)
class Window:
    """Aggregated behaviour over a run of samples."""

    shift: float     # mean per-sample distance, px
    drift: float     # length of the summed movement, px
    response: float  # mean phase-correlation confidence

    @property
    def jitter(self) -> float:
        """Fraction of the movement that cancels itself out."""
        if self.shift < 1e-9:
            return 0.0
        return max(0.0, min(1.0, (self.shift - self.drift) / self.shift))


@dataclass(frozen=True)
class Run:
    """A stretch of consecutive samples sharing one classification."""

    start: float
    end: float
    motion_class: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def usable(self) -> bool:
        return self.motion_class in USABLE


def summarise(samples: Sequence[Sample]) -> Window:
    """Collapse a run of samples into shift, drift, and confidence."""
    if not samples:
        return Window(0.0, 0.0, 0.0)
    shift = sum(s.distance for s in samples) / len(samples)
    drift = math.hypot(
        sum(s.dx for s in samples) / len(samples),
        sum(s.dy for s in samples) / len(samples),
    )
    response = sum(s.response for s in samples) / len(samples)
    return Window(shift=shift, drift=drift, response=response)


def classify(
    window: Window,
    frame_width: int,
    sample_fps: float,
    *,
    max_rate: float = MAX_RATE,
) -> str:
    """Label a window of camera behaviour.

    `rate` converts px-per-sample into frame widths per second, so the same
    thresholds hold at any resolution or sampling rate.
    """
    if frame_width <= 0 or sample_fps <= 0:
        return MOVE
    rate = (window.shift * sample_fps) / frame_width

    if window.response < MIN_RESPONSE:
        # Correlation found no clear peak. On a featureless frame — fog, blown
        # sky, black — that means there is nothing to measure, and treating a
        # still camera as parked is the safe reading. But when the frame is
        # also moving a long way, the absence of a peak *is* the finding: the
        # motion is too chaotic to correlate.
        return HELD if rate < HELD_RATE else REPOSITION
    if rate < HELD_RATE:
        return HELD
    if window.jitter > JITTER:
        return REPOSITION
    if rate > max_rate:
        return REPOSITION
    return MOVE


def label_samples(
    samples: Sequence[Sample],
    frame_width: int,
    sample_fps: float,
    *,
    window: int = 5,
    max_rate: float = MAX_RATE,
) -> list[str]:
    """Classify each sample using the behaviour around it.

    Jitter only exists across several samples — a single frame pair cannot
    reveal that the camera doubled back — so each sample is judged on a window
    centred on it. That also smooths the result, so one noisy pair cannot split
    an otherwise steady move.
    """
    half = max(1, window // 2)
    labels = []
    for index in range(len(samples)):
        lo = max(0, index - half)
        hi = min(len(samples), index + half + 1)
        labels.append(
            classify(
                summarise(samples[lo:hi]), frame_width, sample_fps, max_rate=max_rate
            )
        )
    return labels


def to_runs(samples: Sequence[Sample], labels: Sequence[str]) -> list[Run]:
    """Group consecutive same-label samples into runs."""
    runs: list[Run] = []
    for sample, label in zip(samples, labels):
        if runs and runs[-1].motion_class == label:
            runs[-1] = Run(runs[-1].start, sample.end, label)
        else:
            runs.append(Run(sample.start, sample.end, label))
    return runs


def merge_short_runs(
    runs: Sequence[Run], min_seconds: float, min_reposition_seconds: float = 1.0
) -> list[Run]:
    """Absorb runs too short to stand alone into their longer neighbour.

    The two classes need different minimums, and getting this wrong defeats the
    stage. A *usable* run shorter than `min_seconds` is not postable, so it is
    absorbed. But a reposition only has to last about a second to be worth
    cutting out — judged by the same three-second bar, a two-second reposition
    gets folded back into the shot beside it and relabelled usable, which is
    precisely the footage this exists to separate.

    Below `min_reposition_seconds` a reposition is a wobble mid-move, and
    absorbing it avoids splitting a good shot in two.

    A merged run takes the class of the neighbour it joins, so searching can
    only be relabelled usable when it was too brief to matter.
    """
    merged = list(runs)
    if len(merged) < 2:
        return merged

    def too_short(run: Run) -> bool:
        floor = min_seconds if run.usable else min_reposition_seconds
        return run.duration < floor

    while len(merged) > 1:
        index = next((i for i, run in enumerate(merged) if too_short(run)), None)
        if index is None:
            break

        before = merged[index - 1] if index > 0 else None
        after = merged[index + 1] if index + 1 < len(merged) else None
        # Join whichever neighbour is longer — the short run is noise next to
        # it, so that is the class more likely to be right.
        target = max(
            (n for n in (before, after) if n is not None), key=lambda r: r.duration
        )
        start = min(merged[index].start, target.start)
        end = max(merged[index].end, target.end)
        replace_at = index - 1 if target is before else index
        merged[replace_at : replace_at + 2] = [Run(start, end, target.motion_class)]

    collapsed = [merged[0]]
    for run in merged[1:]:
        if run.motion_class == collapsed[-1].motion_class:
            collapsed[-1] = Run(collapsed[-1].start, run.end, run.motion_class)
        else:
            collapsed.append(run)
    return collapsed


def segment(
    samples: Sequence[Sample],
    frame_width: int,
    sample_fps: float,
    min_seconds: float,
    *,
    min_reposition_seconds: float = 1.0,
    window: int = 5,
    max_rate: float = MAX_RATE,
) -> list[Run]:
    """Full pipeline: label, group, merge."""
    if not samples:
        return []
    labels = label_samples(
        samples, frame_width, sample_fps, window=window, max_rate=max_rate
    )
    return merge_short_runs(
        to_runs(samples, labels), min_seconds, min_reposition_seconds
    )


# ---------------------------------------------------------------------------
# Measurement (needs OpenCV; imported lazily so the rest stays dependency-free)
# ---------------------------------------------------------------------------


def measure(
    frame_paths: Sequence[str], timestamps: Sequence[float]
) -> tuple[list[Sample], int]:
    """Camera movement between consecutive sampled frames.

    Returns the samples and the width they were measured at, since the
    thresholds are expressed relative to frame width.
    """
    import cv2
    import numpy as np

    samples: list[Sample] = []
    previous = None
    window = None
    width = 0

    for index, path in enumerate(frame_paths):
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        frame = frame.astype(np.float32)
        width = frame.shape[1]
        if window is None:
            # Hanning window suppresses the edge discontinuity that would
            # otherwise dominate the correlation peak.
            window = cv2.createHanningWindow(
                (frame.shape[1], frame.shape[0]), cv2.CV_32F
            )
        if previous is not None:
            (dx, dy), response = cv2.phaseCorrelate(previous, frame, window)
            samples.append(
                Sample(
                    start=timestamps[index - 1],
                    end=timestamps[index],
                    dx=float(dx),
                    dy=float(dy),
                    response=float(response),
                )
            )
        previous = frame

    return samples, width
