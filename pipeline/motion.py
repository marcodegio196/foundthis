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

# --- Speed steadiness -------------------------------------------------------
#
# `jitter` asks whether the camera holds its *direction*. It says nothing about
# whether it holds its *speed*, and a move that stops dead and restarts keeps a
# constant heading throughout — so a clean-looking shot can contain a lurch that
# every existing measure reads as perfect. Three real examples, all with jitter
# under 0.03 and stability above 0.4:
#
#   a pan that stops for 0.23s mid-move and resumes
#   a 15s take whose speed collapses 73% in the second second, then recovers
#   a steady shot that steps up 52% in speed halfway through
#
# What separates those from good footage is not how much the speed changes but
# whether the change *reverses*. A deliberate ease ramps one way; a lurch dips
# and comes back. Measured over the judged clips: three bad shots scored 1-3
# reversals, the good one scored 0 despite an 18x speed range.

# Below this the camera is parked and relative changes are noise, not movement.
# In frame widths per second, the same units as HELD_RATE — a raw pixel figure
# would silently mean something different at another sampling rate, and the
# profile has to stay comparable across an archive shot at mixed resolutions.
SPEED_FLOOR = HELD_RATE

# Relative speed change worth noticing between one second and the next.
SPEED_MIN_CHANGE = 0.30

# Bucket width for the speed curve. A second is long enough to average out
# per-frame correlation noise and short enough to resolve a stop-and-go.
SPEED_BUCKET_SECONDS = 1.0


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


def merge_usable_runs(runs: Sequence[Run]) -> list[Run]:
    """Join neighbouring runs that are both usable.

    Segmentation exists to separate usable footage from repositioning, not to
    separate one kind of usable footage from another. A drone easing out of a
    move into a hold is a single take, but the label flips the moment its speed
    crosses HELD_RATE, and splitting there cut an 13-second shot into a 4.4s
    piece and a 9.0s piece — neither long enough to post, from footage that was
    fine.

    The merged run keeps the label of whichever half lasted longer, so `held`
    and `move` still describe what the shot mostly does.
    """
    merged: list[Run] = []
    for run in runs:
        if merged and merged[-1].usable and run.usable:
            previous = merged[-1]
            longer = previous if previous.duration >= run.duration else run
            merged[-1] = Run(previous.start, run.end, longer.motion_class)
        else:
            merged.append(run)
    return merged


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


def speed_buckets(
    samples: Sequence[Sample],
    sample_fps: float,
    frame_width: int,
    *,
    bucket_seconds: float = SPEED_BUCKET_SECONDS,
) -> list[float]:
    """The shot's speed curve, one value per bucket, in frame widths per second.

    Stage 1b already measures every sample this needs, so building it costs no
    extra decode — which is what makes it affordable across a 250GB archive.

    Values are rates rather than raw pixels so the curve means the same thing
    at any resolution or sampling rate: a stored profile stays comparable when
    the archive mixes 1080p and 4K, and the thresholds do not quietly move if
    MOTION_SAMPLE_FPS is retuned.
    """
    if frame_width <= 0 or sample_fps <= 0:
        return []
    per = max(1, int(round(sample_fps * bucket_seconds)))
    out: list[float] = []
    for index in range(0, len(samples) - per + 1, per):
        chunk = samples[index : index + per]
        shift = sum(s.distance for s in chunk) / len(chunk)
        out.append((shift * sample_fps) / frame_width)
    return out


def speed_reversals(
    buckets: Sequence[float],
    *,
    floor: float = SPEED_FLOOR,
    min_change: float = SPEED_MIN_CHANGE,
) -> int:
    """How many times the speed curve changes direction.

    Small wobbles and near-static stretches are ignored, so what remains is the
    stop-and-go: speed falling far enough to see, then climbing back. A single
    deliberate acceleration or deceleration returns 0 however large it is.
    """
    count, direction, anchor = 0, 0, None
    for value in buckets:
        if anchor is None:
            anchor = value
            continue
        if max(anchor, value) < floor:
            anchor = value
            continue
        change = (value - anchor) / max(anchor, floor)
        if abs(change) < min_change:
            continue
        new_direction = 1 if change > 0 else -1
        if direction and new_direction != direction:
            count += 1
        direction, anchor = new_direction, value
    return count


def speed_spread(buckets: Sequence[float], *, floor: float = SPEED_FLOOR) -> float:
    """Ratio of fastest to slowest second. 1.0 is metronomic."""
    if not buckets:
        return 1.0
    fastest, slowest = max(buckets), min(buckets)
    return fastest / max(slowest, floor) if fastest > 0 else 1.0


def eases_in(buckets: Sequence[float], *, floor: float = SPEED_FLOOR) -> bool:
    """Did the shot start from rest?

    An acceleration out of a hover is a reveal and reads as deliberate. The same
    acceleration from an existing cruise reads as the drone departing. Without
    this exemption the spread test rejects legitimate slow reveals, which
    measured an 18x speed range while being the best clip in the set.
    """
    return bool(buckets) and buckets[0] < floor


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
    return merge_usable_runs(
        merge_short_runs(
            to_runs(samples, labels), min_seconds, min_reposition_seconds
        )
    )


@dataclass(frozen=True)
class Span:
    """A stretch of a run that holds a steady speed — a postable clip.

    Named `Span` rather than `Window` because `Window` already means the
    aggregate of a run of samples that `summarise()` returns.
    """

    start: float
    end: float
    reversals: int
    spread: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def find_windows(
    buckets: Sequence[float],
    *,
    start: float,
    min_seconds: float,
    max_seconds: float,
    bucket_seconds: float = SPEED_BUCKET_SECONDS,
    max_spread: float = 3.0,
    floor: float = SPEED_FLOOR,
    min_change: float = SPEED_MIN_CHANGE,
) -> list[Span]:
    """Every non-overlapping stretch that holds a steady speed for long enough.

    Anchoring the render at the shot's in-point keeps whatever happens to be at
    the front. On a 21-second take that meant shipping two stop-and-gos and
    discarding the calm tail; searching the same take found a window 8 seconds
    in with no reversals and a 1.16x spread, which is the part a human picked by
    eye. A long take can hold more than one such stretch, so this returns all of
    them rather than only the best.

    Longest first, then calmest: a clean 15 seconds beats a clean 10, and the
    leftovers around the chosen windows stay addressable as their own segments.
    """
    if not buckets or min_seconds <= 0:
        return []

    min_len = max(1, int(round(min_seconds / bucket_seconds)))
    max_len = max(min_len, int(round(max_seconds / bucket_seconds)))

    def score(lo: int, hi: int, allow_ease_in: bool) -> tuple[int, float] | None:
        chunk = buckets[lo:hi]
        if len(chunk) < min_len:
            return None
        reversals = speed_reversals(chunk, floor=floor, min_change=min_change)
        if reversals:
            return None
        spread = speed_spread(chunk, floor=floor)
        if spread <= max_spread:
            return reversals, spread
        # A reveal that starts from rest necessarily has a huge spread; judging
        # it by the same ratio as a cruising shot would throw the reveal away.
        if allow_ease_in and eases_in(chunk, floor=floor):
            return reversals, spread
        return None

    def search(allow_ease_in: bool) -> list[Span]:
        taken: list[Span] = []
        free = [(0, len(buckets))]

        while free:
            best: tuple[tuple[int, float, int], int, int, int, float] | None = None
            for lo, hi in free:
                for length in range(min(max_len, hi - lo), min_len - 1, -1):
                    for offset in range(lo, hi - length + 1):
                        result = score(offset, offset + length, allow_ease_in)
                        if result is None:
                            continue
                        reversals, spread = result
                        # Prefer longer, then calmer, then earlier.
                        key = (-length, spread, offset)
                        if best is None or key < best[0]:
                            best = (key, length, offset, reversals, spread)
            if best is None:
                break

            _, length, offset, reversals, spread = best
            taken.append(
                Span(
                    start=start + offset * bucket_seconds,
                    end=start + (offset + length) * bucket_seconds,
                    reversals=reversals,
                    spread=round(spread, 3),
                )
            )
            # Carve the chosen span out of the free list; what is left on either
            # side can still hold another window if it is long enough.
            remaining: list[tuple[int, int]] = []
            for lo, hi in free:
                if offset >= hi or offset + length <= lo:
                    remaining.append((lo, hi))
                    continue
                if offset - lo >= min_len:
                    remaining.append((lo, offset))
                if hi - (offset + length) >= min_len:
                    remaining.append((offset + length, hi))
            free = remaining
        return taken

    # Strict first, and only fall back to the ease-in exemption when nothing
    # qualifies without it. The exemption cannot tell a deliberate reveal from a
    # slow drift that happens to sit below the floor before accelerating away,
    # because the two have the same shape — so letting it compete on equal terms
    # loses good footage. On a take that drifted for five seconds and then
    # settled into a steady orbit, the exempt window ran 5-20 with an 11.6x
    # spread, straddling both behaviours; the strict search found 16-28 at 1.8x,
    # which is the orbit its owner actually pointed to. A genuine reveal is
    # unaffected: nothing qualifies strictly, so the exemption still returns it.
    taken = search(allow_ease_in=False) or search(allow_ease_in=True)
    return sorted(taken, key=lambda w: w.start)


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
