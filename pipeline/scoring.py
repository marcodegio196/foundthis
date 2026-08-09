"""Pure scoring maths for Stage 2.

Kept separate from the runner so the thresholds that decide what survives the
archive can be tested without OpenCV, torch, or a single frame of footage.

Every score is normalised to 0..1, higher is better, so they can be weighted
against each other and compared across resolutions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

# Weights for the combined score. Stability is deliberately heavy: the format
# this pipeline feeds is a held shot with a text overlay, and a technically
# sharp but jittery clip is useless for it while still being fine as stock.
WEIGHTS = {
    "aesthetic": 0.40,
    "stability": 0.30,
    "technical": 0.20,
    "motion": 0.10,
}

# Below these, a shot is rejected no matter how well it scores elsewhere —
# out-of-focus or blown-out footage has no use in either output.
HARD_FLOORS = {"technical": 0.15}


@dataclass(frozen=True)
class FrameFlow:
    """Mean optical flow between one pair of sampled frames."""

    dx: float
    dy: float
    magnitude: float  # mean per-pixel magnitude, in pixels


@dataclass(frozen=True)
class Scores:
    motion: float | None = None
    stability: float | None = None
    technical: float | None = None
    aesthetic: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def motion_score(flows: Sequence[FrameFlow], frame_width: int) -> float | None:
    """How much the frame is moving, as a fraction of frame width per step.

    Normalised by width so a 4K and a 1080p copy of the same move score the
    same. Saturates around 4% of frame width per sample, which is already a
    brisk push — beyond that the number stops being informative.
    """
    if not flows or frame_width <= 0:
        return None
    mean_magnitude = sum(f.magnitude for f in flows) / len(flows)
    return _clamp(math.tanh((mean_magnitude / frame_width) / 0.04))


def stability_score(flows: Sequence[FrameFlow]) -> float | None:
    """Whether the movement is deliberate, on a scale of drift to shake.

    Two signals, multiplied:

    - *coherence* — the length of the mean flow vector over the mean flow
      magnitude. A smooth pan moves every pixel the same way, so the two are
      nearly equal (→1). Handheld shake and sensor noise cancel out
      directionally while keeping magnitude, so the ratio collapses (→0).
    - *consistency* — how steady that magnitude is between samples. A push-in
      holds its speed; a bumpy hover oscillates.

    A locked-off tripod shot has no flow to be coherent about, and is treated as
    perfectly stable rather than punished for it.
    """
    if not flows:
        return None

    magnitudes = [f.magnitude for f in flows]
    mean_magnitude = sum(magnitudes) / len(magnitudes)
    if mean_magnitude < 1e-6:
        return 1.0

    mean_dx = sum(f.dx for f in flows) / len(flows)
    mean_dy = sum(f.dy for f in flows) / len(flows)
    coherence = _clamp(math.hypot(mean_dx, mean_dy) / mean_magnitude)

    variance = sum((m - mean_magnitude) ** 2 for m in magnitudes) / len(magnitudes)
    consistency = _clamp(1.0 - (math.sqrt(variance) / mean_magnitude))

    return _clamp(coherence * 0.65 + consistency * 0.35)


def technical_score(
    sharpness: Sequence[float], clipped_fraction: Sequence[float]
) -> float | None:
    """Blur and exposure, combined.

    `sharpness` is the variance of the Laplacian per frame (a standard focus
    measure); 500 is a comfortably sharp 4K frame. `clipped_fraction` is the
    share of pixels crushed to black or blown to white — a little is normal in
    a sunset, a lot means the shot is unrecoverable.
    """
    if not sharpness:
        return None

    focus = _clamp((sum(sharpness) / len(sharpness)) / 500.0)
    if not clipped_fraction:
        return round(focus, 4)

    clipping = sum(clipped_fraction) / len(clipped_fraction)
    exposure = _clamp(1.0 - max(0.0, clipping - 0.02) / 0.20)
    return round(_clamp(focus * 0.6 + exposure * 0.4), 4)


def combine(scores: Scores, weights: dict[str, float] = WEIGHTS) -> float | None:
    """Weighted mean over whichever component scores are present.

    Missing components are dropped and the remaining weights renormalised, so a
    run without the aesthetic model still produces comparable numbers — just
    ones driven by motion and technical quality.
    """
    available = {
        key: value
        for key, value in scores.as_dict().items()
        if value is not None and weights.get(key)
    }
    if not available:
        return None
    total_weight = sum(weights[key] for key in available)
    return round(
        sum(value * weights[key] for key, value in available.items()) / total_weight, 4
    )


def fails_hard_floor(scores: Scores, floors: dict[str, float] = HARD_FLOORS) -> str | None:
    """Reason string if any component is below its absolute minimum."""
    values = scores.as_dict()
    for key, floor in floors.items():
        value = values.get(key)
        if value is not None and value < floor:
            return f"{key} below floor ({value:.2f} < {floor:.2f})"
    return None


def percentile_threshold(values: Sequence[float], percentile: float) -> float | None:
    """Value at `percentile` (0..1) using linear interpolation.

    This is what "reject the bottom 35%" resolves to. It is computed over the
    scores actually present, so the bar moves as more of the archive is scored —
    intentionally, since the whole point is a relative cut.
    """
    if not values:
        return None
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0 and 1")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = percentile * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
