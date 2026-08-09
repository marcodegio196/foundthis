"""Stage 2's aesthetic predictor: CLIP embedding -> small MLP -> 0..1 score.

The model is optional. Without torch/open-clip installed, or without a weights
file, `load_scorer()` returns a scorer that yields None — Stage 2 then runs on
motion and technical quality alone and `scoring.combine` renormalises around the
missing component. That keeps the pipeline runnable on a laptop and on a box
with a GPU without a code change.

Weights are the widely used LAION aesthetic predictor head (a linear stack over
ViT-L/14 embeddings), pointed at by AESTHETIC_WEIGHTS.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, Sequence

log = logging.getLogger(__name__)

# The LAION head is trained on a 1-10 human rating scale; we store 0..1.
RATING_MAX = 10.0


class AestheticScorer(Protocol):
    def __call__(self, frame_paths: Sequence[Path]) -> float | None:
        """Mean aesthetic score over a shot's sampled frames, or None."""


def null_scorer(frame_paths: Sequence[Path]) -> float | None:
    return None


class ClipAestheticScorer:
    """Lazily loads CLIP + the MLP head on first call."""

    def __init__(self, weights_path: Path, model_name: str, pretrained: str, device: str):
        self.weights_path = weights_path
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._preprocess = None
        self._head = None

    def _ensure_loaded(self) -> None:
        if self._head is not None:
            return

        import open_clip
        import torch
        from torch import nn

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self._model.eval()

        embedding_dim = self._model.visual.output_dim
        head = nn.Sequential(
            nn.Linear(embedding_dim, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        state = torch.load(self.weights_path, map_location=self.device)
        head.load_state_dict(state.get("state_dict", state))
        self._head = head.to(self.device).eval()

    def __call__(self, frame_paths: Sequence[Path]) -> float | None:
        if not frame_paths:
            return None
        self._ensure_loaded()

        import torch
        from PIL import Image

        batch = torch.stack(
            [self._preprocess(Image.open(p).convert("RGB")) for p in frame_paths]
        ).to(self.device)

        with torch.no_grad():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            ratings = self._head(features.float()).squeeze(-1)

        return round(min(1.0, max(0.0, ratings.mean().item() / RATING_MAX)), 4)


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_scorer(weights_path: str | Path | None = None) -> AestheticScorer:
    """Return a working CLIP scorer, or `null_scorer` with a reason logged."""
    weights_path = weights_path or os.environ.get("AESTHETIC_WEIGHTS")
    if not weights_path:
        log.info("AESTHETIC_WEIGHTS unset — scoring without the aesthetic component")
        return null_scorer

    weights_path = Path(weights_path).expanduser()
    if not weights_path.is_file():
        log.warning("aesthetic weights not found at %s — skipping component", weights_path)
        return null_scorer

    try:
        import open_clip  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        log.warning("torch/open-clip not installed — skipping aesthetic component")
        return null_scorer

    return ClipAestheticScorer(
        weights_path=weights_path,
        model_name=os.environ.get("CLIP_MODEL", "ViT-L-14"),
        pretrained=os.environ.get("CLIP_PRETRAINED", "openai"),
        device=os.environ.get("TORCH_DEVICE", _default_device()),
    )
