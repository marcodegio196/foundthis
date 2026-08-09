"""Stage 3 — describe each surviving shot with a VLM pass.

The highest-leverage stage: one pass produces both the social pipeline's
selection signals (mood, person-in-frame) and the licensing catalog's sellable
metadata (subject, location, description). Because the same output feeds a
stock listing, tags come from a controlled vocabulary — free-form tags make the
archive unqueryable once it runs to thousands of shots.

Only shots that survived Stage 2 are tagged; rejects stay untagged until the
bar moves and they are re-scored.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Protocol, Sequence

from .. import db, media
from ..config import Config, config as default_config

log = logging.getLogger(__name__)

MODEL = os.environ.get("TAG_MODEL", "claude-opus-5")

# Frames per shot sent to the model. Four spread across a shot capture the move
# without paying for near-duplicates of a held frame.
FRAMES_PER_SHOT = 4

# Controlled vocabularies. The model may only return these, which is what makes
# "every solitude shot in Albania" a query rather than a full-text search.
SUBJECT_TAGS = [
    "coastline", "beach", "cliffs", "mountains", "forest", "desert", "river",
    "lake", "waterfall", "farmland", "village", "town", "city", "harbour",
    "ruins", "religious-site", "castle", "bridge", "road", "boat", "vehicle",
    "livestock", "snow", "island", "canyon", "valley",
]

MOOD_TAGS = [
    "solitude", "vastness", "serenity", "drama", "desolation", "warmth",
    "golden-hour", "blue-hour", "overcast", "harsh-light", "mist", "storm",
    "nostalgia", "grandeur", "intimacy",
]

SYSTEM_PROMPT = """You are cataloguing aerial and gimbal-stabilised travel \
footage for two purposes at once: selecting clips for a short-form social feed, \
and listing them in a stock-footage catalog.

You will be shown several frames sampled in order from a single continuous shot. \
Describe the shot as a whole, not any one frame.

Rules:
- Choose tags only from the vocabularies given. Omit a tag rather than \
approximating it; an empty list is a valid answer.
- `description` is a single sentence a stock buyer would search against: what is \
in frame, where, and how the camera moves. No marketing language, no "this \
video shows".
- `person_in_frame` is true only if a human figure is identifiable, however \
small.
- `inferred_site` is a specific named place (a town, beach, peak, or landmark) \
only when the imagery genuinely identifies it. Return null when you would be \
guessing — a wrong location is worse than none in a licensing catalog."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject_tags": {
            "type": "array",
            "items": {"type": "string", "enum": SUBJECT_TAGS},
            "description": "What is physically in frame.",
        },
        "mood_tags": {
            "type": "array",
            "items": {"type": "string", "enum": MOOD_TAGS},
            "description": "How the shot feels, including its light.",
        },
        "person_in_frame": {
            "type": "boolean",
            "description": "True if any identifiable human figure appears.",
        },
        "description": {
            "type": "string",
            "description": "One sentence: subject, setting, and camera movement.",
        },
        "inferred_site": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Specific named location, or null if not identifiable.",
        },
    },
    "required": [
        "subject_tags", "mood_tags", "person_in_frame", "description", "inferred_site",
    ],
    "additionalProperties": False,
}


class Tagger(Protocol):
    def __call__(self, frame_paths: Sequence[Path], context: dict[str, Any]) -> dict[str, Any]:
        """Describe one shot from its sampled frames."""


def _image_block(path: Path) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(path.read_bytes()).decode("utf-8"),
        },
    }


class ClaudeTagger:
    """Vision + structured-output pass over a shot's frames.

    Structured outputs (`output_config.format`) rather than prompt-and-parse:
    the schema is enforced server-side, so a malformed tag list can't reach the
    database and there is no JSON repair path to maintain.
    """

    def __init__(self, model: str = MODEL, client: Any | None = None):
        self.model = model
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _prompt(self, context: dict[str, Any]) -> str:
        known = []
        if context.get("country"):
            known.append(f"country: {context['country']}")
        if context.get("site"):
            known.append(f"site noted in the archive: {context['site']}")
        if context.get("captured_at"):
            known.append(f"captured: {context['captured_at']}")
        if context.get("gps_lat") is not None:
            known.append(f"GPS: {context['gps_lat']:.4f}, {context['gps_lon']:.4f}")
        known.append(f"shot length: {context.get('duration', 0):.1f}s")

        return (
            f"{len(context['frames'])} frames sampled in order from one shot.\n"
            "What the archive already knows about it:\n- "
            + "\n- ".join(known)
            + "\n\nCatalogue this shot."
        )

    def __call__(
        self, frame_paths: Sequence[Path], context: dict[str, Any]
    ) -> dict[str, Any]:
        context = {**context, "frames": frame_paths}
        content: list[dict[str, Any]] = [_image_block(p) for p in frame_paths]
        content.append({"type": "text", "text": self._prompt(context)})

        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_config={
                # Cataloguing is extraction, not reasoning — low effort keeps a
                # few-thousand-shot archive affordable.
                "effort": "low",
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            },
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"model declined to describe the shot: "
                f"{getattr(response.stop_details, 'category', None)}"
            )

        text = next(block.text for block in response.content if block.type == "text")
        return json.loads(text)


def tag_shot(
    row: db.Row, tagger: Tagger, cfg: Config = default_config
) -> dict[str, Any] | None:
    """Sample frames from one shot and return the tagger's verdict."""
    timestamps = media.sample_timestamps(
        row["in_point"], row["out_point"], cfg.sample_fps, max_frames=FRAMES_PER_SHOT
    )
    if not timestamps:
        return None

    with tempfile.TemporaryDirectory(prefix="tag-") as tmp:
        frame_paths = media.extract_frames(
            row["source_path"], timestamps, tmp, width=768
        )
        if not frame_paths:
            return None
        return tagger(
            frame_paths,
            {
                "country": row["country"],
                "site": row["site"],
                "captured_at": row["captured_at"],
                "gps_lat": row["gps_lat"],
                "gps_lon": row["gps_lon"],
                "duration": row["duration"],
            },
        )


def run(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
    tagger: Tagger | None = None,
) -> dict[str, int]:
    """Describe every scored, un-rejected shot that has no tags yet."""
    tagger = tagger if tagger is not None else ClaudeTagger()
    counts = {"tagged": 0, "skipped": 0, "failed": 0}

    for row in db.shots_pending_tagging(conn, limit=limit):
        try:
            result = tag_shot(row, tagger, cfg)
        except Exception as exc:  # a single bad shot must not stall the pass
            log.warning("tagging failed for shot %s: %s", row["id"], exc)
            counts["failed"] += 1
            continue

        if result is None:
            counts["skipped"] += 1
            continue

        with db.transaction(conn):
            db.update_shot(
                conn,
                row["id"],
                subject_tags=result["subject_tags"],
                mood_tags=result["mood_tags"],
                person_in_frame=int(result["person_in_frame"]),
                description=result["description"],
                inferred_site=result["inferred_site"],
                tagged_at=db.now(),
                tag_model=getattr(tagger, "model", None),
            )
        counts["tagged"] += 1
        log.info("shot %s: %s", row["id"], result["description"])

    return counts
