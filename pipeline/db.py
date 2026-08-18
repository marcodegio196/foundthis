"""SQLite access layer — the spine every pipeline stage reads from and writes to.

Stages never open the database themselves; they call `connect()` and use the
helpers here so schema knowledge stays in one place. Nothing in this module
touches the raw footage.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Bumped when schema.sql changes in a way that needs a migration step.
SCHEMA_VERSION = 4

# Columns stored as JSON text. Read helpers decode them; write helpers encode.
JSON_COLUMNS = frozenset(
    {
        "subject_tags", "mood_tags", "platform_ids", "licensed_to",
        "delivery_status", "metrics", "speed_profile",
    }
)

# `CREATE TABLE IF NOT EXISTS` won't add a column to a table that already
# exists, so each version's additive changes are replayed explicitly.
MIGRATIONS: dict[int, tuple[str, ...]] = {
    2: (
        "ALTER TABLE shots ADD COLUMN zernio_post_id TEXT",
        "ALTER TABLE shots ADD COLUMN delivery_status TEXT",
        "ALTER TABLE shots ADD COLUMN delivery_checked_at TEXT",
    ),
    3: (
        "ALTER TABLE shots ADD COLUMN motion_class TEXT",
    ),
    4: (
        "ALTER TABLE shots ADD COLUMN speed_profile TEXT",
        "ALTER TABLE shots ADD COLUMN speed_reversals INTEGER",
        "ALTER TABLE shots ADD COLUMN speed_spread REAL",
        "ALTER TABLE shots ADD COLUMN is_window INTEGER NOT NULL DEFAULT 1",
    ),
}


class Row(sqlite3.Row):
    """sqlite3.Row that also decodes the JSON columns on access."""

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        if isinstance(key, str) and key in JSON_COLUMNS and isinstance(value, str):
            return json.loads(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (IndexError, KeyError):
            return default

    def dict(self) -> dict[str, Any]:
        return {key: self[key] for key in self.keys()}


def now() -> str:
    """UTC timestamp in the same format the schema's DEFAULTs produce."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def connect(db_path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the pipeline database with the pragmas every stage expects."""
    db_path = Path(db_path)
    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)

    conn.row_factory = Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        # Stage 2/3 runs are long; WAL lets you query progress from another shell.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Create the database if needed and bring it up to `SCHEMA_VERSION`."""
    conn = connect(db_path)
    existing = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'shots'"
    ).fetchone()[0]
    current = conn.execute("PRAGMA user_version").fetchone()[0]

    conn.executescript(SCHEMA_PATH.read_text())

    if existing:
        for version in sorted(MIGRATIONS):
            if current < version:
                for statement in MIGRATIONS[version]:
                    try:
                        conn.execute(statement)
                    except sqlite3.OperationalError as exc:
                        # A fresh table already has the column; anything else is real.
                        if "duplicate column name" not in str(exc):
                            raise

    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Commit on success, roll back on any exception."""
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def _encode(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value) if key in JSON_COLUMNS and value is not None else value
        for key, value in values.items()
    }


def _speed_reversals(profile: Sequence[float] | None) -> int | None:
    """Summary of the speed curve, stored so queries need not decode the JSON."""
    if not profile:
        return None
    from . import motion  # local import: db.py stays dependency-free at module level

    return motion.speed_reversals(profile)


def _speed_spread(profile: Sequence[float] | None) -> float | None:
    if not profile:
        return None
    from . import motion

    return round(motion.speed_spread(profile), 3)


def upsert_source(conn: sqlite3.Connection, **values: Any) -> int:
    """Register a raw file, or refresh its probe data if it is already known.

    Keyed on `path`, so re-running ingest over the archive is idempotent.
    """
    if "path" not in values:
        raise ValueError("upsert_source requires 'path'")

    columns = list(values)
    placeholders = ", ".join(f":{c}" for c in columns)
    updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c != "path")
    conn.execute(
        f"INSERT INTO sources ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(path) DO UPDATE SET {updates}",
        _encode(values),
    )
    return conn.execute(
        "SELECT id FROM sources WHERE path = ?", (values["path"],)
    ).fetchone()[0]


def replace_shots(
    conn: sqlite3.Connection,
    source_id: int,
    segments: Sequence[tuple[float, float] | tuple[float, float, str]],
) -> list[int]:
    """Write the shot list for a source, replacing any previous detection.

    Segments are `(in, out)`, `(in, out, motion_class)`, or
    `(in, out, motion_class, speed_profile)`.

    Re-running detection with different thresholds is expected, so this clears
    the source's existing shots first. That discards downstream state for those
    shots by design — the segments they described no longer exist.
    """
    with transaction(conn):
        conn.execute("DELETE FROM shots WHERE source_id = ?", (source_id,))
        shot_ids: list[int] = []
        for index, segment in enumerate(segments):
            in_point, out_point = segment[0], segment[1]
            motion_class = segment[2] if len(segment) > 2 else None
            profile = list(segment[3]) if len(segment) > 3 and segment[3] else None
            is_window = bool(segment[4]) if len(segment) > 4 else True
            cursor = conn.execute(
                "INSERT INTO shots (source_id, shot_index, in_point, out_point, "
                "continuous_move, motion_class, speed_profile, speed_reversals, "
                "speed_spread, is_window) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_id,
                    index,
                    in_point,
                    out_point,
                    1 if len(segments) == 1 else 0,
                    motion_class,
                    json.dumps(profile) if profile else None,
                    _speed_reversals(profile),
                    _speed_spread(profile),
                    1 if is_window else 0,
                ),
            )
            shot_ids.append(int(cursor.lastrowid))
        conn.execute(
            "UPDATE sources SET scene_detected_at = datetime('now') WHERE id = ?",
            (source_id,),
        )
    return shot_ids


def update_shot(conn: sqlite3.Connection, shot_id: int, **values: Any) -> None:
    """Patch a shot row. Only the named columns are touched."""
    if not values:
        return
    assignments = ", ".join(f"{c} = :{c}" for c in values)
    conn.execute(
        f"UPDATE shots SET {assignments} WHERE id = :shot_id",
        {**_encode(values), "shot_id": shot_id},
    )


# ---------------------------------------------------------------------------
# Reads — one query per stage, so runners stay thin
# ---------------------------------------------------------------------------


def sources_pending_scene_detection(conn: sqlite3.Connection) -> list[Row]:
    return conn.execute(
        "SELECT * FROM sources WHERE scene_detected_at IS NULL ORDER BY relpath"
    ).fetchall()


def shots_pending_scoring(conn: sqlite3.Connection, limit: int | None = None) -> list[Row]:
    return conn.execute(
        "SELECT * FROM shot_details WHERE scored_at IS NULL ORDER BY id"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()


def shots_pending_tagging(conn: sqlite3.Connection, limit: int | None = None) -> list[Row]:
    """Survivors of stage 2 that stage 3 has not described yet."""
    return conn.execute(
        "SELECT * FROM shot_details "
        "WHERE rejected = 0 AND scored_at IS NOT NULL AND tagged_at IS NULL "
        "ORDER BY aesthetic_score DESC"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()


def selection_queue(conn: sqlite3.Connection, limit: int | None = None) -> list[Row]:
    """Tagged, un-posted survivors awaiting a stage 4 decision."""
    return conn.execute(
        "SELECT * FROM shot_details "
        "WHERE rejected = 0 AND tagged_at IS NOT NULL AND posted_at IS NULL "
        "AND selection_state IN ('unreviewed', 'queued') "
        "ORDER BY selection_state DESC, aesthetic_score DESC"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()


def reject(conn: sqlite3.Connection, shot_ids: Iterable[int], reason: str) -> int:
    """Flag shots as rejected. Nothing is deleted — rejects stay licensable and
    can be re-evaluated later under a different bar."""
    ids = list(shot_ids)
    if not ids:
        return 0
    with transaction(conn):
        conn.executemany(
            "UPDATE shots SET rejected = 1, reject_reason = ? WHERE id = ?",
            [(reason, shot_id) for shot_id in ids],
        )
    return len(ids)


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    """One-glance pipeline state, used by the CLI and worth logging per run."""
    scalar = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "sources": scalar("SELECT count(*) FROM sources"),
        "sources_unsegmented": scalar(
            "SELECT count(*) FROM sources WHERE scene_detected_at IS NULL"
        ),
        "shots": scalar("SELECT count(*) FROM shots"),
        "held": scalar("SELECT count(*) FROM shots WHERE motion_class = 'held'"),
        "moving": scalar("SELECT count(*) FROM shots WHERE motion_class = 'move'"),
        "repositioning": scalar(
            "SELECT count(*) FROM shots WHERE motion_class = 'reposition'"
        ),
        "scored": scalar("SELECT count(*) FROM shots WHERE scored_at IS NOT NULL"),
        "rejected": scalar("SELECT count(*) FROM shots WHERE rejected = 1"),
        "tagged": scalar("SELECT count(*) FROM shots WHERE tagged_at IS NOT NULL"),
        "approved": scalar(
            "SELECT count(*) FROM shots WHERE selection_state = 'approved'"
        ),
        "rendered": scalar("SELECT count(*) FROM shots WHERE rendered_at IS NOT NULL"),
        "posted": scalar("SELECT count(*) FROM shots WHERE posted_at IS NOT NULL"),
    }
