"""Free local disk by deleting raw sources whose shots are all done.

Every file under `archive_root` is already backed up to cloud storage outside
this pipeline, so the local copy only exists for the pipeline to cut shots
from it. Once every shot cut from a source has either rendered (the render
*is* the copy that gets posted or sold — see `select_render.render_shot`) or
been rejected (it will never render), the raw file has nothing left to do
locally. `sources` rows and every `shots` row are untouched: only the bytes on
disk go away, so the archive stays fully queryable and a re-render just needs
the file pulled back down from cloud.

Defaults to a dry run. Deleting source footage is not reversible from this
machine, so seeing what would be freed before it happens is the safer default
— pass `dry_run=False` (or `--yes` on the CLI) once you've checked the list.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from .. import db
from ..config import Config, config as default_config

log = logging.getLogger(__name__)

# A source only qualifies once stage 1b has actually cut it into shots — an
# unsegmented file has no shots to check yet, so treating it as "nothing left
# to do" would be silence, not readiness.
_ELIGIBLE_SQL = """
    SELECT * FROM sources
    WHERE scene_detected_at IS NOT NULL
      AND purged_at IS NULL
      AND NOT EXISTS (
          SELECT 1 FROM shots
          WHERE shots.source_id = sources.id
            AND shots.rejected = 0
            AND shots.rendered_at IS NULL
      )
    ORDER BY relpath
"""


def eligible_sources(conn: sqlite3.Connection, limit: int | None = None) -> list[db.Row]:
    """Sources where every shot has either rendered or been rejected."""
    sql = _ELIGIBLE_SQL + (f" LIMIT {int(limit)}" if limit else "")
    return conn.execute(sql).fetchall()


def run(
    conn: sqlite3.Connection,
    cfg: Config = default_config,
    *,
    limit: int | None = None,
    dry_run: bool = True,
) -> dict[str, int]:
    """Delete the raw file for every fully-processed source."""
    counts = {"purged": 0, "missing": 0, "freed_bytes": 0}

    for row in eligible_sources(conn, limit):
        path = Path(row["path"])
        size = row["size_bytes"] or 0

        if dry_run:
            note = "already gone" if not path.is_file() else f"{size / 1e6:.1f} MB"
            log.info("[dry run] would purge %s (%s)", path, note)
            counts["purged"] += 1
            continue

        if path.is_file():
            path.unlink()
            counts["freed_bytes"] += size
            log.info("deleted %s (%.1f MB)", path, size / 1e6)
        else:
            # Already gone by some other means (moved by hand, disk swapped).
            # The goal here — no local bytes left — is already true, so this
            # still stamps purged_at rather than flagging the same file again
            # on every future run.
            counts["missing"] += 1
            log.info("%s already gone, marking purged", path)

        with db.transaction(conn):
            db.update_source(conn, row["id"], purged_at=db.now())
        counts["purged"] += 1

    return counts
