# foundthis

Pipeline that turns a ~250GB raw drone/Osmo archive into two things at once: an
automated "Found this." short-form social presence, and a stock/licensing
catalog built from the same tagged shots.

Five stages, run in order, all sharing one SQLite database. Raw footage is never
modified — the database accumulates metadata and state as a shot moves through.

```
archive/ ──▶ 1 ingest & segment ──▶ 2 score & filter ──▶ 3 tag & describe ──┬──▶ 4 render social ──▶ 5 post via Zernio
                                                                            └──▶ 4 render master ──▶ 5 licensing upload
```

Full design: [`docs/architecture.md`](docs/architecture.md).

## Layout

| Path | What it is |
|---|---|
| `pipeline/schema.sql` | The spine — `sources`, `shots`, and the `shot_details` view |
| `pipeline/db.py` | Connection, migrations, and the per-stage queue queries |
| `pipeline/config.py` | Paths and thresholds, all env-overridable |
| `pipeline/cli.py` | `init` / `stats` / `queue` |
| `tests/` | Schema and DB-layer tests (stdlib `unittest`, no deps needed) |

## Getting started

```bash
python3 -m pipeline.cli init      # create ./pipeline.db
python3 -m pipeline.cli stats     # see how far the archive has moved
python3 -m pipeline.cli queue     # shots awaiting a stage 4 decision
python3 -m unittest discover tests
```

Configuration is environment-driven, so the same code runs against the real
archive or a small test folder:

```bash
export ARCHIVE_ROOT=/Volumes/drone/archive   # expects archive/<country>/<file>
export PIPELINE_DB=./pipeline.db
export RENDER_ROOT=./renders
```

Stage 2+ needs `requirements.txt` installed plus `ffmpeg`/`ffprobe` on `PATH`.
The database layer and CLI deliberately need neither.

## Status

| Stage | State |
|---|---|
| 1 — ingest & segment | Written and tested locally (`01_ingest.py`, `02_scene_detect.py`), **not yet in this repo** |
| 2 — score & filter | Schema in place, runner not built |
| 3 — tag & describe | Schema in place, runner not built |
| 4 — select & render | Designed |
| 5 — distribute & learn | Zernio side ready; feedback loop later |

## Conventions

- **Nothing is ever deleted.** Filtering sets `rejected = 1` with a reason. A
  shot that fails the social bar is still stock inventory, and the bar moves.
- **Every stage is idempotent.** Ingest upserts on path; each stage finds its
  work by looking for `NULL` in its own timestamp column, so an interrupted run
  is resumed by re-running it.
- **Stages talk only through the database.** No stage imports another.
