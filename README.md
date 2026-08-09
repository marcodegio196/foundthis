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
| `pipeline/media.py` | The only module that shells out to ffmpeg/ffprobe |
| `pipeline/scoring.py` | Stage 2 maths — pure, no OpenCV or torch needed to test |
| `pipeline/aesthetic.py` | Optional CLIP+MLP aesthetic head |
| `pipeline/render.py` | Stage 4 ffmpeg command construction — pure |
| `pipeline/stages/` | One module per stage, each `run(conn, cfg, **options) -> dict` |
| `pipeline/cli.py` | One subcommand per stage, plus `init` / `stats` |
| `tests/` | 127 tests (stdlib `unittest`, no deps needed) |

## Getting started

```bash
python3 -m pipeline.cli init                 # create ./pipeline.db
python3 -m unittest discover tests           # 127 tests, no dependencies

python3 -m pipeline.cli -v ingest            # 1a: register source files
python3 -m pipeline.cli -v segment           # 1b: split into shots
python3 -m pipeline.cli -v score             # 2:  score, then cut the bottom
python3 -m pipeline.cli -v tag               # 3:  describe survivors
python3 -m pipeline.cli queue --stage        # 4:  propose shots to review
python3 -m pipeline.cli approve 12 47 93     # 4:  approve them
python3 -m pipeline.cli -v render            # 4:  render both profiles
python3 -m pipeline.cli -v publish           # 5:  post + pull metrics

python3 -m pipeline.cli stats                # how far the archive has moved
```

Re-drawing the Stage 2 bar doesn't need a rescore — scoring and rejection are
separate passes:

```bash
REJECT_PERCENTILE=0.25 python3 -m pipeline.cli score --rejections-only --dry-run
```

Configuration is environment-driven, so the same code runs against the real
archive or a small test folder:

```bash
export ARCHIVE_ROOT=/Volumes/drone/archive   # expects archive/<country>/<file>
export PIPELINE_DB=./pipeline.db
export RENDER_ROOT=./renders

# Stage 3
export ANTHROPIC_API_KEY=sk-ant-...

# Stage 5 — same variable names the new-visu app uses, so one .env serves both
export ZERNIO_API_KEY=sk_...                 # sk_ + 64 hex
export ZERNIO_BASE_URL=https://zernio.com/api/v1
export ZERNIO_PROFILE_ID=...                 # optional
export SOCIAL_PLATFORMS=instagram,tiktok
```

Accounts are connected through Zernio's OAuth redirect (`GET /connect/{platform}`),
which is an interactive browser flow — do it in the new-visu web app. This
pipeline only reads accounts that are already connected.

Stage 2+ needs `requirements.txt` installed plus `ffmpeg`/`ffprobe` on `PATH`.
The database layer and CLI deliberately need neither.

## Status

| Stage | State |
|---|---|
| 1 — ingest & segment | Built. Rewritten here rather than ported from `01_ingest.py` / `02_scene_detect.py`, which were never pushed to this repo |
| 2 — score & filter | Built. Aesthetic component needs `AESTHETIC_WEIGHTS`; without it the run uses motion and technical quality alone |
| 3 — tag & describe | Built. Needs `ANTHROPIC_API_KEY` |
| 4 — select & render | Built. Approval is a manual step until the format is validated |
| 5 — distribute & learn | Built. Zernio client ported from the working integration in `new-visu`, dry-run until `ZERNIO_API_KEY` is set. Delivery status works; **performance metrics have no source yet** — see below |

## Delivery status is not performance

Zernio's API covers `accounts`, `connect`, `posts`, `media/presign`, `logs`, and
`inbox` — it reports **whether a post was delivered** (queued / published /
failed, per platform), not how it performed. Those are stored separately:

| Column | Source | State |
|---|---|---|
| `delivery_status`, `delivery_checked_at` | Zernio `GET /posts/{id}` | Working |
| `metrics`, `metrics_updated_at` | Platform analytics — no source wired up | Empty |

Stage 5's feedback loop reweights Stage 4 on *performance*, so it stays
unbuilt until views and engagement come from somewhere — platform Graph/Display
APIs, or a Zernio analytics endpoint if one exists that isn't in the client the
`new-visu` app uses. Keeping the two columns apart is deliberate: "it posted"
must never be mistaken for "it did well" once selection starts training on this
data.

## Conventions

- **Nothing is ever deleted.** Filtering sets `rejected = 1` with a reason. A
  shot that fails the social bar is still stock inventory, and the bar moves.
- **Every stage is idempotent.** Ingest upserts on path; each stage finds its
  work by looking for `NULL` in its own timestamp column, so an interrupted run
  is resumed by re-running it.
- **Stages talk only through the database.** No stage imports another.
