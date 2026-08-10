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
| `pipeline/zernio.py` | Zernio API client, stdlib only |
| `pipeline/layout.py` | Where files land on disk |
| `pipeline/cli.py` | One subcommand per stage, plus `init` / `stats` / `export` |
| `tests/` | 227 tests. Most need no dependencies; `test_integration.py` renders real video and skips itself when ffmpeg or OpenCV are absent |

## Getting started

```bash
python3 -m pipeline.cli init                 # create ./pipeline.db
python3 -m unittest discover tests           # 227 tests

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

`OVERLAY_FONT` is required for the burned-in text on Linux and Windows (macOS
falls back to a system font). Windows paths are escaped for ffmpeg
automatically — set it raw:

```powershell
$env:OVERLAY_FONT = "C:\Windows\Fonts\segoeui.ttf"
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

## Everything stays on local disk

Raw footage is never written to, moved, or deleted — the pipeline only reads it.
Renders and exports go into their own trees, mirroring the `archive/<country>/`
convention so they're browsable in Explorer without opening the database:

```
archive/albania/DJI_0001.MP4              # untouched, read-only
renders/social/albania/2024/000012_dji_0001_ksamil.mp4
renders/licensing/albania/2024/000012_dji_0001_ksamil.mp4
exports/non-exclusive/albania/2024/000012_dji_0001_ksamil.mp4
exports/non-exclusive/albania/2024/000012_dji_0001_ksamil.json
exports/non-exclusive/manifest.csv
```

Filenames lead with the zero-padded shot id so they sort stably and map back to
a database row, and keep the source name so a file found on disk can be traced
to its original clip without a query. Paths are deterministic, so re-rendering
overwrites rather than accumulating a second copy. Undated footage sorts into
`undated/` rather than a guessed year.

```bash
python3 -m pipeline.cli export --dry-run              # see what would go out
python3 -m pipeline.cli export --tier non-exclusive   # one marketplace batch
```

`export` assembles the licensing deliverable: the clean master, a per-clip JSON
sidecar (title, keywords, location, GPS, drone model, resolution, provenance),
and one `manifest.csv` for the batch. **Masters are hard-linked, not copied**,
where the filesystem allows it — a second name for the same bytes, costing no
extra disk. It falls back to a copy on external or network drives that don't
support links, and deleting an export never touches the render.

## What Stage 2 actually measures

The stability score is the one Stage 2 leans on, so it's validated against
frames with displacements chosen in advance rather than assumed
(`tests/test_integration.py`):

| Shot | Stability | Motion | Technical |
|---|---:|---:|---:|
| Smooth pan | 1.00 | 0.44 | 0.95 |
| Locked off (incl. sensor noise) | 1.00 | 0.00 | 0.95 |
| Blurred pan | 1.00 | 0.38 | **0.40** |
| Handheld shake | 0.63 | 0.29 | 0.95 |
| Drifting hover | 0.35 | 0.22 | 0.95 |

A held shot and a deliberate pan both score 1.00; shake and aimless drift fall
away. A blurred pan is *correctly* stable — it's the technical score that
catches it, which is why the two are separate components rather than one number.

Two things that only showed up against real footage: identical frames still
produce ~0.0002px of optical flow, and footage with sensor noise ~0.03px, so
"no movement" has to be a realistic threshold rather than exact zero — a real
one-pixel-per-frame pan measures 1.0px, leaving a wide margin.

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
