# Drone Archive Pipeline — 5-Stage Architecture (High Level)

**Asset:** ~250GB raw 4K drone/Osmo footage, 10 countries, 9:16 + 16:9 masters already available.
**Outputs:** (1) automated "Found this." short-form social presence via Zernio, (2) parallel stock/licensing catalog from the same tagged archive.
**Spine:** a single SQLite database (`shots` table) that every stage reads from and writes to. Raw footage is never modified — the DB just accumulates metadata and state as a shot moves through the pipeline.

---

## Stage 1 — Ingest & Segment
**Turns raw files into addressable, timestamped shots.**

- Walk the archive, register every source file (ffprobe: duration, resolution, codec, aspect, creation date, embedded GPS).
- Content-aware scene detection splits each file into shots (`in_point`/`out_point`), or keeps it as one shot if it's a continuous move — common for drone footage.
- Folder convention (`input/country/file`) auto-fills `country`; `site` stays blank for now.

*Status: built (`01_ingest.py`, `02_scene_detect.py`), tested — not yet committed to this repo.*

---

## Stage 2 — Score & Filter
**Cuts the archive down to the usable 30-40% cheaply, before any expensive processing.**

- **Motion/stability score** — frame-diff approach, ideally upgraded with optical flow to distinguish deliberate smooth movement from static-but-noisy footage. This is the key filter for the "held shot with text overlay" format specifically.
- **Aesthetic score** — CLIP+MLP predictor on sampled frames.
- **Technical score** — blur/exposure/artifact check.
- Bottom percentile auto-rejected (flagged, never deleted — stays available for licensing or future re-evaluation under a different bar).

*Status: next to build.*

---

## Stage 3 — Tag & Describe
**Turns surviving shots into a queryable, sellable catalog — this is the highest-leverage stage, feeds both monetization paths.**

- One VLM pass per surviving shot: subject tags (village/coastline/ruins/etc.), mood tags (solitude/vastness/golden-hour/etc.), person-in-frame flag, draft one-line description.
- This is also where `site`/specific-location detail gets inferred when folder structure doesn't provide it.
- Output metadata doubles as licensing catalog metadata (location, subject, resolution, drone model) — one tagging pass serves both the social pipeline and stock buyers.

*Status: next to build, after scoring.*

---

## Stage 4 — Select & Render
**Two profiles, one source table, two audiences.**

| | Social | Licensing |
|---|---|---|
| Source | native 9:16, or 16:9 cropped | native 16:9/4K master |
| Output | overlay burned in ("Found this." / year · location), compressed for platform | clean, no overlay, no watermark, full resolution |
| Gate | selection queue (mood/country diversity rules, periodic you-in-frame clip) | `license_tier` flag (public / non-exclusive / exclusive) |

Selection starts semi-manual (approve the queue) until the format is validated, then automates using Stage 3 tags.

*Status: designed, not yet built — natural next build target.*

---

## Stage 5 — Distribute & Learn
**Publish, then feed results back into selection.**

- **Social:** rendered clip + generated caption pushed via Zernio to all platforms; `platform_ids` and `posted_at` stored back on the shot.
- **Licensing:** rendered master + metadata uploaded to chosen marketplace(s) — manual/semi-manual at first, automate only once a platform proves it converts.
- **Feedback loop:** pull performance metrics periodically, store on the shot row. Once there is volume, this reweights Stage 4 selection (which moods/countries/pacing actually perform) — and gives the reach data sponsors actually want to see.

*Status: Zernio side ready; feedback loop is a later addition once posting volume exists.*

---

## Build priority
Build stages roughly in order (1 → 2 → 3 → 4 → 5) since each depends on the last. No need to gate full automation on validating the social format first — the archive has standing value as a personal record and licensing catalog regardless of whether any given post performs, so there's little reason to hold back Stage 2/3 automation for proof-of-concept posting.

---

## Schema notes

`pipeline/schema.sql` implements the spine described above, with two deviations
worth stating explicitly:

1. **`sources` is split out from `shots`.** File-level probe data (codec,
   resolution, GPS, drone model) belongs to the file, not to each shot inside
   it. Keeping it in one place means re-probing a file doesn't have to be
   written to every shot it contains. The `shot_details` view joins the two
   back together, so stage code still reads a single flat row.

2. **Stage columns live on the shot row rather than in per-stage tables.** The
   pipeline is linear and each stage writes each column at most once, so a wide
   row is simpler than five joins — and a `NULL` in a stage's columns is the
   stage's own work queue (`scored_at IS NULL`, `tagged_at IS NULL`, …).

Rejection is a flag (`rejected`, `reject_reason`), never a delete. A shot cut
from the social pipeline is still licensable, and the bar may move later.
