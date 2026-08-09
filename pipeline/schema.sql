-- Drone archive pipeline schema.
--
-- Two tables: `sources` is one row per raw file on disk, `shots` is the spine
-- every stage reads from and writes to. A shot is an addressable time range
-- inside a source; raw footage is never modified, so the DB is the only thing
-- that accumulates state as a shot moves 1 -> 5.
--
-- Columns are grouped by the stage that populates them. A NULL in a stage's
-- columns means that stage has not run for that shot yet, which is also how
-- the stage runners find their work.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Stage 1 — ingest
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sources (
    id                INTEGER PRIMARY KEY,
    path              TEXT    NOT NULL UNIQUE,   -- absolute path in the archive
    relpath           TEXT    NOT NULL,          -- path relative to archive root
    size_bytes        INTEGER NOT NULL,
    mtime             REAL    NOT NULL,          -- used to detect re-encodes/replacements

    -- ffprobe
    duration          REAL,
    width             INTEGER,
    height            INTEGER,
    fps               REAL,
    codec             TEXT,
    bitrate           INTEGER,
    aspect            TEXT,                      -- '16:9' | '9:16' | other, derived
    captured_at       TEXT,                      -- creation_time from container metadata
    gps_lat           REAL,
    gps_lon           REAL,
    drone_model       TEXT,                      -- embedded make/model when present

    -- from folder convention: archive/<country>/<file>
    country           TEXT,
    site              TEXT,                      -- blank at ingest, inferred in stage 3

    scene_detected_at TEXT,                      -- NULL = stage 1b still owes this file
    ingested_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    notes             TEXT
);

CREATE INDEX IF NOT EXISTS idx_sources_country ON sources(country);
CREATE INDEX IF NOT EXISTS idx_sources_pending_scenes
    ON sources(scene_detected_at) WHERE scene_detected_at IS NULL;

CREATE TABLE IF NOT EXISTS shots (
    id                INTEGER PRIMARY KEY,
    source_id         INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    shot_index        INTEGER NOT NULL,          -- 0-based position within the source
    in_point          REAL    NOT NULL,          -- seconds
    out_point         REAL    NOT NULL,          -- seconds
    duration          REAL GENERATED ALWAYS AS (out_point - in_point) VIRTUAL,
    continuous_move   INTEGER NOT NULL DEFAULT 0,-- 1 = file kept whole, no cut found

    -- ---------------------------------------------------------------------
    -- Stage 2 — score & filter
    -- ---------------------------------------------------------------------
    motion_score      REAL,                      -- optical-flow magnitude, normalised
    stability_score   REAL,                      -- flow coherence: smooth move vs. jitter
    aesthetic_score   REAL,                      -- CLIP+MLP predictor
    technical_score   REAL,                      -- blur / exposure / artifact check
    scored_at         TEXT,
    rejected          INTEGER NOT NULL DEFAULT 0,-- flag only; nothing is ever deleted
    reject_reason     TEXT,

    -- ---------------------------------------------------------------------
    -- Stage 3 — tag & describe
    -- ---------------------------------------------------------------------
    subject_tags      TEXT,                      -- JSON array: village, coastline, ruins...
    mood_tags         TEXT,                      -- JSON array: solitude, vastness...
    person_in_frame   INTEGER,                   -- 0/1, drives the periodic you-in-frame clip
    description       TEXT,                      -- draft one-liner, doubles as stock caption
    inferred_site     TEXT,                      -- specific location when folders don't say
    tagged_at         TEXT,
    tag_model         TEXT,                      -- which VLM produced the above

    -- ---------------------------------------------------------------------
    -- Stage 4 — select & render
    -- ---------------------------------------------------------------------
    selection_state   TEXT    NOT NULL DEFAULT 'unreviewed',
                                                 -- unreviewed | queued | approved | skipped
    approved_at       TEXT,
    social_render_path    TEXT,                  -- 9:16, overlay burned in, compressed
    licensing_render_path TEXT,                  -- clean 16:9 master, no overlay
    overlay_text      TEXT,                      -- resolved "year · location" line
    license_tier      TEXT,                      -- public | non-exclusive | exclusive
    rendered_at       TEXT,

    -- ---------------------------------------------------------------------
    -- Stage 5 — distribute & learn
    -- ---------------------------------------------------------------------
    caption           TEXT,
    platform_ids      TEXT,                      -- JSON object: {platform: post_id}
    posted_at         TEXT,
    licensed_to       TEXT,                      -- JSON array of marketplace uploads
    metrics           TEXT,                      -- JSON object of latest pulled metrics
    metrics_updated_at TEXT,

    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),

    UNIQUE (source_id, shot_index),
    CHECK (out_point > in_point),
    CHECK (selection_state IN ('unreviewed', 'queued', 'approved', 'skipped')),
    CHECK (license_tier IS NULL
           OR license_tier IN ('public', 'non-exclusive', 'exclusive'))
);

CREATE INDEX IF NOT EXISTS idx_shots_source ON shots(source_id);

-- Stage 2 picks up anything unscored; stages 3+ only ever look at survivors.
CREATE INDEX IF NOT EXISTS idx_shots_unscored ON shots(scored_at) WHERE scored_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_shots_untagged
    ON shots(tagged_at) WHERE tagged_at IS NULL AND rejected = 0;
CREATE INDEX IF NOT EXISTS idx_shots_selection ON shots(selection_state) WHERE rejected = 0;
CREATE INDEX IF NOT EXISTS idx_shots_posted ON shots(posted_at);

-- Convenience view: everything a selection or licensing query needs, already joined
-- to the source so callers don't repeat the join.
CREATE VIEW IF NOT EXISTS shot_details AS
SELECT
    s.*,
    src.path            AS source_path,
    src.relpath         AS source_relpath,
    src.country         AS country,
    COALESCE(s.inferred_site, src.site) AS site,
    src.aspect          AS source_aspect,
    src.width           AS source_width,
    src.height          AS source_height,
    src.fps             AS source_fps,
    src.captured_at     AS captured_at,
    src.drone_model     AS drone_model,
    src.gps_lat         AS gps_lat,
    src.gps_lon         AS gps_lon
FROM shots s
JOIN sources src ON src.id = s.source_id;
