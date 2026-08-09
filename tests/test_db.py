import sqlite3
import tempfile
import unittest
from pathlib import Path

from pipeline import db


class DBTestCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.conn = db.init_db(Path(self._dir.name) / "test.db")
        self.addCleanup(self.conn.close)

    def make_source(self, **overrides):
        values = {
            "path": "/archive/albania/DJI_0001.MP4",
            "relpath": "albania/DJI_0001.MP4",
            "size_bytes": 1024,
            "mtime": 1.0,
            "duration": 30.0,
            "width": 3840,
            "height": 2160,
            "aspect": "16:9",
            "country": "albania",
        }
        values.update(overrides)
        return db.upsert_source(self.conn, **values)


class TestSchema(DBTestCase):
    def test_init_is_idempotent(self):
        db.init_db(Path(self._dir.name) / "test.db").close()
        self.assertEqual(
            self.conn.execute("PRAGMA user_version").fetchone()[0], db.SCHEMA_VERSION
        )

    def test_shot_duration_is_derived(self):
        source_id = self.make_source()
        shot_id = db.replace_shots(self.conn, source_id, [(2.0, 7.5)])[0]
        row = self.conn.execute(
            "SELECT duration FROM shots WHERE id = ?", (shot_id,)
        ).fetchone()
        self.assertAlmostEqual(row["duration"], 5.5)

    def test_rejects_backwards_shot(self):
        source_id = self.make_source()
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO shots (source_id, shot_index, in_point, out_point) "
                "VALUES (?, 0, 10.0, 4.0)",
                (source_id,),
            )

    def test_rejects_unknown_license_tier(self):
        source_id = self.make_source()
        shot_id = db.replace_shots(self.conn, source_id, [(0.0, 5.0)])[0]
        with self.assertRaises(sqlite3.IntegrityError):
            db.update_shot(self.conn, shot_id, license_tier="free-for-all")

    def test_deleting_source_removes_its_shots(self):
        source_id = self.make_source()
        db.replace_shots(self.conn, source_id, [(0.0, 5.0), (5.0, 9.0)])
        self.conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        self.assertEqual(self.conn.execute("SELECT count(*) FROM shots").fetchone()[0], 0)


class TestIngestWrites(DBTestCase):
    def test_upsert_source_is_idempotent_on_path(self):
        first = self.make_source()
        second = self.make_source(size_bytes=2048, duration=31.0)
        self.assertEqual(first, second)
        row = self.conn.execute(
            "SELECT * FROM sources WHERE id = ?", (first,)
        ).fetchone()
        self.assertEqual(row["size_bytes"], 2048)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM sources").fetchone()[0], 1)

    def test_single_segment_is_flagged_continuous(self):
        source_id = self.make_source()
        shot_id = db.replace_shots(self.conn, source_id, [(0.0, 30.0)])[0]
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertEqual(row["continuous_move"], 1)

    def test_multiple_segments_are_not_continuous(self):
        source_id = self.make_source()
        ids = db.replace_shots(self.conn, source_id, [(0.0, 10.0), (10.0, 30.0)])
        rows = self.conn.execute(
            "SELECT continuous_move FROM shots ORDER BY shot_index"
        ).fetchall()
        self.assertEqual(len(ids), 2)
        self.assertEqual([r["continuous_move"] for r in rows], [0, 0])

    def test_replace_shots_clears_previous_detection(self):
        source_id = self.make_source()
        db.replace_shots(self.conn, source_id, [(0.0, 10.0), (10.0, 30.0)])
        db.replace_shots(self.conn, source_id, [(0.0, 30.0)])
        rows = self.conn.execute("SELECT * FROM shots").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["shot_index"], 0)

    def test_replace_shots_marks_source_detected(self):
        source_id = self.make_source()
        self.assertEqual(len(db.sources_pending_scene_detection(self.conn)), 1)
        db.replace_shots(self.conn, source_id, [(0.0, 30.0)])
        self.assertEqual(db.sources_pending_scene_detection(self.conn), [])


class TestJSONColumns(DBTestCase):
    def test_round_trip(self):
        source_id = self.make_source()
        shot_id = db.replace_shots(self.conn, source_id, [(0.0, 5.0)])[0]
        db.update_shot(
            self.conn,
            shot_id,
            subject_tags=["coastline", "ruins"],
            metrics={"views": 1200},
        )
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertEqual(row["subject_tags"], ["coastline", "ruins"])
        self.assertEqual(row["metrics"], {"views": 1200})

    def test_stored_as_json_text(self):
        source_id = self.make_source()
        shot_id = db.replace_shots(self.conn, source_id, [(0.0, 5.0)])[0]
        db.update_shot(self.conn, shot_id, mood_tags=["solitude"])
        raw = self.conn.execute(
            "SELECT json_extract(mood_tags, '$[0]') AS first FROM shots WHERE id = ?",
            (shot_id,),
        ).fetchone()
        self.assertEqual(raw["first"], "solitude")

    def test_null_json_column_stays_null(self):
        source_id = self.make_source()
        shot_id = db.replace_shots(self.conn, source_id, [(0.0, 5.0)])[0]
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertIsNone(row["subject_tags"])


class TestStageQueues(DBTestCase):
    def setUp(self):
        super().setUp()
        source_id = self.make_source()
        self.unscored, self.rejected, self.survivor = db.replace_shots(
            self.conn, source_id, [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0)]
        )
        db.update_shot(self.conn, self.rejected, scored_at="2026-01-01", aesthetic_score=0.1)
        db.update_shot(self.conn, self.survivor, scored_at="2026-01-01", aesthetic_score=0.9)
        db.reject(self.conn, [self.rejected], "bottom percentile")
        self.conn.commit()

    def test_scoring_queue_is_only_unscored(self):
        rows = db.shots_pending_scoring(self.conn)
        self.assertEqual([r["id"] for r in rows], [self.unscored])

    def test_tagging_queue_skips_rejects_and_unscored(self):
        rows = db.shots_pending_tagging(self.conn)
        self.assertEqual([r["id"] for r in rows], [self.survivor])

    def test_reject_flags_rather_than_deletes(self):
        row = self.conn.execute(
            "SELECT * FROM shots WHERE id = ?", (self.rejected,)
        ).fetchone()
        self.assertEqual(row["rejected"], 1)
        self.assertEqual(row["reject_reason"], "bottom percentile")

    def test_selection_queue_needs_tags(self):
        self.assertEqual(db.selection_queue(self.conn), [])
        db.update_shot(self.conn, self.survivor, tagged_at="2026-01-02")
        rows = db.selection_queue(self.conn)
        self.assertEqual([r["id"] for r in rows], [self.survivor])

    def test_selection_queue_drops_posted_shots(self):
        db.update_shot(
            self.conn, self.survivor, tagged_at="2026-01-02", posted_at="2026-01-03"
        )
        self.assertEqual(db.selection_queue(self.conn), [])

    def test_shot_details_joins_source_fields(self):
        rows = db.shots_pending_tagging(self.conn)
        self.assertEqual(rows[0]["country"], "albania")
        self.assertEqual(rows[0]["source_aspect"], "16:9")

    def test_inferred_site_overrides_folder_site(self):
        self.conn.execute("UPDATE sources SET site = 'unknown'")
        db.update_shot(self.conn, self.survivor, inferred_site="Ksamil")
        rows = db.shots_pending_tagging(self.conn)
        self.assertEqual(rows[0]["site"], "Ksamil")

    def test_stats_counts_each_stage(self):
        counts = db.stats(self.conn)
        self.assertEqual(counts["sources"], 1)
        self.assertEqual(counts["shots"], 3)
        self.assertEqual(counts["scored"], 2)
        self.assertEqual(counts["rejected"], 1)
        self.assertEqual(counts["tagged"], 0)


class TestTransaction(DBTestCase):
    def test_rolls_back_on_error(self):
        source_id = self.make_source()
        shot_id = db.replace_shots(self.conn, source_id, [(0.0, 5.0)])[0]
        with self.assertRaises(RuntimeError):
            with db.transaction(self.conn):
                db.update_shot(self.conn, shot_id, description="written")
                raise RuntimeError("boom")
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertIsNone(row["description"])


if __name__ == "__main__":
    unittest.main()
