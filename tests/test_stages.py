"""Stage runner tests.

Everything that would shell out to ffmpeg or call an API is substituted, so the
suite exercises the control flow — what gets picked up, what gets written back,
what a failure does to the rest of the batch — without the toolchain.
"""

import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from pipeline import db
from pipeline.config import config as base_config
from pipeline.stages import distribute, ingest, scene_detect, select_render, tag


class StageTestCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)
        self.conn = db.init_db(self.root / "test.db")
        self.addCleanup(self.conn.close)
        self.cfg = replace(base_config, archive_root=self.root / "archive")

    def add_source(self, **overrides):
        values = {
            "path": str(self.root / "archive/albania/DJI_0001.MP4"),
            "relpath": "albania/DJI_0001.MP4",
            "size_bytes": 1024,
            "mtime": 1.0,
            "duration": 30.0,
            "width": 3840,
            "height": 2160,
            "aspect": "16:9",
            "country": "albania",
            "captured_at": "2024-06-01T08:30:00Z",
        }
        values.update(overrides)
        with db.transaction(self.conn):
            return db.upsert_source(self.conn, **values)

    def add_shot(self, source_id=None, *, scored=True, tagged=True, **fields):
        source_id = source_id if source_id is not None else self.add_source()
        index = self.conn.execute(
            "SELECT count(*) FROM shots WHERE source_id = ?", (source_id,)
        ).fetchone()[0]
        cursor = self.conn.execute(
            "INSERT INTO shots (source_id, shot_index, in_point, out_point) "
            "VALUES (?, ?, ?, ?)",
            (source_id, index, 0.0 + index * 10, 10.0 + index * 10),
        )
        shot_id = int(cursor.lastrowid)
        values = {}
        if scored:
            values.update(scored_at="2026-01-01", aesthetic_score=0.8, technical_score=0.8)
        if tagged:
            values.update(
                tagged_at="2026-01-02",
                subject_tags=["coastline"],
                mood_tags=["solitude"],
                description="A coastline.",
                person_in_frame=0,
            )
        values.update(fields)
        with db.transaction(self.conn):
            db.update_shot(self.conn, shot_id, **values)
        return shot_id


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

PROBE = {
    "duration": 30.0, "width": 3840, "height": 2160, "fps": 30.0,
    "codec": "hevc", "bitrate": 1, "aspect": "16:9", "captured_at": None,
    "gps_lat": None, "gps_lon": None, "drone_model": "DJI",
}


class TestIngest(StageTestCase):
    def make_file(self, relpath, content=b"x" * 16):
        path = self.cfg.archive_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_country_from_folder(self):
        root = Path("/archive")
        self.assertEqual(ingest.country_from_path(Path("/archive/peru/a.mp4"), root), "peru")

    def test_country_normalises_underscores(self):
        root = Path("/archive")
        self.assertEqual(
            ingest.country_from_path(Path("/archive/New_Zealand/a.mp4"), root), "new zealand"
        )

    def test_file_at_root_has_no_country(self):
        self.assertIsNone(ingest.country_from_path(Path("/archive/a.mp4"), Path("/archive")))

    def test_only_video_extensions_walked(self):
        self.make_file("albania/a.mp4")
        self.make_file("albania/notes.txt")
        self.make_file("albania/.DS_Store")
        found = list(ingest.iter_video_files(self.cfg.archive_root, self.cfg.video_extensions))
        self.assertEqual([p.name for p in found], ["a.mp4"])

    def test_extension_match_is_case_insensitive(self):
        self.make_file("albania/A.MOV")
        found = list(ingest.iter_video_files(self.cfg.archive_root, self.cfg.video_extensions))
        self.assertEqual(len(found), 1)

    def test_registers_files_with_country(self):
        self.make_file("albania/a.mp4")
        self.make_file("peru/b.mp4")
        with mock.patch.object(ingest.media, "probe", return_value=dict(PROBE)):
            counts = ingest.run(self.conn, self.cfg)
        self.assertEqual(counts["added"], 2)
        countries = {
            r["country"] for r in self.conn.execute("SELECT country FROM sources").fetchall()
        }
        self.assertEqual(countries, {"albania", "peru"})

    def test_rerun_skips_unchanged_files(self):
        self.make_file("albania/a.mp4")
        with mock.patch.object(ingest.media, "probe", return_value=dict(PROBE)) as probe:
            ingest.run(self.conn, self.cfg)
            counts = ingest.run(self.conn, self.cfg)
        self.assertEqual(counts["skipped"], 1)
        self.assertEqual(probe.call_count, 1)

    def test_changed_file_is_resegmented(self):
        path = self.make_file("albania/a.mp4")
        with mock.patch.object(ingest.media, "probe", return_value=dict(PROBE)):
            ingest.run(self.conn, self.cfg)
            source_id = self.conn.execute("SELECT id FROM sources").fetchone()[0]
            db.replace_shots(self.conn, source_id, [(0.0, 30.0)])
            path.write_bytes(b"y" * 64)
            counts = ingest.run(self.conn, self.cfg)
        self.assertEqual(counts["updated"], 1)
        self.assertEqual(len(db.sources_pending_scene_detection(self.conn)), 1)

    def test_force_reprobe_keeps_downstream_state(self):
        """--force re-reads metadata; it must not discard shots and their scores."""
        self.make_file("albania/a.mp4")
        with mock.patch.object(ingest.media, "probe", return_value=dict(PROBE)):
            ingest.run(self.conn, self.cfg)
            source_id = self.conn.execute("SELECT id FROM sources").fetchone()[0]
            db.replace_shots(self.conn, source_id, [(0.0, 30.0)])
            ingest.run(self.conn, self.cfg, force=True)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM shots").fetchone()[0], 1)
        self.assertEqual(db.sources_pending_scene_detection(self.conn), [])

    def test_probe_failure_does_not_stall_the_walk(self):
        self.make_file("albania/bad.mp4")
        self.make_file("albania/good.mp4")
        with mock.patch.object(
            ingest.media, "probe",
            side_effect=[ingest.media.MediaError("boom"), dict(PROBE)],
        ):
            counts = ingest.run(self.conn, self.cfg)
        self.assertEqual((counts["failed"], counts["added"]), (1, 1))

    def test_missing_archive_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            ingest.run(self.conn, self.cfg)


class TestSceneDetect(StageTestCase):
    def test_short_segment_absorbed_into_neighbour(self):
        merged = scene_detect.merge_short_segments([(0.0, 1.0), (1.0, 20.0)], 3.0)
        self.assertEqual(merged, [(0.0, 20.0)])

    def test_short_tail_folded_back(self):
        merged = scene_detect.merge_short_segments([(0.0, 20.0), (20.0, 21.0)], 3.0)
        self.assertEqual(merged, [(0.0, 21.0)])

    def test_merging_preserves_the_full_timeline(self):
        segments = [(0.0, 1.0), (1.0, 2.0), (2.0, 30.0), (30.0, 31.0)]
        merged = scene_detect.merge_short_segments(segments, 3.0)
        self.assertEqual(merged[0][0], 0.0)
        self.assertEqual(merged[-1][1], 31.0)
        for (_, end), (start, _) in zip(merged, merged[1:]):
            self.assertEqual(end, start)

    def test_long_segments_untouched(self):
        segments = [(0.0, 10.0), (10.0, 20.0)]
        self.assertEqual(scene_detect.merge_short_segments(segments, 3.0), segments)

    def test_empty_input(self):
        self.assertEqual(scene_detect.merge_short_segments([], 3.0), [])

    def test_run_writes_shots_and_marks_source(self):
        self.add_source()
        with mock.patch.object(
            scene_detect, "detect_segments", return_value=[(0.0, 12.0), (12.0, 30.0)]
        ):
            counts = scene_detect.run(self.conn, self.cfg)
        self.assertEqual((counts["sources"], counts["shots"]), (1, 2))
        self.assertEqual(db.sources_pending_scene_detection(self.conn), [])

    def test_single_segment_marked_continuous(self):
        self.add_source()
        with mock.patch.object(scene_detect, "detect_segments", return_value=[(0.0, 30.0)]):
            counts = scene_detect.run(self.conn, self.cfg)
        self.assertEqual(counts["continuous"], 1)

    def test_detector_failure_leaves_source_pending(self):
        self.add_source()
        with mock.patch.object(scene_detect, "detect_segments", side_effect=RuntimeError):
            counts = scene_detect.run(self.conn, self.cfg)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(len(db.sources_pending_scene_detection(self.conn)), 1)

    def test_source_without_duration_skipped(self):
        self.add_source(duration=None)
        counts = scene_detect.run(self.conn, self.cfg)
        self.assertEqual(counts["failed"], 1)


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

TAG_RESULT = {
    "subject_tags": ["coastline", "village"],
    "mood_tags": ["solitude"],
    "person_in_frame": True,
    "description": "A village above a coastline, slow push in.",
    "inferred_site": "Ksamil",
}


class TestTag(StageTestCase):
    def setUp(self):
        super().setUp()
        self.frames = [self.root / "f0.jpg"]
        for frame in self.frames:
            frame.write_bytes(b"\xff\xd8\xff")

    def run_stage(self, result=TAG_RESULT, **kwargs):
        tagger = mock.Mock(return_value=result, model="test-model")
        with mock.patch.object(tag.media, "extract_frames", return_value=self.frames):
            counts = tag.run(self.conn, self.cfg, tagger=tagger, **kwargs)
        return counts, tagger

    def test_writes_tags_back_to_the_shot(self):
        shot_id = self.add_shot(tagged=False)
        counts, _ = self.run_stage()
        self.assertEqual(counts["tagged"], 1)
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertEqual(row["subject_tags"], ["coastline", "village"])
        self.assertEqual(row["person_in_frame"], 1)
        self.assertEqual(row["inferred_site"], "Ksamil")
        self.assertEqual(row["tag_model"], "test-model")

    def test_skips_rejected_shots(self):
        self.add_shot(tagged=False, rejected=1)
        counts, tagger = self.run_stage()
        self.assertEqual(counts["tagged"], 0)
        tagger.assert_not_called()

    def test_skips_already_tagged_shots(self):
        self.add_shot(tagged=True)
        counts, tagger = self.run_stage()
        tagger.assert_not_called()
        self.assertEqual(counts["tagged"], 0)

    def test_passes_archive_context_to_the_model(self):
        self.add_shot(tagged=False)
        _, tagger = self.run_stage()
        context = tagger.call_args.args[1]
        self.assertEqual(context["country"], "albania")
        self.assertEqual(context["captured_at"], "2024-06-01T08:30:00Z")

    def test_one_failure_does_not_stall_the_batch(self):
        source_id = self.add_source()
        self.add_shot(source_id, tagged=False)
        self.add_shot(source_id, tagged=False)
        tagger = mock.Mock(side_effect=[RuntimeError("refused"), TAG_RESULT], model="m")
        with mock.patch.object(tag.media, "extract_frames", return_value=self.frames):
            counts = tag.run(self.conn, self.cfg, tagger=tagger)
        self.assertEqual((counts["failed"], counts["tagged"]), (1, 1))

    def test_null_site_stored_as_null(self):
        shot_id = self.add_shot(tagged=False)
        self.run_stage({**TAG_RESULT, "inferred_site": None})
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertIsNone(row["inferred_site"])

    def test_schema_enumerates_the_controlled_vocabulary(self):
        """Free-form tags would make the catalog unqueryable at scale."""
        properties = tag.RESPONSE_SCHEMA["properties"]
        self.assertEqual(properties["mood_tags"]["items"]["enum"], tag.MOOD_TAGS)
        self.assertFalse(tag.RESPONSE_SCHEMA["additionalProperties"])
        self.assertEqual(set(tag.RESPONSE_SCHEMA["required"]), set(properties))


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------


class TestSelection(StageTestCase):
    def test_queue_holds_only_tagged_survivors(self):
        wanted = self.add_shot()
        self.add_shot(tagged=False)
        self.add_shot(rejected=1)
        proposed = select_render.propose_queue(self.conn, 10)
        self.assertEqual([row["id"] for row, _ in proposed], [wanted])

    def test_recently_posted_country_is_skipped(self):
        albania = self.add_source()
        peru = self.add_source(path="/archive/peru/x.mp4", relpath="peru/x.mp4", country="peru")
        self.add_shot(albania, posted_at="2026-02-01", selection_state="approved")
        self.add_shot(albania)
        peru_shot = self.add_shot(peru)
        proposed = select_render.propose_queue(self.conn, 1)
        self.assertEqual(proposed[0][0]["id"], peru_shot)

    def test_one_shot_per_country_within_a_batch(self):
        albania = self.add_source()
        self.add_shot(albania)
        self.add_shot(albania)
        proposed = select_render.propose_queue(self.conn, 5)
        countries = [row["country"] for row, reason in proposed if "new country" in reason]
        self.assertEqual(len(countries), len(set(countries)))

    def test_falls_back_rather_than_returning_nothing(self):
        """A thin archive must still produce a queue to review."""
        albania = self.add_source()
        self.add_shot(albania)
        self.add_shot(albania)
        self.add_shot(albania)
        proposed = select_render.propose_queue(self.conn, 3)
        self.assertEqual(len(proposed), 3)
        self.assertTrue(any("diversity rules exhausted" in r for _, r in proposed))

    def test_every_pick_carries_a_reason(self):
        self.add_shot()
        for _, reason in select_render.propose_queue(self.conn, 5):
            self.assertTrue(reason)

    def test_person_in_frame_slot(self):
        """Every PERSON_EVERY posts, only a you-in-frame shot should surface."""
        source = self.add_source()
        for _ in range(select_render.PERSON_EVERY):
            self.add_shot(source, posted_at="2026-02-01", selection_state="approved")
        plain = self.add_shot(source, person_in_frame=0)
        with_person = self.add_shot(source, person_in_frame=1)
        picked = [row["id"] for row, r in select_render.propose_queue(self.conn, 1)]
        self.assertIn(with_person, picked)
        self.assertNotIn(plain, picked)

    def test_staging_marks_shots_queued(self):
        shot_id = self.add_shot()
        select_render.stage_queue(self.conn, 5)
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertEqual(row["selection_state"], "queued")

    def test_approval_records_a_timestamp(self):
        shot_id = self.add_shot()
        select_render.approve(self.conn, [shot_id])
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertEqual(row["selection_state"], "approved")
        self.assertIsNotNone(row["approved_at"])

    def test_overlay_uses_capture_year_and_place(self):
        shot_id = self.add_shot(inferred_site="Ksamil")
        row = self.conn.execute(
            "SELECT * FROM shot_details WHERE id = ?", (shot_id,)
        ).fetchone()
        self.assertEqual(
            select_render.overlay_for(row, self.cfg), ["Found this.", "2024 · Ksamil · albania"]
        )


class TestRenderStage(StageTestCase):
    def approved_shot(self, **fields):
        return self.add_shot(selection_state="approved", approved_at="2026-02-01", **fields)

    def test_renders_both_profiles_and_records_paths(self):
        shot_id = self.approved_shot()
        with mock.patch.object(select_render.media, "run") as run:
            counts = select_render.run(self.conn, self.cfg)
        self.assertEqual(counts["rendered"], 1)
        self.assertEqual(run.call_count, 2)
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertIsNotNone(row["social_render_path"])
        self.assertIsNotNone(row["licensing_render_path"])
        self.assertIsNotNone(row["rendered_at"])

    def test_single_profile(self):
        self.approved_shot()
        with mock.patch.object(select_render.media, "run") as run:
            select_render.run(self.conn, self.cfg, profiles=("social",))
        self.assertEqual(run.call_count, 1)

    def test_unapproved_shots_are_not_rendered(self):
        self.add_shot()
        with mock.patch.object(select_render.media, "run") as run:
            counts = select_render.run(self.conn, self.cfg)
        run.assert_not_called()
        self.assertEqual(counts["rendered"], 0)

    def test_already_rendered_shots_skipped(self):
        self.approved_shot(rendered_at="2026-02-02", social_render_path="/x.mp4")
        with mock.patch.object(select_render.media, "run") as run:
            select_render.run(self.conn, self.cfg)
        run.assert_not_called()

    def test_ffmpeg_failure_leaves_shot_unrendered(self):
        shot_id = self.approved_shot()
        with mock.patch.object(
            select_render.media, "run", side_effect=select_render.media.MediaError("boom")
        ):
            counts = select_render.run(self.conn, self.cfg)
        self.assertEqual(counts["failed"], 1)
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertIsNone(row["rendered_at"])


# ---------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------




class FakeClient:
    """Stands in for ZernioClient, returning the response shape Zernio uses."""

    def __init__(self, post_id="zp_1", statuses=None):
        self.post_id = post_id
        self.statuses = statuses if statuses is not None else {"instagram": "published"}
        self.published = []

    def _post(self):
        return {
            "_id": self.post_id,
            "platforms": [
                {"platform": p, "accountId": f"acct_{p}", "status": s}
                for p, s in self.statuses.items()
            ],
        }

    def publish(self, video_path, caption, *, platforms=None):
        self.published.append((Path(video_path), caption, platforms))
        return {"post": self._post()}

    def fetch_post(self, post_id):
        return self._post() if post_id == self.post_id else None


class TestDistribute(StageTestCase):
    def postable_shot(self, **fields):
        return self.add_shot(
            selection_state="approved",
            approved_at="2026-02-01",
            social_render_path=str(self.root / "clip.mp4"),
            rendered_at="2026-02-01",
            **fields,
        )

    def test_caption_includes_place_description_and_tags(self):
        shot_id = self.add_shot(inferred_site="Ksamil")
        row = self.conn.execute(
            "SELECT * FROM shot_details WHERE id = ?", (shot_id,)
        ).fetchone()
        caption = distribute.build_caption(row, self.cfg)
        self.assertIn("Found this.", caption)
        self.assertIn("Ksamil", caption)
        self.assertIn("#solitude", caption)

    def test_caption_hashtags_drop_hyphens(self):
        shot_id = self.add_shot(mood_tags=["golden-hour"])
        row = self.conn.execute(
            "SELECT * FROM shot_details WHERE id = ?", (shot_id,)
        ).fetchone()
        self.assertIn("#goldenhour", distribute.build_caption(row, self.cfg))

    def test_publish_records_zernio_post_id_and_accounts(self):
        shot_id = self.postable_shot()
        counts = distribute.publish_pending(self.conn, self.cfg, client=FakeClient())
        self.assertEqual(counts["posted"], 1)
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertEqual(row["zernio_post_id"], "zp_1")
        self.assertEqual(row["platform_ids"], {"instagram": "acct_instagram"})
        self.assertEqual(row["delivery_status"], {"instagram": "published"})
        self.assertIsNotNone(row["posted_at"])

    def test_publish_targets_the_configured_platforms(self):
        self.postable_shot()
        client = FakeClient()
        distribute.publish_pending(
            self.conn, self.cfg, client=client, platforms=("tiktok",)
        )
        self.assertEqual(client.published[0][2], ("tiktok",))

    def test_unrendered_shots_are_not_published(self):
        self.add_shot(selection_state="approved")
        client = FakeClient()
        distribute.publish_pending(self.conn, self.cfg, client=client)
        self.assertEqual(client.published, [])

    def test_dry_run_leaves_the_shot_in_the_queue(self):
        """No post id back means nothing was posted — don't mark it as posted."""
        shot_id = self.postable_shot()
        counts = distribute.publish_pending(
            self.conn, self.cfg, client=distribute.DryRunClient()
        )
        self.assertEqual(counts["posted"], 0)
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertIsNone(row["posted_at"])

    def test_publish_failure_leaves_shot_unposted(self):
        shot_id = self.postable_shot()
        client = FakeClient()
        client.publish = mock.Mock(side_effect=RuntimeError("502"))
        counts = distribute.publish_pending(self.conn, self.cfg, client=client)
        self.assertEqual(counts["failed"], 1)
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertIsNone(row["posted_at"])

    def test_sync_surfaces_a_late_platform_failure(self):
        """A post the API accepted can still fail at the platform minutes later."""
        shot_id = self.postable_shot()
        distribute.publish_pending(self.conn, self.cfg, client=FakeClient())
        counts = distribute.sync_delivery(
            self.conn, self.cfg,
            client=FakeClient(statuses={"instagram": "failed: media rejected"}),
        )
        self.assertEqual(counts["failed_delivery"], 1)
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertIn("failed", row["delivery_status"]["instagram"])

    def test_sync_skips_shots_never_posted(self):
        self.postable_shot()
        counts = distribute.sync_delivery(self.conn, self.cfg, client=FakeClient())
        self.assertEqual(counts["checked"], 0)

    def test_delivery_status_is_kept_apart_from_metrics(self):
        """Zernio reports delivery, not performance; the feedback loop needs both."""
        shot_id = self.postable_shot()
        distribute.publish_pending(self.conn, self.cfg, client=FakeClient())
        row = self.conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        self.assertIsNotNone(row["delivery_status"])
        self.assertIsNone(row["metrics"])


if __name__ == "__main__":
    unittest.main()


class TestScoreConcurrency(StageTestCase):
    """Parallel scoring must agree with serial scoring, exactly.

    Measurement runs on worker threads; every database write is supposed to
    happen on the calling thread, because SQLite connections are not safe to
    share. These assert both halves of that.
    """

    def setUp(self):
        super().setUp()
        from pipeline import scoring
        from pipeline.stages import score as score_stage

        self.score_stage = score_stage
        self.scoring = scoring
        source = self.add_source()
        self.shot_ids = [
            self.add_shot(source, scored=False, tagged=False) for _ in range(12)
        ]

    def fake_scorer(self, record=None):
        """Deterministic per-shot scores, derived from the shot's in_point."""
        def score_shot(row, cfg, aesthetic):
            if record is not None:
                record.append(threading.current_thread().name)
            value = round(0.5 + row["in_point"] / 1000, 4)
            return self.scoring.Scores(
                motion=value, stability=value, technical=value, aesthetic=None
            )
        return score_shot

    def scores_in_db(self):
        return {
            row["id"]: row["motion_score"]
            for row in self.conn.execute(
                "SELECT id, motion_score FROM shots ORDER BY id"
            ).fetchall()
        }

    def test_parallel_matches_serial(self):
        with mock.patch.object(self.score_stage, "score_shot", self.fake_scorer()):
            self.score_stage.score_pending(self.conn, self.cfg, workers=4)
        parallel = self.scores_in_db()

        with db.transaction(self.conn):
            for shot_id in self.shot_ids:
                db.update_shot(self.conn, shot_id, scored_at=None, motion_score=None)
        with mock.patch.object(self.score_stage, "score_shot", self.fake_scorer()):
            self.score_stage.score_pending(self.conn, self.cfg, workers=1)

        self.assertEqual(parallel, self.scores_in_db())

    def test_every_shot_is_scored_exactly_once(self):
        with mock.patch.object(self.score_stage, "score_shot", self.fake_scorer()):
            counts = self.score_stage.score_pending(self.conn, self.cfg, workers=4)
        self.assertEqual(counts["scored"], len(self.shot_ids))
        self.assertEqual(
            self.conn.execute(
                "SELECT count(*) FROM shots WHERE scored_at IS NULL"
            ).fetchone()[0],
            0,
        )

    def test_work_really_runs_on_worker_threads(self):
        seen: list[str] = []
        with mock.patch.object(self.score_stage, "score_shot", self.fake_scorer(seen)):
            self.score_stage.score_pending(self.conn, self.cfg, workers=4)
        self.assertGreater(len(set(seen)), 1, "expected more than one worker thread")

    def test_database_writes_stay_on_the_calling_thread(self):
        """A write from a worker thread would raise ProgrammingError under
        SQLite's default check_same_thread, intermittently."""
        main_thread = threading.current_thread().name
        writing_threads: list[str] = []
        original = db.update_shot

        def spy(conn, shot_id, **values):
            writing_threads.append(threading.current_thread().name)
            return original(conn, shot_id, **values)

        with mock.patch.object(self.score_stage, "score_shot", self.fake_scorer()), \
             mock.patch.object(self.score_stage.db, "update_shot", spy):
            self.score_stage.score_pending(self.conn, self.cfg, workers=4)

        self.assertTrue(writing_threads)
        self.assertEqual(set(writing_threads), {main_thread})

    def test_one_failure_does_not_stall_the_pool(self):
        def flaky(row, cfg, aesthetic):
            if row["id"] == self.shot_ids[3]:
                raise RuntimeError("unreadable file")
            return self.scoring.Scores(motion=0.5, stability=0.5, technical=0.5)

        with mock.patch.object(self.score_stage, "score_shot", flaky):
            counts = self.score_stage.score_pending(self.conn, self.cfg, workers=4)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["scored"], len(self.shot_ids) - 1)

    def test_nothing_to_do_is_cheap(self):
        with db.transaction(self.conn):
            for shot_id in self.shot_ids:
                db.update_shot(self.conn, shot_id, scored_at="2026-01-01")
        with mock.patch.object(self.score_stage, "score_shot") as scorer:
            counts = self.score_stage.score_pending(self.conn, self.cfg, workers=4)
        scorer.assert_not_called()
        self.assertEqual(counts, {"scored": 0, "failed": 0})
