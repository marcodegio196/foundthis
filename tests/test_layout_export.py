"""Local output layout and the licensing export.

Everything the pipeline writes lands in a browsable folder tree; nothing it
writes ever touches the archive. Both are asserted here.
"""

import csv
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pipeline import db, layout
from pipeline.config import config as base_config
from pipeline.stages import export_catalog


class TestSlugify(unittest.TestCase):
    def test_lowercases_and_hyphenates(self):
        self.assertEqual(layout.slugify("New Zealand", "x"), "new-zealand")

    def test_strips_characters_windows_forbids(self):
        self.assertEqual(layout.slugify('a:b/c\\d?e*f"g', "x"), "abcdefg")

    def test_empty_falls_back(self):
        self.assertEqual(layout.slugify("   ", "unknown"), "unknown")
        self.assertEqual(layout.slugify(None, "unknown"), "unknown")

    def test_reserved_windows_names_are_escaped(self):
        """A folder literally named CON is uncreatable on Windows."""
        self.assertEqual(layout.slugify("con", "x"), "con-")
        self.assertEqual(layout.slugify("LPT1", "x"), "lpt1-")

    def test_trailing_dot_removed(self):
        """Windows silently strips trailing dots, which breaks path round-trips."""
        self.assertFalse(layout.slugify("site.", "x").endswith("."))

    def test_length_bounded(self):
        self.assertLessEqual(len(layout.slugify("x" * 300, "y")), 64)


class TestYearAndDirs(unittest.TestCase):
    def test_year_from_timestamp(self):
        self.assertEqual(layout.year_of("2024-06-01T08:30:00Z"), "2024")

    def test_undated_footage_gets_its_own_folder(self):
        """Better an honest gap than a guessed year in a licensing catalog."""
        self.assertEqual(layout.year_of(None), "undated")
        self.assertEqual(layout.year_of("not-a-date"), "undated")

    def test_country_year_tree(self):
        self.assertEqual(
            layout.shot_dir("Albania", "2024-06-01T00:00:00Z"), Path("albania/2024")
        )

    def test_missing_country(self):
        self.assertEqual(
            layout.shot_dir(None, None), Path("unknown-country/undated")
        )

    def test_stem_leads_with_padded_id(self):
        stem = layout.shot_stem(12, "albania/DJI_0001.MP4")
        self.assertTrue(stem.startswith("000012_"))
        self.assertIn("dji_0001", stem)

    def test_stem_includes_site_when_known(self):
        self.assertIn("ksamil", layout.shot_stem(12, "a/b.mp4", site="Ksamil"))

    def test_full_render_path(self):
        path = layout.render_path(
            Path("/renders/social"),
            shot_id=7,
            source_relpath="albania/DJI_0009.MP4",
            country="albania",
            captured_at="2024-06-01T00:00:00Z",
        )
        self.assertEqual(path.parent, Path("/renders/social/albania/2024"))
        self.assertEqual(path.suffix, ".mp4")

    def test_paths_are_stable_across_calls(self):
        """Re-rendering must overwrite, not accumulate a second file."""
        args = dict(
            shot_id=7, source_relpath="a/b.mp4", country="peru", captured_at="2023-01-01"
        )
        self.assertEqual(
            layout.render_path(Path("/r"), **args), layout.render_path(Path("/r"), **args)
        )


class ExportTestCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)
        self.conn = db.init_db(self.root / "test.db")
        self.addCleanup(self.conn.close)
        self.cfg = replace(base_config, export_root=self.root / "exports")

        self.master = self.root / "renders/licensing/albania/2024/000001_dji.mp4"
        self.master.parent.mkdir(parents=True, exist_ok=True)
        self.master.write_bytes(b"video-bytes")

        with db.transaction(self.conn):
            source_id = db.upsert_source(
                self.conn,
                path=str(self.root / "archive/albania/DJI_0001.MP4"),
                relpath="albania/DJI_0001.MP4",
                size_bytes=1024, mtime=1.0, duration=30.0,
                width=3840, height=2160, fps=29.97, aspect="16:9",
                country="albania", captured_at="2024-06-01T08:30:00Z",
                gps_lat=39.7, gps_lon=20.0, drone_model="DJI FC3582",
            )
        self.shot_id = db.replace_shots(self.conn, source_id, [(0.0, 8.0)])[0]
        with db.transaction(self.conn):
            db.update_shot(
                self.conn, self.shot_id,
                scored_at="2026-01-01", aesthetic_score=0.82, technical_score=0.77,
                tagged_at="2026-01-02",
                subject_tags=["coastline", "village"],
                mood_tags=["solitude", "golden-hour"],
                description="A village above a coastline, slow push in.",
                inferred_site="Ksamil",
                person_in_frame=0,
                license_tier="non-exclusive",
                licensing_render_path=str(self.master),
                rendered_at="2026-01-03",
            )

    def row(self):
        return self.conn.execute(
            "SELECT * FROM shot_details WHERE id = ?", (self.shot_id,)
        ).fetchone()


class TestMetadata(ExportTestCase):
    def test_keywords_merge_tags_and_place(self):
        keywords = export_catalog.keywords_for(self.row())
        for expected in ("coastline", "solitude", "golden hour", "albania", "ksamil"):
            self.assertIn(expected, keywords)

    def test_keywords_note_aerial_and_resolution(self):
        keywords = export_catalog.keywords_for(self.row())
        self.assertIn("aerial", keywords)
        self.assertIn("4k", keywords)

    def test_keywords_deduplicated(self):
        keywords = export_catalog.keywords_for(self.row())
        self.assertEqual(len(keywords), len(set(keywords)))

    def test_title_from_first_clause(self):
        self.assertEqual(
            export_catalog.title_for(self.row()), "A village above a coastline"
        )

    def test_title_falls_back_to_place(self):
        with db.transaction(self.conn):
            db.update_shot(self.conn, self.shot_id, description=None)
        self.assertEqual(export_catalog.title_for(self.row()), "Ksamil · Albania")

    def test_sidecar_carries_provenance(self):
        sidecar = export_catalog.sidecar_for(self.row())
        self.assertEqual(sidecar["provenance"]["source_file"], "albania/DJI_0001.MP4")
        self.assertEqual(sidecar["provenance"]["in_point"], 0.0)

    def test_sidecar_carries_gps_and_drone(self):
        sidecar = export_catalog.sidecar_for(self.row())
        self.assertEqual(sidecar["location"]["latitude"], 39.7)
        self.assertEqual(sidecar["capture"]["drone_model"], "DJI FC3582")


class TestExportRun(ExportTestCase):
    def test_writes_master_sidecar_and_manifest(self):
        counts = export_catalog.run(self.conn, self.cfg)
        self.assertEqual(counts["exported"], 1)

        exported = self.cfg.export_root / "albania/2024/000001_dji_0001_ksamil.mp4"
        self.assertTrue(exported.is_file())
        self.assertEqual(exported.read_bytes(), b"video-bytes")
        self.assertTrue(exported.with_suffix(".json").is_file())

        sidecar = json.loads(exported.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(sidecar["shot_id"], self.shot_id)

        manifest = self.cfg.export_root / export_catalog.MANIFEST_NAME
        with manifest.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["country"], "albania")
        self.assertIn("coastline", rows[0]["keywords"])

    def test_manifest_paths_use_forward_slashes(self):
        """A manifest opened on another machine shouldn't carry Windows separators."""
        export_catalog.run(self.conn, self.cfg)
        manifest = self.cfg.export_root / export_catalog.MANIFEST_NAME
        with manifest.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertNotIn("\\", rows[0]["file"])

    def test_master_is_hard_linked_not_duplicated(self):
        """250GB should not be copied again just to group it differently."""
        counts = export_catalog.run(self.conn, self.cfg)
        exported = self.cfg.export_root / "albania/2024/000001_dji_0001_ksamil.mp4"
        self.assertEqual(counts["linked"], 1)
        self.assertEqual(os.stat(exported).st_ino, os.stat(self.master).st_ino)

    def test_falls_back_to_copy_when_linking_is_unsupported(self):
        def refuse(*args, **kwargs):
            raise OSError("cross-device link")

        original = export_catalog.os.link
        export_catalog.os.link = refuse
        self.addCleanup(setattr, export_catalog.os, "link", original)

        counts = export_catalog.run(self.conn, self.cfg)
        exported = self.cfg.export_root / "albania/2024/000001_dji_0001_ksamil.mp4"
        self.assertEqual(counts["copied"], 1)
        self.assertEqual(exported.read_bytes(), b"video-bytes")

    def test_export_never_touches_the_render(self):
        export_catalog.run(self.conn, self.cfg)
        self.assertTrue(self.master.is_file())
        self.assertEqual(self.master.read_bytes(), b"video-bytes")

    def test_rerun_is_idempotent(self):
        export_catalog.run(self.conn, self.cfg)
        counts = export_catalog.run(self.conn, self.cfg)
        self.assertEqual(counts["exported"], 1)
        folder = self.cfg.export_root / "albania/2024"
        self.assertEqual(len(list(folder.glob("*.mp4"))), 1)

    def test_tier_filter_writes_its_own_folder(self):
        counts = export_catalog.run(self.conn, self.cfg, tier="non-exclusive")
        self.assertEqual(counts["exported"], 1)
        self.assertTrue(
            (self.cfg.export_root / "non-exclusive/albania/2024").is_dir()
        )

    def test_tier_filter_excludes_other_tiers(self):
        counts = export_catalog.run(self.conn, self.cfg, tier="exclusive")
        self.assertEqual(counts["exported"], 0)

    def test_untagged_shots_are_not_exported(self):
        with db.transaction(self.conn):
            db.update_shot(self.conn, self.shot_id, tagged_at=None)
        self.assertEqual(export_catalog.run(self.conn, self.cfg)["exported"], 0)

    def test_rejected_shots_are_not_exported(self):
        db.reject(self.conn, [self.shot_id], "bottom percentile")
        self.assertEqual(export_catalog.run(self.conn, self.cfg)["exported"], 0)

    def test_missing_master_is_reported_not_fatal(self):
        self.master.unlink()
        counts = export_catalog.run(self.conn, self.cfg)
        self.assertEqual(counts["missing"], 1)
        self.assertEqual(counts["exported"], 0)

    def test_dry_run_writes_nothing(self):
        counts = export_catalog.run(self.conn, self.cfg, dry_run=True)
        self.assertEqual(counts["exported"], 1)
        self.assertFalse(self.cfg.export_root.exists())


if __name__ == "__main__":
    unittest.main()
