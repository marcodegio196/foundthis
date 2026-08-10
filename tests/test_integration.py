"""End-to-end tests against a real ffmpeg.

The rest of the suite substitutes the toolchain, which cannot catch a filter
graph that is syntactically fine and produces the wrong pixels. These render
actual video and probe the result.

Skipped automatically when ffmpeg/ffprobe are absent, so the dependency-free
suite still runs everywhere. They do run on a machine set up to use the
pipeline, which is where it matters.
"""

import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pipeline import db, media, motion, render
from pipeline.config import config as base_config
from pipeline.stages import export_catalog, ingest, scene_detect, select_render

HAS_FFMPEG = media.toolchain_available()

# Small and short: these run on every `unittest discover`, so they must stay
# fast. The filter maths does not care about source resolution.
DURATION = 2


def make_clip(path: Path, width: int, height: int, *, created: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"testsrc2=size={width}x{height}:rate=30:duration={DURATION}",
    ]
    if created:
        command += ["-metadata", f"creation_time={created}"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return path


def probe_stream(path: Path) -> dict[str, str]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,sample_aspect_ratio,nb_frames",
            "-show_entries", "format=duration",
            "-of", "default=nw=1", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    return dict(
        line.split("=", 1) for line in out.strip().splitlines() if "=" in line
    )


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not on PATH")
class IntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)
        self.conn = db.init_db(self.root / "test.db")
        self.addCleanup(self.conn.close)
        self.cfg = replace(
            base_config,
            archive_root=self.root / "archive",
            render_root=self.root / "renders",
            export_root=self.root / "exports",
            overlay_font=None,
        )


class TestRealProbe(IntegrationTestCase):
    def test_parses_real_ffprobe_output(self):
        clip = make_clip(
            self.cfg.archive_root / "albania/DJI_0001.MP4", 640, 360,
            created="2024-06-01T08:30:00Z",
        )
        probed = media.probe(clip)
        self.assertAlmostEqual(probed["duration"], DURATION, places=1)
        self.assertEqual((probed["width"], probed["height"]), (640, 360))
        self.assertEqual(probed["aspect"], "16:9")
        self.assertEqual(probed["fps"], 30.0)
        self.assertEqual(probed["codec"], "h264")
        self.assertTrue(probed["captured_at"].startswith("2024-06-01"))

    def test_vertical_source_reads_as_vertical(self):
        clip = make_clip(self.cfg.archive_root / "peru/v.MP4", 360, 640)
        self.assertEqual(media.probe(clip)["aspect"], "9:16")

    def test_extracts_the_requested_frames(self):
        clip = make_clip(self.cfg.archive_root / "albania/a.MP4", 320, 180)
        stamps = media.sample_timestamps(0.0, float(DURATION), 1.0, max_frames=3)
        frames = media.extract_frames(clip, stamps, self.root / "frames", width=160)
        self.assertEqual(len(frames), len(stamps))
        for frame in frames:
            self.assertGreater(frame.stat().st_size, 0)

    def test_probe_failure_raises_media_error(self):
        broken = self.root / "broken.mp4"
        broken.write_bytes(b"not a video")
        with self.assertRaises(media.MediaError):
            media.probe(broken)


class TestRealRender(IntegrationTestCase):
    def render_social(self, source_width, source_height, **kwargs):
        clip = make_clip(
            self.cfg.archive_root / "albania/DJI_0001.MP4", source_width, source_height
        )
        out = self.root / "out.mp4"
        params = dict(
            in_point=0.0, out_point=1.0,
            width=source_width, height=source_height,
            lines=["Found this.", "2024 · Ksamil"],
        )
        params.update(kwargs)
        media.run(render.build_social_command(clip, out, **params))
        return out

    def test_landscape_source_renders_exactly_9x16(self):
        """A 720-tall source wants a 405-wide crop; odd widths are illegal, and
        deriving the width with -2 afterwards yielded 1078x1920 — not 9:16."""
        info = probe_stream(self.render_social(1280, 720))
        self.assertEqual((info["width"], info["height"]), ("1080", "1920"))

    def test_native_vertical_source_renders_exactly_9x16(self):
        info = probe_stream(self.render_social(360, 640))
        self.assertEqual((info["width"], info["height"]), ("1080", "1920"))

    def test_square_source_renders_exactly_9x16(self):
        info = probe_stream(self.render_social(480, 480))
        self.assertEqual((info["width"], info["height"]), ("1080", "1920"))

    def test_taller_than_9x16_source_renders_exactly_9x16(self):
        info = probe_stream(self.render_social(360, 900))
        self.assertEqual((info["width"], info["height"]), ("1080", "1920"))

    def test_pixels_are_square(self):
        """A non-square SAR inherited from the source would skew playback."""
        self.assertIn(
            probe_stream(self.render_social(1280, 720))["sample_aspect_ratio"],
            ("1:1", "N/A"),
        )

    def test_alternate_output_height_stays_9x16(self):
        info = probe_stream(self.render_social(1280, 720, output_height=1280))
        self.assertEqual((info["width"], info["height"]), ("720", "1280"))

    def test_overlay_filter_graph_is_accepted_by_ffmpeg(self):
        """drawtext is the most fragile part of the chain; a bad escape is a
        hard ffmpeg failure, not a silently ugly frame."""
        out = self.render_social(640, 360, lines=["Found this.", "2024 · Vlorë: l'estate"])
        self.assertGreater(out.stat().st_size, 0)

    def test_trim_respects_the_shot_boundaries(self):
        clip = make_clip(self.cfg.archive_root / "albania/a.MP4", 320, 180)
        out = self.root / "trim.mp4"
        media.run(
            render.build_licensing_command(clip, out, in_point=0.5, out_point=1.5)
        )
        self.assertAlmostEqual(float(probe_stream(out)["duration"]), 1.0, delta=0.15)

    def test_licensing_master_keeps_source_resolution(self):
        clip = make_clip(self.cfg.archive_root / "albania/a.MP4", 640, 360)
        out = self.root / "master.mp4"
        media.run(render.build_licensing_command(clip, out, in_point=0.0, out_point=1.0))
        info = probe_stream(out)
        self.assertEqual((info["width"], info["height"]), ("640", "360"))


class TestFullRun(IntegrationTestCase):
    def test_archive_to_export(self):
        """Ingest, segment, render, and export a real file end to end."""
        make_clip(
            self.cfg.archive_root / "albania/DJI_0001.MP4", 640, 360,
            created="2024-06-01T08:30:00Z",
        )
        make_clip(self.cfg.archive_root / "peru/DJI_0002.MP4", 360, 640)

        self.assertEqual(ingest.run(self.conn, self.cfg)["added"], 2)
        self.assertEqual(scene_detect.run(self.conn, self.cfg)["sources"], 2)

        for row in self.conn.execute("SELECT id FROM shots").fetchall():
            with db.transaction(self.conn):
                db.update_shot(
                    self.conn, row["id"],
                    scored_at=db.now(), aesthetic_score=0.8, technical_score=0.8,
                    tagged_at=db.now(), subject_tags=["coastline"],
                    mood_tags=["solitude"], description="A coastline, slow push in.",
                    person_in_frame=0, license_tier="non-exclusive",
                    selection_state="approved", approved_at=db.now(),
                )

        self.assertEqual(select_render.run(self.conn, self.cfg)["rendered"], 2)

        social = sorted(self.cfg.social_render_dir.rglob("*.mp4"))
        self.assertEqual(len(social), 2)
        self.assertEqual(
            {p.parent.relative_to(self.cfg.social_render_dir).as_posix() for p in social},
            {"albania/2024", "peru/undated"},
        )
        for clip in social:
            info = probe_stream(clip)
            self.assertEqual((info["width"], info["height"]), ("1080", "1920"))

        counts = export_catalog.run(self.conn, self.cfg, tier="non-exclusive")
        self.assertEqual(counts["exported"], 2)
        root = self.cfg.export_root / "non-exclusive"
        self.assertTrue((root / export_catalog.MANIFEST_NAME).is_file())
        self.assertEqual(len(list(root.rglob("*.json"))), 2)

    def test_archive_is_never_modified(self):
        """The whole pipeline is read-only with respect to the raw footage."""
        clip = make_clip(self.cfg.archive_root / "albania/DJI_0001.MP4", 320, 180)
        before = (clip.read_bytes(), clip.stat().st_mtime)

        ingest.run(self.conn, self.cfg)
        scene_detect.run(self.conn, self.cfg)
        shot_id = self.conn.execute("SELECT id FROM shots").fetchone()["id"]
        with db.transaction(self.conn):
            db.update_shot(
                self.conn, shot_id,
                scored_at=db.now(), tagged_at=db.now(), description="x",
                selection_state="approved", approved_at=db.now(),
                license_tier="public",
            )
        select_render.run(self.conn, self.cfg)
        export_catalog.run(self.conn, self.cfg)

        self.assertEqual((clip.read_bytes(), clip.stat().st_mtime), before)
        self.assertEqual(len(list(self.cfg.archive_root.rglob("*"))), 2)  # dir + file


if __name__ == "__main__":
    unittest.main()


try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False


@unittest.skipUnless(HAS_CV2, "opencv/numpy not installed")
class TestScoringAgainstKnownMotion(unittest.TestCase):
    """Score real frames whose displacement we chose, so the numbers can be
    checked rather than assumed.

    The unit tests build `FrameFlow` values by hand; these run the actual
    optical-flow path over image files and assert the discrimination Stage 2
    exists to make.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        # Blurred noise: rich, non-repeating texture that optical flow can lock
        # onto. A synthetic test pattern correlates too poorly to measure with.
        cls.base = cv2.GaussianBlur(
            rng.integers(0, 255, (720, 1280), dtype=np.uint8), (5, 5), 0
        )
        cls.rng = rng

    def scores(self, offsets, *, noise=0, blur=0):
        from pipeline.stages.score import analyse_frames

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, (dx, dy) in enumerate(offsets):
                frame = np.roll(np.roll(self.base, -dy, axis=0), -dx, axis=1)
                frame = frame[100:460, 100:740].astype(np.int16)
                if noise:
                    frame = np.clip(frame + self.rng.normal(0, noise, frame.shape), 0, 255)
                frame = frame.astype(np.uint8)
                if blur:
                    frame = cv2.GaussianBlur(frame, (blur | 1, blur | 1), 0)
                path = Path(tmp) / f"{index:03d}.png"
                cv2.imwrite(str(path), frame)
                paths.append(path)
            flows, sharpness, clipped, width = analyse_frames(paths)

        from pipeline import scoring

        return scoring.Scores(
            motion=scoring.motion_score(flows, width),
            stability=scoring.stability_score(flows, width),
            technical=scoring.technical_score(sharpness, clipped),
        )

    @staticmethod
    def pan(count=12, step=12):
        return [(i * step, 0) for i in range(count)]

    @staticmethod
    def shake(count=12):
        return [(20 * ((i % 2) * 2 - 1), 15 * ((i % 3) - 1)) for i in range(count)]

    @staticmethod
    def hover(count=12):
        return [(int(6 * np.sin(i)), int(6 * np.cos(i * 1.3))) for i in range(count)]

    def test_smooth_pan_is_maximally_stable(self):
        self.assertAlmostEqual(self.scores(self.pan()).stability, 1.0, places=2)

    def test_diagonal_pan_is_maximally_stable(self):
        offsets = [(i * 8, i * 5) for i in range(12)]
        self.assertAlmostEqual(self.scores(offsets).stability, 1.0, places=2)

    def test_locked_off_shot_is_stable(self):
        """Identical frames still produce ~0.0002px of flow, so the static
        branch has to trigger on a realistic threshold, not on exact zero."""
        self.assertAlmostEqual(self.scores([(0, 0)] * 12).stability, 1.0, places=2)

    def test_sensor_noise_does_not_read_as_movement(self):
        self.assertAlmostEqual(
            self.scores([(0, 0)] * 12, noise=3).stability, 1.0, places=2
        )

    def test_pan_beats_shake(self):
        """The discrimination the held-shot-with-overlay format depends on."""
        self.assertGreater(
            self.scores(self.pan()).stability, self.scores(self.shake()).stability
        )

    def test_locked_off_beats_shake(self):
        self.assertGreater(
            self.scores([(0, 0)] * 12).stability, self.scores(self.shake()).stability
        )

    def test_drifting_hover_scores_worst(self):
        """Incoherent wander is the least usable of the moving cases."""
        self.assertLess(
            self.scores(self.hover()).stability, self.scores(self.shake()).stability
        )

    def test_faster_pan_reads_as_more_motion(self):
        self.assertGreater(
            self.scores(self.pan(step=24)).motion, self.scores(self.pan(step=4)).motion
        )

    def test_static_shot_has_no_motion(self):
        self.assertAlmostEqual(self.scores([(0, 0)] * 12).motion, 0.0, places=2)

    def test_blur_is_caught_by_technical_not_stability(self):
        """A blurred pan is perfectly smooth — only the technical score saves it."""
        blurred = self.scores(self.pan(), blur=15)
        sharp = self.scores(self.pan())
        self.assertGreater(blurred.stability, 0.9)
        self.assertLess(blurred.technical, sharp.technical * 0.6)

    def test_combined_ranking_matches_the_format(self):
        ranked = {
            name: scoring_result
            for name, scoring_result in (
                ("pan", self.scores(self.pan())),
                ("locked", self.scores([(0, 0)] * 12)),
                ("shake", self.scores(self.shake())),
                ("hover", self.scores(self.hover())),
            )
        }
        from pipeline import scoring

        combined = {k: scoring.combine(v) for k, v in ranked.items()}
        self.assertGreater(combined["pan"], combined["shake"])
        self.assertGreater(combined["locked"], combined["shake"])
        self.assertGreater(combined["shake"], combined["hover"])


try:
    import scenedetect  # noqa: F401
    HAS_SCENEDETECT = True
except ImportError:  # pragma: no cover
    HAS_SCENEDETECT = False


@unittest.skipUnless(HAS_FFMPEG and HAS_SCENEDETECT, "ffmpeg or scenedetect missing")
class TestRealSceneDetection(IntegrationTestCase):
    """Exercise the actual detector.

    Without scenedetect installed, `detect_segments` falls back to one
    whole-file shot — which means the detector path and the merge logic that
    consumes its output are never executed by the rest of the suite.
    """

    def concat(self, name: str, clips: list[Path]) -> Path:
        listing = self.root / f"{name}.txt"
        listing.write_text("".join(f"file '{c}'\n" for c in clips))
        out = self.root / f"{name}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", str(out),
            ],
            check=True, capture_output=True,
        )
        return out

    def hued_clip(self, name: str, hue: int, duration: int) -> Path:
        out = self.root / f"{name}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi",
                "-i", f"testsrc2=size=640x360:rate=30:duration={duration}",
                "-vf", f"hue=h={hue}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
            ],
            check=True, capture_output=True,
        )
        return out

    def panning_clip(self, name: str, duration: int) -> Path:
        still = self.root / f"{name}.png"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=1:duration=1",
                "-frames:v", "1", str(still),
            ],
            check=True, capture_output=True,
        )
        out = self.root / f"{name}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-loop", "1", "-i", str(still), "-t", str(duration), "-r", "30",
                "-vf", "crop=640:360:x='min(t*100,1200)':y=360",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
            ],
            check=True, capture_output=True,
        )
        return out

    def test_finds_the_cuts(self):
        clips = [self.hued_clip(f"seg{i}", i * 120, 3) for i in range(1, 4)]
        segments = scene_detect.detect_segments(self.concat("cuts", clips), 9.0, self.cfg)
        self.assertEqual(len(segments), 3)
        for (start, end), expected in zip(segments, [(0, 3), (3, 6), (6, 9)]):
            self.assertAlmostEqual(start, expected[0], delta=0.2)
            self.assertAlmostEqual(end, expected[1], delta=0.2)

    def test_continuous_move_stays_one_shot(self):
        """The common case for drone footage — a slow reveal must not be chopped."""
        clip = self.panning_clip("continuous", 8)
        segments = scene_detect.detect_segments(clip, 8.0, self.cfg)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], (0.0, 8.0))

    def test_short_tail_is_merged_into_its_neighbour(self):
        """A one-second sliver is a detector artifact, not a shot."""
        clips = [
            self.hued_clip("a", 0, 3),
            self.hued_clip("b", 120, 3),
            self.hued_clip("c", 240, 1),
        ]
        segments = scene_detect.detect_segments(self.concat("sliver", clips), 7.0, self.cfg)
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[-1][1], 7.0, delta=0.2)
        for start, end in segments:
            self.assertGreaterEqual(end - start, self.cfg.min_shot_seconds)

    def test_segments_tile_the_whole_timeline(self):
        clips = [self.hued_clip(f"t{i}", i * 120, 3) for i in range(1, 4)]
        segments = scene_detect.detect_segments(self.concat("tile", clips), 9.0, self.cfg)
        self.assertEqual(segments[0][0], 0.0)
        self.assertAlmostEqual(segments[-1][1], 9.0, delta=0.2)
        for (_, end), (start, _) in zip(segments, segments[1:]):
            self.assertEqual(end, start)


@unittest.skipUnless(HAS_FFMPEG and HAS_CV2, "ffmpeg or opencv missing")
class TestMotionAgainstKnownCameraMoves(IntegrationTestCase):
    """Measure real footage whose camera speed we chose.

    The unit tests assert the logic; these assert the measurement — that a
    given camera behaviour produces the numbers the thresholds assume.
    """

    @classmethod
    def setUpClass(cls):
        cls._plate_dir = tempfile.TemporaryDirectory()
        plate = Path(cls._plate_dir.name) / "plate.png"
        # Blurred noise: rich, non-repeating detail at the scale correlation
        # locks onto. A synthetic test pattern correlates too poorly to measure.
        rng = np.random.default_rng(11)
        image = cv2.GaussianBlur(
            rng.integers(0, 255, (2160, 3840), dtype=np.uint8), (9, 9), 0
        )
        cv2.imwrite(str(plate), cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX))
        cls.plate = plate

    @classmethod
    def tearDownClass(cls):
        cls._plate_dir.cleanup()

    def fly(self, name: str, crop: str, seconds: int = 4, extra: str = "") -> Path:
        out = self.root / f"{name}.mp4"
        chain = f"crop=640:360:{crop}" + (f",{extra}" if extra else "")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-loop", "1", "-i", str(self.plate), "-t", str(seconds), "-r", "30",
                "-vf", chain, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
            ],
            check=True, capture_output=True,
        )
        return out

    def profile(self, clip: Path, seconds: float = 4.0):
        from pipeline.stages.scene_detect import profile_motion

        samples, width = profile_motion(clip, 0.0, seconds, self.cfg)
        return motion.summarise(samples), width

    def verdict(self, clip: Path, seconds: float = 4.0) -> str:
        window, width = self.profile(clip, seconds)
        return motion.classify(window, width, self.cfg.motion_sample_fps)

    # -- the camera is still -------------------------------------------------

    def test_locked_off_shot_is_held(self):
        self.assertEqual(self.verdict(self.fly("held", "x=1600:y=900")), motion.HELD)

    def test_subject_crossing_frame_does_not_move_the_camera(self):
        clip = self.fly(
            "subject", "x=1600:y=900",
            extra="drawbox=x='50+t*140':y=120:w=90:h=90:color=black@1:t=fill",
        )
        self.assertEqual(self.verdict(clip), motion.HELD)

    def test_moving_water_does_not_read_as_repositioning(self):
        """The false positive that would discard a good held shot: mean optical
        flow calls this chaotic, phase correlation reads the camera as parked."""
        clip = self.fly("water", "x=1600:y=900", extra="noise=alls=45:allf=t+u")
        self.assertEqual(self.verdict(clip), motion.HELD)

    # -- the camera moves deliberately ---------------------------------------

    def test_slow_pan_is_usable(self):
        self.assertEqual(self.verdict(self.fly("slow", "x='1600+t*40':y=900")), motion.MOVE)

    def test_medium_pan_is_usable(self):
        self.assertEqual(self.verdict(self.fly("medium", "x='1600+t*150':y=900")), motion.MOVE)

    def test_measured_speed_tracks_actual_speed(self):
        """Optical flow under-reported fast motion so badly that a 1200 px/s
        reposition measured slower than a 40 px/s pan."""
        slow, _ = self.profile(self.fly("s1", "x='1600+t*40':y=900"))
        medium, _ = self.profile(self.fly("s2", "x='1600+t*150':y=900"))
        fast, _ = self.profile(self.fly("s3", "x='1600+t*400':y=900"))
        self.assertLess(slow.shift, medium.shift)
        self.assertLess(medium.shift, fast.shift)

    def test_measured_speed_is_proportional(self):
        """150 px/s should measure about four times 40 px/s."""
        slow, _ = self.profile(self.fly("p1", "x='1600+t*40':y=900"))
        medium, _ = self.profile(self.fly("p2", "x='1600+t*150':y=900"))
        self.assertAlmostEqual(medium.shift / slow.shift, 150 / 40, delta=0.6)

    # -- the camera is searching ---------------------------------------------

    def test_reposition_is_rejected(self):
        clip = self.fly(
            "repo", "x='1600+t*1200':y=900",
        )
        window, width = self.profile(clip)
        self.assertGreater(window.jitter, motion.JITTER)
        self.assertEqual(
            motion.classify(window, width, self.cfg.motion_sample_fps),
            motion.REPOSITION,
        )

    def test_oscillating_search_is_rejected(self):
        clip = self.fly("osc", "x='1600+300*sin(t*6)':y='900+200*cos(t*7)'")
        self.assertEqual(self.verdict(clip), motion.REPOSITION)

    def test_deliberate_moves_have_no_jitter(self):
        for name, speed in (("j1", 40), ("j2", 150), ("j3", 400)):
            window, _ = self.profile(self.fly(name, f"x='1600+t*{speed}':y=900"))
            self.assertLess(window.jitter, motion.JITTER, f"{speed} px/s")

    # -- putting it together -------------------------------------------------

    def test_one_file_two_usable_shots(self):
        """A file holding a good move, a reposition, then another good move
        becomes two usable shots and one flagged one — not a single shot whose
        score is the average of all three."""
        clip = self.fly(
            "multi",
            "x='if(lt(t,4), 200+t*40, if(lt(t,6), 360+(t-4)*900+120*sin((t-4)*25),"
            " 2160+(t-6)*40))'"
            ":y='if(lt(t,4), 900, if(lt(t,6), 900+300*sin((t-4)*18), 900))'",
            seconds=10,
        )
        segments = scene_detect.split_by_motion(clip, [(0.0, 10.0)], self.cfg)
        classes = [motion_class for _, _, motion_class in segments]
        self.assertEqual(classes.count(motion.REPOSITION), 1)
        self.assertEqual(len([c for c in classes if c in motion.USABLE]), 2)

        # The reposition lands where it actually is, and the timeline is whole.
        reposition = next(s for s in segments if s[2] == motion.REPOSITION)
        self.assertAlmostEqual(reposition[0], 4.0, delta=0.6)
        self.assertAlmostEqual(reposition[1], 6.0, delta=0.6)
        self.assertEqual(segments[0][0], 0.0)
        self.assertEqual(segments[-1][1], 10.0)
        for (_, end, _), (start, _, _) in zip(segments, segments[1:]):
            self.assertEqual(end, start)

    def test_repositioning_is_flagged_not_deleted(self):
        clip = self.cfg.archive_root / "albania/DJI_MULTI.MP4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        self.fly("tmp", "x='if(lt(t,4), 200+t*40, if(lt(t,6),"
                 " 360+(t-4)*900+120*sin((t-4)*25), 2160+(t-6)*40))'"
                 ":y='if(lt(t,4), 900, if(lt(t,6), 900+300*sin((t-4)*18), 900))'",
                 seconds=10).rename(clip)

        ingest.run(self.conn, self.cfg)
        counts = scene_detect.run(self.conn, self.cfg)
        self.assertEqual(counts["reposition"], 1)
        self.assertEqual(counts["usable"], 2)

        rows = self.conn.execute(
            "SELECT * FROM shots ORDER BY in_point"
        ).fetchall()
        rejected = [r for r in rows if r["rejected"]]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["motion_class"], motion.REPOSITION)
        self.assertIn("reposition", rejected[0]["reject_reason"])
        # Flagged, never deleted: the footage stays addressable.
        self.assertEqual(len(rows), 3)
