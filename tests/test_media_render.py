import unittest

from pipeline import media, render


class TestProbeParsing(unittest.TestCase):
    def probe(self, **overrides):
        raw = {
            "format": {
                "duration": "42.5",
                "bit_rate": "98000000",
                "tags": {
                    "creation_time": "2024-06-01T08:30:00.000000Z",
                    "com.apple.quicktime.location.ISO6709": "+41.3275+019.8187+123.456/",
                    "com.apple.quicktime.make": "DJI",
                    "com.apple.quicktime.model": "FC3582",
                },
            },
            "streams": [
                {"codec_type": "audio", "codec_name": "aac"},
                {
                    "codec_type": "video",
                    "codec_name": "hevc",
                    "width": 3840,
                    "height": 2160,
                    "r_frame_rate": "30000/1001",
                },
            ],
        }
        raw["streams"][1].update(overrides)
        return media.parse_probe(raw)

    def test_extracts_core_fields(self):
        probed = self.probe()
        self.assertEqual(probed["duration"], 42.5)
        self.assertEqual(probed["codec"], "hevc")
        self.assertEqual(probed["width"], 3840)
        self.assertEqual(probed["aspect"], "16:9")
        self.assertEqual(probed["fps"], 29.97)

    def test_extracts_gps_and_drone_model(self):
        probed = self.probe()
        self.assertAlmostEqual(probed["gps_lat"], 41.3275)
        self.assertAlmostEqual(probed["gps_lon"], 19.8187)
        self.assertEqual(probed["drone_model"], "DJI FC3582")

    def test_rotation_flips_dimensions(self):
        """A 9:16 master stored landscape with a rotation flag must read as 9:16."""
        probed = self.probe(side_data_list=[{"rotation": -90}])
        self.assertEqual((probed["width"], probed["height"]), (2160, 3840))
        self.assertEqual(probed["aspect"], "9:16")

    def test_rotate_tag_also_honoured(self):
        probed = self.probe(tags={"rotate": "90"})
        self.assertEqual(probed["aspect"], "9:16")

    def test_missing_video_stream_is_tolerated(self):
        probed = media.parse_probe({"format": {}, "streams": []})
        self.assertIsNone(probed["width"])
        self.assertIsNone(probed["aspect"])


class TestHelpers(unittest.TestCase):
    def test_parse_fps_handles_zero_rate(self):
        self.assertIsNone(media.parse_fps("0/0"))

    def test_aspect_falls_back_to_reduced_ratio(self):
        self.assertEqual(media.aspect_label(1000, 300), "10:3")

    def test_iso6709_negative_coordinates(self):
        self.assertEqual(media.parse_iso6709("-33.8688+151.2093/"), (-33.8688, 151.2093))

    def test_iso6709_rejects_garbage(self):
        self.assertEqual(media.parse_iso6709("somewhere nice"), (None, None))


class TestSampleTimestamps(unittest.TestCase):
    def test_avoids_the_cut_points(self):
        stamps = media.sample_timestamps(10.0, 20.0, fps=1.0)
        self.assertGreater(stamps[0], 10.0)
        self.assertLess(stamps[-1], 20.0)

    def test_respects_max_frames(self):
        self.assertEqual(len(media.sample_timestamps(0.0, 600.0, 1.0, max_frames=4)), 4)

    def test_always_samples_a_short_shot(self):
        self.assertEqual(len(media.sample_timestamps(0.0, 0.4, 1.0)), 1)

    def test_zero_length_shot_yields_nothing(self):
        self.assertEqual(media.sample_timestamps(5.0, 5.0, 1.0), [])


class TestDrawtextEscaping(unittest.TestCase):
    def test_escapes_colon_and_quote(self):
        """A location with a colon otherwise breaks the whole filter chain."""
        escaped = render.escape_drawtext("Vlorë: l'old town")
        self.assertNotIn(":", escaped.replace(r"\:", ""))
        self.assertIn(r"\:", escaped)
        self.assertIn(r"\'", escaped)


class TestOverlayLines(unittest.TestCase):
    def test_year_and_location(self):
        self.assertEqual(
            render.overlay_lines("2024", "albania", "Ksamil", "Found this."),
            ["Found this.", "2024 · Ksamil · albania"],
        )

    def test_headline_only_when_nothing_known(self):
        self.assertEqual(
            render.overlay_lines(None, None, None, "Found this."), ["Found this."]
        )

    def test_skips_missing_parts(self):
        self.assertEqual(
            render.overlay_lines("2024", "peru", None, "Found this."),
            ["Found this.", "2024 · peru"],
        )


class TestCrop(unittest.TestCase):
    def test_landscape_cropped_to_vertical(self):
        self.assertEqual(render.crop_to_vertical(3840, 2160), "crop=1214:2160")

    def test_native_vertical_not_cropped(self):
        self.assertIsNone(render.crop_to_vertical(2160, 3840))

    def test_taller_than_vertical_is_trimmed_top_and_bottom(self):
        """Cropping width on an already-narrow source would overcrop it."""
        self.assertEqual(render.crop_to_vertical(1080, 2400), "crop=1080:1920")

    def test_all_crop_dimensions_are_even(self):
        """Odd dimensions break yuv420p encoding."""
        for width, height in (
            (1920, 1080), (3840, 2160), (1280, 719), (481, 481), (1080, 2401)
        ):
            crop = render.crop_to_vertical(width, height)
            dims = crop.split("=")[1].split(":")
            self.assertEqual([int(d) % 2 for d in dims], [0, 0], crop)

    def test_crop_never_exceeds_the_source(self):
        for width, height in ((1920, 1080), (100, 5000), (5000, 100)):
            crop = render.crop_to_vertical(width, height)
            if crop:
                crop_w, crop_h = (int(d) for d in crop.split("=")[1].split(":"))
                self.assertLessEqual(crop_w, width)
                self.assertLessEqual(crop_h, height)


class TestVerticalOutputSize(unittest.TestCase):
    def test_standard_size(self):
        self.assertEqual(render.vertical_output_size(1920), (1080, 1920))

    def test_alternate_sizes_stay_even_and_9x16(self):
        for height in (1280, 1600, 1920, 2160):
            width, out_height = render.vertical_output_size(height)
            self.assertEqual(width % 2, 0)
            self.assertEqual(out_height % 2, 0)
            self.assertAlmostEqual(width / out_height, 9 / 16, places=2)

    def test_scale_filter_pins_exact_dimensions(self):
        """Deriving width with -2 produced 1078x1920 from a 720-tall source."""
        command = render.build_social_command(
            "/a.mp4", "/b.mp4", in_point=0, out_point=1,
            width=1280, height=720, lines=["x"],
        )
        self.assertIn("scale=1080:1920,setsar=1", command[command.index("-vf") + 1])


class TestRenderCommands(unittest.TestCase):
    def social(self, **kwargs):
        params = dict(
            in_point=2.0, out_point=9.5, width=3840, height=2160,
            lines=["Found this.", "2024 · Ksamil"],
        )
        params.update(kwargs)
        return render.build_social_command("/a/in.mp4", "/b/out.mp4", **params)

    def test_seeks_before_input(self):
        """-ss after -i decodes from zero; on 4K that is the whole cost."""
        command = self.social()
        self.assertLess(command.index("-ss"), command.index("-i"))

    def test_trims_to_the_shot(self):
        command = self.social()
        self.assertEqual(command[command.index("-ss") + 1], "2.000")
        self.assertEqual(command[command.index("-to") + 1], "9.500")

    def test_burns_in_both_lines(self):
        filters = self.social()[self.social().index("-vf") + 1]
        self.assertEqual(filters.count("drawtext="), 2)
        self.assertIn("Found this.", filters)

    def test_crops_landscape_source(self):
        self.assertIn("crop=", self.social()[self.social().index("-vf") + 1])

    def test_does_not_crop_vertical_source(self):
        command = self.social(width=2160, height=3840)
        self.assertNotIn("crop=", command[command.index("-vf") + 1])

    def test_font_file_included_when_configured(self):
        command = self.social(font_file="/fonts/Inter.ttf")
        self.assertIn("fontfile=", command[command.index("-vf") + 1])

    def test_windows_font_path_is_escaped(self):
        """A raw C:\\ path breaks drawtext — the drive colon reads as a separator."""
        command = self.social(font_file=r"C:\Windows\Fonts\segoeui.ttf")
        filters = command[command.index("-vf") + 1]
        self.assertIn(r"fontfile='C\:\\Windows\\Fonts\\segoeui.ttf'", filters)

    def test_licensing_master_has_no_overlay(self):
        command = render.build_licensing_command(
            "/a/in.mp4", "/b/out.mov", in_point=2.0, out_point=9.5
        )
        self.assertNotIn("-vf", command)
        self.assertNotIn("drawtext", " ".join(command))

    def test_licensing_keeps_full_resolution_and_audio(self):
        command = render.build_licensing_command(
            "/a/in.mp4", "/b/out.mov", in_point=0.0, out_point=5.0
        )
        self.assertNotIn("scale=", " ".join(command))
        self.assertIn("copy", command)

    def test_licensing_quality_beats_social(self):
        self.assertLess(render.LICENSING_CRF, render.SOCIAL_CRF)

    def test_prores_profile(self):
        command = render.build_licensing_command(
            "/a/in.mp4", "/b/out.mov", in_point=0.0, out_point=5.0, codec="prores_ks"
        )
        self.assertIn("prores_ks", command)
        self.assertIn("-profile:v", command)


if __name__ == "__main__":
    unittest.main()
