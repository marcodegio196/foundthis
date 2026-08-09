import unittest

from pipeline import scoring
from pipeline.scoring import FrameFlow, Scores


def pan(count=8, magnitude=4.0):
    """A smooth horizontal move: every frame drifts the same way."""
    return [FrameFlow(dx=magnitude, dy=0.0, magnitude=magnitude) for _ in range(count)]


def shake(count=8, magnitude=4.0):
    """Jitter: same magnitude, direction reverses each frame."""
    return [
        FrameFlow(dx=magnitude * (1 if i % 2 else -1), dy=0.0, magnitude=magnitude)
        for i in range(count)
    ]


class TestMotion(unittest.TestCase):
    def test_static_shot_scores_zero(self):
        flows = [FrameFlow(0.0, 0.0, 0.0) for _ in range(4)]
        self.assertEqual(scoring.motion_score(flows, 1920), 0.0)

    def test_faster_move_scores_higher(self):
        slow = scoring.motion_score(pan(magnitude=2.0), 1920)
        fast = scoring.motion_score(pan(magnitude=40.0), 1920)
        self.assertLess(slow, fast)

    def test_normalised_by_frame_width(self):
        """The same move at 4K and 1080p must score the same."""
        hd = scoring.motion_score(pan(magnitude=4.0), 1920)
        uhd = scoring.motion_score(pan(magnitude=8.0), 3840)
        self.assertAlmostEqual(hd, uhd, places=6)

    def test_no_flows_is_none(self):
        self.assertIsNone(scoring.motion_score([], 1920))


class TestStability(unittest.TestCase):
    def test_smooth_pan_beats_shake(self):
        """The key discrimination Stage 2 exists to make."""
        self.assertGreater(scoring.stability_score(pan()), scoring.stability_score(shake()))

    def test_smooth_pan_is_near_perfect(self):
        self.assertGreater(scoring.stability_score(pan()), 0.95)

    def test_shake_is_heavily_penalised(self):
        self.assertLess(scoring.stability_score(shake()), 0.4)

    def test_locked_off_shot_is_stable(self):
        flows = [FrameFlow(0.0, 0.0, 0.0) for _ in range(4)]
        self.assertEqual(scoring.stability_score(flows), 1.0)

    def test_speed_wobble_scores_below_steady_move(self):
        steady = pan(magnitude=4.0)
        wobbly = [
            FrameFlow(dx=m, dy=0.0, magnitude=m) for m in (1.0, 7.0, 1.0, 7.0, 1.0, 7.0)
        ]
        self.assertGreater(scoring.stability_score(steady), scoring.stability_score(wobbly))

    def test_bounded_zero_to_one(self):
        for flows in (pan(), shake(), pan(magnitude=100.0)):
            self.assertGreaterEqual(scoring.stability_score(flows), 0.0)
            self.assertLessEqual(scoring.stability_score(flows), 1.0)


class TestTechnical(unittest.TestCase):
    def test_sharper_scores_higher(self):
        self.assertLess(
            scoring.technical_score([50.0], [0.0]), scoring.technical_score([600.0], [0.0])
        )

    def test_clipping_penalised(self):
        clean = scoring.technical_score([500.0], [0.0])
        blown = scoring.technical_score([500.0], [0.25])
        self.assertLess(blown, clean)

    def test_small_clipping_tolerated(self):
        """A sunset clips a little; that must not read as a defect."""
        self.assertAlmostEqual(
            scoring.technical_score([500.0], [0.0]),
            scoring.technical_score([500.0], [0.02]),
            places=6,
        )

    def test_no_samples_is_none(self):
        self.assertIsNone(scoring.technical_score([], []))


class TestCombine(unittest.TestCase):
    def test_renormalises_around_missing_components(self):
        """A run without the CLIP model must still produce comparable numbers."""
        full = Scores(motion=0.5, stability=0.5, technical=0.5, aesthetic=0.5)
        partial = Scores(motion=0.5, stability=0.5, technical=0.5)
        self.assertAlmostEqual(scoring.combine(full), 0.5)
        self.assertAlmostEqual(scoring.combine(partial), 0.5)

    def test_stability_outweighs_motion(self):
        stable = Scores(stability=1.0, motion=0.0)
        moving = Scores(stability=0.0, motion=1.0)
        self.assertGreater(scoring.combine(stable), scoring.combine(moving))

    def test_all_missing_is_none(self):
        self.assertIsNone(scoring.combine(Scores()))


class TestFloorsAndPercentile(unittest.TestCase):
    def test_blurry_shot_fails_floor(self):
        reason = scoring.fails_hard_floor(Scores(technical=0.05, aesthetic=0.99))
        self.assertIsNotNone(reason)
        self.assertIn("technical", reason)

    def test_good_shot_passes_floor(self):
        self.assertIsNone(scoring.fails_hard_floor(Scores(technical=0.9)))

    def test_missing_component_cannot_fail_floor(self):
        self.assertIsNone(scoring.fails_hard_floor(Scores()))

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(scoring.percentile_threshold([0.0, 1.0], 0.5), 0.5)

    def test_percentile_cuts_expected_share(self):
        values = [i / 100 for i in range(100)]
        threshold = scoring.percentile_threshold(values, 0.35)
        cut = [v for v in values if v < threshold]
        self.assertEqual(len(cut), 35)

    def test_percentile_single_value(self):
        self.assertEqual(scoring.percentile_threshold([0.4], 0.35), 0.4)

    def test_percentile_empty(self):
        self.assertIsNone(scoring.percentile_threshold([], 0.35))

    def test_percentile_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            scoring.percentile_threshold([0.1, 0.2], 35)


if __name__ == "__main__":
    unittest.main()
