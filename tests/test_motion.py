"""Motion classification.

The numbers in these tests come from measuring clips with known camera speeds
(see tests/test_integration.py for the measurement itself). What is asserted
here is the logic built on top: how a window is labelled, and how runs are
merged into shots.
"""

import unittest

from pipeline import motion
from pipeline.motion import Run, Sample, Window

WIDTH = 512
FPS = 4.0


def window(shift, drift=None, response=1.0):
    """A window with a given per-sample shift; drift defaults to no jitter."""
    return Window(shift=shift, drift=shift if drift is None else drift, response=response)


def samples(specs, step=0.25):
    """Build samples from (dx, dy) pairs, one per `step` seconds."""
    return [
        Sample(start=i * step, end=(i + 1) * step, dx=dx, dy=dy, response=1.0)
        for i, (dx, dy) in enumerate(specs)
    ]


class TestJitter(unittest.TestCase):
    def test_straight_move_has_no_jitter(self):
        self.assertAlmostEqual(motion.summarise(samples([(8, 0)] * 6)).jitter, 0.0)

    def test_reversing_motion_is_all_jitter(self):
        """Back-and-forth cancels in the sum but not in the distance."""
        self.assertAlmostEqual(motion.summarise(samples([(8, 0), (-8, 0)] * 3)).jitter, 1.0)

    def test_partial_reversal_is_partial_jitter(self):
        jitter = motion.summarise(samples([(8, 0), (8, 0), (8, 0), (-8, 0)])).jitter
        self.assertGreater(jitter, 0.2)
        self.assertLess(jitter, 0.8)

    def test_still_camera_has_no_jitter(self):
        self.assertAlmostEqual(motion.summarise(samples([(0, 0)] * 4)).jitter, 0.0)

    def test_diagonal_move_is_coherent(self):
        self.assertAlmostEqual(motion.summarise(samples([(6, 6)] * 5)).jitter, 0.0)

    def test_empty_window(self):
        self.assertEqual(motion.summarise([]).jitter, 0.0)


class TestClassify(unittest.TestCase):
    def classify(self, win, **kwargs):
        return motion.classify(win, WIDTH, FPS, **kwargs)

    def test_parked_camera_is_held(self):
        self.assertEqual(self.classify(window(0.0)), motion.HELD)

    def test_sub_pixel_drift_is_still_held(self):
        """Measured 0.02px for a locked-off shot over moving water."""
        self.assertEqual(self.classify(window(0.02)), motion.HELD)

    def test_slow_pan_is_a_move(self):
        self.assertEqual(self.classify(window(8.0)), motion.MOVE)

    def test_fast_but_straight_pan_is_still_a_move(self):
        """A quick reveal is a legitimate shot; only jitter or the ceiling cut it."""
        self.assertEqual(self.classify(window(80.0)), motion.MOVE)

    def test_erratic_motion_is_repositioning(self):
        self.assertEqual(
            self.classify(window(76.0, drift=43.0)), motion.REPOSITION
        )

    def test_speed_ceiling_rejects_a_whip_pan(self):
        self.assertEqual(self.classify(window(200.0), max_rate=1.0), motion.REPOSITION)

    def test_ceiling_is_configurable(self):
        self.assertEqual(self.classify(window(200.0), max_rate=5.0), motion.MOVE)

    def test_featureless_still_frame_is_held_not_discarded(self):
        """Fog or a blown sky gives no correlation peak; assuming the camera is
        parked keeps the footage rather than throwing it away on a bad number."""
        self.assertEqual(self.classify(window(0.0, response=0.01)), motion.HELD)

    def test_unmeasurable_but_moving_is_repositioning(self):
        """No peak *and* a long displacement means the motion is chaotic."""
        self.assertEqual(self.classify(window(90.0, response=0.01)), motion.REPOSITION)

    def test_thresholds_are_resolution_independent(self):
        """Same movement, double the frame: same verdict."""
        self.assertEqual(
            motion.classify(window(8.0), 512, FPS),
            motion.classify(window(16.0), 1024, FPS),
        )

    def test_thresholds_are_sample_rate_independent(self):
        """Half the sampling rate means twice the movement per sample."""
        self.assertEqual(
            motion.classify(window(8.0), WIDTH, 4.0),
            motion.classify(window(16.0), WIDTH, 2.0),
        )

    def test_degenerate_inputs_do_not_crash(self):
        self.assertEqual(motion.classify(window(8.0), 0, FPS), motion.MOVE)
        self.assertEqual(motion.classify(window(8.0), WIDTH, 0), motion.MOVE)


class TestRuns(unittest.TestCase):
    def test_groups_consecutive_labels(self):
        data = samples([(0, 0)] * 3 + [(8, 0)] * 3)
        labels = [motion.HELD] * 3 + [motion.MOVE] * 3
        runs = motion.to_runs(data, labels)
        self.assertEqual([r.motion_class for r in runs], [motion.HELD, motion.MOVE])
        self.assertEqual(runs[0].start, 0.0)
        self.assertEqual(runs[-1].end, data[-1].end)

    def test_runs_tile_the_timeline(self):
        data = samples([(0, 0), (8, 0), (0, 0)])
        runs = motion.to_runs(data, [motion.HELD, motion.MOVE, motion.HELD])
        for previous, following in zip(runs, runs[1:]):
            self.assertEqual(previous.end, following.start)


class TestMergeShortRuns(unittest.TestCase):
    def test_two_second_reposition_survives(self):
        """The bug this exists to prevent: judged by min_shot_seconds a 2s
        reposition gets folded into the shot beside it and relabelled usable."""
        runs = [
            Run(0.0, 4.0, motion.MOVE),
            Run(4.0, 6.0, motion.REPOSITION),
            Run(6.0, 10.0, motion.MOVE),
        ]
        merged = motion.merge_short_runs(runs, min_seconds=3.0, min_reposition_seconds=1.0)
        self.assertEqual(
            [r.motion_class for r in merged],
            [motion.MOVE, motion.REPOSITION, motion.MOVE],
        )

    def test_brief_wobble_is_absorbed(self):
        """Half a second of turbulence mid-pan should not split the shot."""
        runs = [
            Run(0.0, 4.0, motion.MOVE),
            Run(4.0, 4.4, motion.REPOSITION),
            Run(4.4, 9.0, motion.MOVE),
        ]
        merged = motion.merge_short_runs(runs, min_seconds=3.0, min_reposition_seconds=1.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].motion_class, motion.MOVE)

    def test_short_usable_run_is_absorbed(self):
        runs = [
            Run(0.0, 1.0, motion.MOVE),
            Run(1.0, 8.0, motion.HELD),
        ]
        merged = motion.merge_short_runs(runs, min_seconds=3.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].motion_class, motion.HELD)

    def test_merging_preserves_the_whole_timeline(self):
        runs = [
            Run(0.0, 1.0, motion.MOVE),
            Run(1.0, 1.4, motion.REPOSITION),
            Run(1.4, 9.0, motion.HELD),
            Run(9.0, 9.5, motion.MOVE),
        ]
        merged = motion.merge_short_runs(runs, min_seconds=3.0)
        self.assertEqual(merged[0].start, 0.0)
        self.assertEqual(merged[-1].end, 9.5)
        for previous, following in zip(merged, merged[1:]):
            self.assertEqual(previous.end, following.start)

    def test_short_run_joins_the_longer_neighbour(self):
        runs = [
            Run(0.0, 1.0, motion.HELD),
            Run(1.0, 2.0, motion.MOVE),
            Run(2.0, 12.0, motion.REPOSITION),
        ]
        merged = motion.merge_short_runs(runs, min_seconds=3.0, min_reposition_seconds=1.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].motion_class, motion.REPOSITION)

    def test_single_run_untouched(self):
        runs = [Run(0.0, 1.0, motion.MOVE)]
        self.assertEqual(motion.merge_short_runs(runs, min_seconds=3.0), runs)

    def test_adjacent_same_class_runs_collapse(self):
        runs = [
            Run(0.0, 5.0, motion.MOVE),
            Run(5.0, 5.5, motion.REPOSITION),
            Run(5.5, 11.0, motion.MOVE),
        ]
        merged = motion.merge_short_runs(runs, min_seconds=3.0, min_reposition_seconds=1.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (0.0, 11.0))

    def test_empty(self):
        self.assertEqual(motion.merge_short_runs([], min_seconds=3.0), [])


class TestSegment(unittest.TestCase):
    def test_splits_a_move_reposition_move_file(self):
        """The case that motivated the stage: one file, two usable shots."""
        data = samples(
            [(8, 0)] * 20                                  # 5s steady pan
            + [(60, 20), (-55, -25), (58, 18), (-52, -22)] * 3  # 3s searching
            + [(8, 0)] * 20                                # 5s steady pan
        )
        runs = motion.segment(data, WIDTH, FPS, min_seconds=3.0)
        classes = [r.motion_class for r in runs]
        self.assertEqual(classes, [motion.MOVE, motion.REPOSITION, motion.MOVE])

    def test_a_clean_pan_stays_one_shot(self):
        data = samples([(8, 0)] * 40)
        runs = motion.segment(data, WIDTH, FPS, min_seconds=3.0)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].motion_class, motion.MOVE)

    def test_a_locked_off_shot_stays_one_shot(self):
        data = samples([(0, 0)] * 40)
        runs = motion.segment(data, WIDTH, FPS, min_seconds=3.0)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].motion_class, motion.HELD)

    def test_no_samples(self):
        self.assertEqual(motion.segment([], WIDTH, FPS, min_seconds=3.0), [])


if __name__ == "__main__":
    unittest.main()
