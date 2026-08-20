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

    def test_a_move_easing_into_a_hold_stays_one_shot(self):
        """Crossing HELD_RATE is not a shot boundary.

        Segmentation separates usable footage from repositioning. A drone
        slowing to a stop is one take, and splitting it where the speed dips
        below the held threshold turned 13 usable seconds into a 4.4s piece and
        a 9.0s piece, neither long enough to post.
        """
        data = samples([(8, 0)] * 20 + [(0, 0)] * 20)
        runs = motion.segment(data, WIDTH, FPS, min_seconds=3.0)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].start, 0.0)
        self.assertAlmostEqual(runs[0].end, data[-1].end)

    def test_repositioning_still_breaks_a_run(self):
        """Merging usable neighbours must not swallow a hunt between them."""
        data = samples(
            [(8, 0)] * 16 + [(30, 0), (-30, 0)] * 8 + [(8, 0)] * 16
        )
        classes = [r.motion_class for r in
                   motion.segment(data, WIDTH, FPS, min_seconds=3.0)]
        self.assertIn(motion.REPOSITION, classes)
        self.assertEqual(classes.count(motion.REPOSITION), 1)

    def test_merged_run_keeps_the_label_of_its_longer_half(self):
        Run = motion.Run
        merged = motion.merge_usable_runs([
            Run(0.0, 2.0, motion.MOVE),
            Run(2.0, 12.0, motion.HELD),
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].motion_class, motion.HELD)
        self.assertEqual((merged[0].start, merged[0].end), (0.0, 12.0))


class TestSpeedCurve(unittest.TestCase):
    """Speed steadiness — the thing `jitter` cannot see.

    A camera that stops dead and restarts holds its heading the whole time, so
    jitter reads it as a clean move. These are the per-second speed curves
    measured off real clips that were judged by eye, in frame widths per second
    (the units `speed_buckets` produces), so a hovering drone sits below
    HELD_RATE and a normal move is a few hundredths.
    """

    # A steady drift along a coastline, judged good by eye.
    STEADY = [0.085, 0.113, 0.131, 0.124, 0.125, 0.132, 0.141, 0.119, 0.141,
              0.121, 0.122, 0.138, 0.146, 0.124]
    # A move that accelerates hard and then collapses in its last seconds.
    STOP_AND_GO = [0.078, 0.093, 0.123, 0.139, 0.134, 0.157, 0.182, 0.198,
                   0.306, 0.400, 0.437, 0.246, 0.058, 0.040]
    # A drone lifting out of a hover: the opening seconds sit below HELD_RATE.
    EASE_IN = [0.008, 0.007, 0.039, 0.034, 0.065, 0.097, 0.085, 0.094, 0.116,
               0.206, 0.258]

    def test_steady_speed_has_no_reversals(self):
        self.assertEqual(motion.speed_reversals(self.STEADY), 0)

    def test_stop_and_go_reverses(self):
        self.assertGreater(motion.speed_reversals(self.STOP_AND_GO), 0)

    def test_a_single_ramp_is_not_a_reversal(self):
        """Speeding up throughout is an ease, however large the change."""
        self.assertEqual(motion.speed_reversals(self.EASE_IN), 0)
        self.assertGreater(motion.speed_spread(self.EASE_IN), 3.0)

    def test_near_static_noise_is_not_a_reversal(self):
        """Below the floor, relative swings are measurement noise."""
        self.assertEqual(
            motion.speed_reversals([0.0005, 0.008, 0.001, 0.009, 0.0005]), 0
        )

    def test_ease_in_is_recognised_from_rest(self):
        self.assertTrue(motion.eases_in(self.EASE_IN))
        self.assertFalse(motion.eases_in(self.STOP_AND_GO))

    def test_buckets_are_rates_not_pixels(self):
        """A rate, so the curve means the same at any resolution or sample rate.

        4px per sample at 4fps across a 512px frame is 16px/s, which is
        16/512 = 0.03125 frame widths per second.
        """
        data = samples([(4, 0)] * 12)  # 4 samples/sec at FPS=4 -> 3 buckets
        buckets = motion.speed_buckets(data, FPS, WIDTH)
        self.assertEqual(len(buckets), 3)
        self.assertAlmostEqual(buckets[0], 4.0 * FPS / WIDTH)

    def test_same_movement_at_double_resolution_reads_the_same(self):
        slow = motion.speed_buckets(samples([(4, 0)] * 8), FPS, 512)
        fast = motion.speed_buckets(samples([(8, 0)] * 8), FPS, 1024)
        self.assertAlmostEqual(slow[0], fast[0])

    def test_degenerate_width_or_rate_yields_nothing(self):
        self.assertEqual(motion.speed_buckets(samples([(4, 0)] * 8), FPS, 0), [])
        self.assertEqual(motion.speed_buckets(samples([(4, 0)] * 8), 0, WIDTH), [])


class TestFindWindows(unittest.TestCase):
    """Choosing which seconds to render, rather than the first ones.

    Profiles are in frame widths per second, as `speed_buckets` produces.
    """

    def test_steady_run_yields_one_window(self):
        spans = motion.find_windows([0.03] * 14, start=0.0, min_seconds=10, max_seconds=15)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].reversals, 0)
        self.assertGreaterEqual(spans[0].duration, 10)

    def test_window_skips_a_lurch_at_the_front(self):
        """The calm tail is chosen over the stop-and-go at the start."""
        profile = [0.055, 0.012, 0.045] + [0.03] * 12
        spans = motion.find_windows(profile, start=100.0, min_seconds=10, max_seconds=15)
        self.assertEqual(len(spans), 1)
        self.assertGreaterEqual(spans[0].start, 102.0)
        self.assertEqual(spans[0].reversals, 0)

    def test_two_clean_stretches_yield_two_windows(self):
        """A long take holding two good sections gives two clips, not one."""
        profile = [0.03] * 12 + [0.009, 0.042, 0.011] + [0.031] * 12
        spans = motion.find_windows(profile, start=0.0, min_seconds=10, max_seconds=15)
        self.assertEqual(len(spans), 2)
        self.assertLessEqual(spans[0].end, spans[1].start)

    def test_windows_never_overlap(self):
        spans = motion.find_windows([0.03] * 40, start=0.0, min_seconds=10, max_seconds=15)
        for earlier, later in zip(spans, spans[1:]):
            self.assertLessEqual(earlier.end, later.start)

    def test_run_shorter_than_the_minimum_yields_nothing(self):
        self.assertEqual(
            motion.find_windows([0.03] * 6, start=0.0, min_seconds=10, max_seconds=15), []
        )

    def test_a_reveal_from_rest_is_not_rejected_for_spread(self):
        """An ease-in has a huge fast/slow ratio by definition; keep it.

        This is the case that failed on real footage: a drone lifting out of a
        hover measured a 37x spread and was thrown away, despite being the best
        clip in the set. The opening seconds sit below HELD_RATE, which is what
        marks it as a reveal rather than a shot that changes speed mid-flight.
        """
        profile = TestSpeedCurve.EASE_IN
        self.assertGreater(motion.speed_spread(profile), 3.0)
        spans = motion.find_windows(
            profile, start=0.0, min_seconds=10, max_seconds=15, max_spread=3.0
        )
        self.assertEqual(len(spans), 1)

    def test_a_cruise_that_speeds_up_is_not_exempt(self):
        """The same ramp starting from an existing cruise is a departure.

        Measured off a clip that begins already moving and accelerates away in
        its final seconds — judged bad by eye, and the shape the ease-in
        exemption must not let through.
        """
        profile = [0.100, 0.124, 0.147, 0.120, 0.109, 0.149, 0.206, 0.242,
                   0.257, 0.398]
        self.assertFalse(motion.eases_in(profile))
        spans = motion.find_windows(
            profile, start=0.0, min_seconds=10, max_seconds=15, max_spread=3.0
        )
        self.assertEqual(spans, [])

    def test_the_exemption_does_not_outbid_a_strict_window(self):
        """A drift that sits below the floor then accelerates has the same shape
        as a reveal, so the exemption cannot tell them apart. Letting it compete
        on length lost the good part: measured on a take that drifted for five
        seconds and then settled into a steady orbit, the exempt window spanned
        both at 11.6x spread while a strict one sat inside the orbit at 1.8x.
        """
        profile = (
            [0.017] * 5 + [0.007, 0.007, 0.003, 0.002, 0.003, 0.002, 0.014]
            + [0.044, 0.056, 0.086, 0.097, 0.110, 0.114, 0.116, 0.136, 0.149,
               0.159, 0.169, 0.170, 0.149, 0.125, 0.103, 0.088, 0.078]
        )
        spans = motion.find_windows(
            profile, start=0.0, min_seconds=10, max_seconds=15, max_spread=3.0
        )
        self.assertTrue(spans)
        for span in spans:
            self.assertLessEqual(span.spread, 3.0)

    def test_a_reveal_still_wins_when_nothing_else_qualifies(self):
        """The exemption is a fallback, not a deletion: a genuine reveal has no
        strict alternative, so it must still come back."""
        spans = motion.find_windows(
            TestSpeedCurve.EASE_IN, start=0.0, min_seconds=10, max_seconds=15,
            max_spread=3.0,
        )
        self.assertEqual(len(spans), 1)
        self.assertGreater(spans[0].spread, 3.0)

    def test_longer_window_wins_when_both_are_clean(self):
        spans = motion.find_windows([0.03] * 15, start=0.0, min_seconds=10, max_seconds=15)
        self.assertAlmostEqual(spans[0].duration, 15.0)


if __name__ == "__main__":
    unittest.main()
