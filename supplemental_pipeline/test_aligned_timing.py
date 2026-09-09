import unittest

from supplemental_pipeline.aligned_timing import keyframe_steps, steps_to_target


class AlignedTimingTests(unittest.TestCase):
    def test_cat_schedule_uses_absolute_timestamps(self):
        times = [i / 24 for i in range(14)]
        dt = 0.0002
        steps = keyframe_steps(times, dt, 13)
        self.assertEqual(steps[-1], 2708)
        self.assertTrue(all(abs(s * dt - t) <= dt / 2 + 1e-12 for s, t in zip(steps, times)))
        # Simulate the float accumulation performed by the MPM solver.
        now = 0.
        total = 0
        for target in steps[1:]:
            n = steps_to_target(target, now, dt)
            total += n
            for _ in range(n):
                now += dt
        self.assertEqual(total, steps[-1])

    def test_invalid_schedules_fail(self):
        for times, dt, count in [([0, 1], 0, 1), ([0, 1], .1, 2),
                                 ([1, 2], .1, 1), ([0, 0], .1, 1),
                                 ([0, .001], 1, 1), ([0, float('nan')], .1, 1)]:
            with self.subTest(times=times, dt=dt):
                with self.assertRaises(ValueError):
                    keyframe_steps(times, dt, count)
        with self.assertRaises(ValueError):
            steps_to_target(10, .002, .0002)


if __name__ == '__main__':
    unittest.main()
