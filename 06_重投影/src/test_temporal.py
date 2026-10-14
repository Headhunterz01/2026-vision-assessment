"""Temporal regression checks: still-image noise, spinning and anchor changes."""
import math
import unittest

import numpy as np

from temporal import StateSmoother


def yaw_difference(a, b):
    """Physical yaw difference for equivalent four-board phases."""
    return (a-b+math.pi/4) % (math.pi/2) - math.pi/4


class StateSmootherTests(unittest.TestCase):
    def test_stationary_noise_is_reduced_without_freezing_state(self):
        rng = np.random.default_rng(71)
        smoother = StateSmoother()
        truth = np.array([4.6, .04, -.54, math.pi-.015])
        raw_states, filtered_states = [], []
        for i in range(500):
            raw = truth + rng.normal(0, [.03, .015, .003, .05])
            # Anchor relabels and +/-pi wrapping must not introduce motion.
            raw[3] += (i//31 % 4) * math.pi/2
            raw[3] = math.atan2(math.sin(raw[3]), math.cos(raw[3]))
            filtered, _ = smoother.update(raw, i*.02, .01)
            if i >= 100:
                raw_states.append(raw)
                filtered_states.append(filtered)
        raw_states, filtered_states = np.array(raw_states), np.array(filtered_states)
        raw_center_noise = np.sqrt(np.mean((raw_states[:, :3]-truth[:3])**2))
        filtered_center_noise = np.sqrt(np.mean((filtered_states[:, :3]-truth[:3])**2))
        self.assertLess(filtered_center_noise, .4*raw_center_noise)
        raw_yaw_noise = np.std(yaw_difference(raw_states[:, 3], truth[3]))
        filtered_yaw_noise = np.std(yaw_difference(filtered_states[:, 3], truth[3]))
        self.assertLess(filtered_yaw_noise, .4*raw_yaw_noise)
        # Small state updates remain observable; the filter is not a hold flag.
        self.assertGreater(np.ptp(filtered_states[:, 0]), .001)
        self.assertGreater(np.ptp(yaw_difference(filtered_states[:, 3], truth[3])), .001)

    def test_fast_constant_rotation_preserves_motion_and_anchor_phase(self):
        smoother = StateSmoother()
        center = np.array([4.6, .04, -.54])
        velocity = np.array([.15, -.04, .005])
        rate = math.radians(400.)
        center_errors, phase_errors, yaw_errors = [], [], []
        for i in range(250):
            stamp = .02*i
            # This video rotates about 8 degrees per processed frame. Every
            # seventh frame choose a new physical board as the pose anchor.
            raw = np.r_[center + velocity*stamp,
                        3.10 + rate*stamp + (i//7 % 4)*math.pi/2]
            raw[3] = math.atan2(math.sin(raw[3]), math.cos(raw[3]))
            filtered, _ = smoother.update(raw, stamp, 3.)
            center_errors.append(np.linalg.norm(filtered[:3]-raw[:3]))
            phase_errors.append(abs(yaw_difference(filtered[3], raw[3])))
            yaw_errors.append(abs(math.atan2(math.sin(filtered[3]-raw[3]),
                                            math.cos(filtered[3]-raw[3]))))
        self.assertLess(max(center_errors), 1e-5)
        self.assertLess(max(phase_errors), math.radians(.002))
        # The returned model also restores the chosen anchor orientation.
        self.assertLess(max(yaw_errors), math.radians(.002))

    def test_motion_onset_and_stop_do_not_hold_or_overshoot(self):
        smoother = StateSmoother()
        rng = np.random.default_rng(32)
        initial = np.array([4.6, .04, -.54, 2.9])
        for i in range(50):
            smoother.update(initial, i*.02, 0.)
        moved = initial + [.08, -.025, .005, .3]
        filtered, _ = smoother.update(moved, 1., 3.)
        self.assertLess(np.linalg.norm(filtered[:3]-moved[:3]), 1e-5)
        self.assertLess(abs(yaw_difference(filtered[3], moved[3])), math.radians(.002))
        raw_noise, filtered_noise = [], []
        for i in range(1, 151):
            raw = moved + rng.normal(0, [.03, .015, .003, .05])
            filtered, _ = smoother.update(raw, 1.+i*.02, .01)
            if i >= 40:
                raw_noise.append(yaw_difference(raw[3], moved[3]))
                filtered_noise.append(yaw_difference(filtered[3], moved[3]))
        self.assertLess(np.std(filtered_noise), .4*np.std(raw_noise))
        self.assertLess(abs(np.mean(filtered_noise)), .02)

    def test_missing_motion_measurement_does_not_suppress_real_motion(self):
        smoother = StateSmoother()
        initial = np.array([4.6, .04, -.54, 2.9])
        smoother.update(initial, 0., .01)
        changed = initial + [.1, .05, 0., .3]
        filtered, _ = smoother.update(changed, .02, None)
        np.testing.assert_allclose(filtered[:3], changed[:3], atol=1e-12)
        self.assertLess(abs(yaw_difference(filtered[3], changed[3])), 1e-12)

    def test_dropout_time_restart_and_explicit_reset_reacquire_current_pose(self):
        initial = np.array([4.6, .04, -.54, 2.9])
        fresh = initial + [.12, -.03, .01, .31]
        for next_stamp in [.61, 0., -.02]:
            smoother = StateSmoother()
            smoother.update(initial, .1, .01)
            filtered, _ = smoother.update(fresh, next_stamp, .01)
            np.testing.assert_allclose(filtered, fresh, atol=1e-12)
        smoother = StateSmoother()
        smoother.update(initial, 0., .01)
        smoother.reset()
        filtered, _ = smoother.update(fresh, .02, .01)
        np.testing.assert_allclose(filtered, fresh, atol=1e-12)

    def test_different_robot_reacquires_instead_of_blending_positions(self):
        smoother = StateSmoother()
        initial = np.array([4.6, .04, -.54, 2.9])
        smoother.update(initial, 0., .01)
        fresh = initial + [.7, .2, -.03, -.2]
        filtered, _ = smoother.update(fresh, .02, .01)
        np.testing.assert_allclose(filtered[:3], fresh[:3], atol=1e-12)
        self.assertLess(abs(yaw_difference(filtered[3], fresh[3])), 1e-12)


if __name__ == '__main__':
    unittest.main()
