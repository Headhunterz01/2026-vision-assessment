"""Regression tests with synthetic pose truth and recorded ambiguous corners."""
import unittest

import numpy as np

from reproject import Estimator, Geometry, valid_corners


def detection(points, label=2):
    points = np.asarray(points, np.float64)
    return dict(corners=points, center=points.mean(0),
                box=np.r_[points.min(0), points.max(0)], confidence=.9, class_id=label)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.geometry = Geometry()
        center = self.geometry.world_from_camera @ [0, .04, 4.5] + self.geometry.camera_origin
        self.truth = np.r_[center, 2.6]

    def test_single_board_recovers_known_center(self):
        for yaw in [2.2, 2.6, 3.0, -2.8]:
            truth = self.truth.copy()
            truth[3] = yaw
            corners = self.geometry.plates(truth)[0]['pixels']
            fit, _ = Estimator(self.geometry).estimate([detection(corners)], 0.)
            self.assertIsNotNone(fit)
            np.testing.assert_allclose(fit['state'][:3], truth[:3], atol=1e-5)

    def test_noisy_two_board_pose(self):
        plates = self.geometry.plates(self.truth)
        visible = [p for p in plates if p['front']]
        self.assertEqual(len(visible), 2)
        rng = np.random.default_rng(42)
        ds = [detection(p['pixels'] + rng.normal(0, .2, (4, 2)), i + 1)
              for i, p in enumerate(visible)]
        fit, status = Estimator(self.geometry).estimate(ds, 0.)
        self.assertEqual(status, 'MULTI PLATE')
        np.testing.assert_allclose(fit['state'][:3], self.truth[:3], atol=.02)
        self.assertLess(max(fit['errors']), .5)

    def test_each_anchor_gives_same_center(self):
        for plate in self.geometry.plates(self.truth):
            angle = plate['yaw']
            recovered = plate['position'] - .25 * np.array([np.cos(angle), np.sin(angle), 0])
            np.testing.assert_allclose(recovered, self.truth[:3], atol=1e-12)

    def test_nonzero_camera_extrinsic_round_trip(self):
        geometry = Geometry(gimbal_pitch=1.34)
        center = geometry.world_from_camera @ [0, 0, 4.5] + geometry.camera_origin
        truth = np.r_[center, 2.6]
        plate = geometry.plates(truth)[0]
        np.testing.assert_allclose(geometry.world_from_camera @ plate['camera_position']
                                   + geometry.camera_origin, plate['position'], atol=1e-12)
        np.testing.assert_allclose(geometry.world_from_camera @ plate['camera_rotation'],
                                   plate['rotation'], atol=1e-12)
        fit = geometry.fit(geometry.pnp_seeds(plate['pixels'])[0], [(0, plate['pixels'])])
        self.assertIsNotNone(fit)
        np.testing.assert_allclose(fit['state'][:3], center, atol=1e-4)

    def test_empty_input_clears_display_and_expires_prior(self):
        d = detection(self.geometry.plates(self.truth)[0]['pixels'])
        estimator = Estimator(self.geometry)
        self.assertIsNotNone(estimator.estimate([d], 0.)[0])
        self.assertIsNone(estimator.estimate([], .1)[0])
        self.assertIsNone(estimator.estimate([], .4)[0])
        self.assertIsNone(estimator.last)

    def test_recorded_frame_978_rejects_remote_mirror_center(self):
        right = detection([[802.39990, 606.10162], [811.46484, 650.86755],
                           [891.95685, 635.80115], [884.47504, 593.86316]])
        left = detection([[553.52179, 604.45142], [548.88013, 642.62396],
                          [617.40240, 651.78387], [621.86859, 612.00391]], 1)
        fit, status = Estimator(self.geometry).estimate([right, left], 0.)
        self.assertEqual(status, 'MULTI PLATE')
        self.assertTrue(650 < fit['center_pixel'][0] < 800)
        self.assertLess(max(fit['errors']), 8)

    def test_contradictory_board_cannot_support_model(self):
        ds = [detection(self.geometry.plates(self.truth)[0]['pixels'])]
        ds.append(detection(ds[0]['corners'] + [500, 200]))
        self.assertIsNone(Estimator(self.geometry).estimate(ds, 0.)[0])

    def test_behind_camera_and_invalid_parameters(self):
        behind = self.truth.copy()
        behind[:3] = self.geometry.world_from_camera @ [0, 0, -4] + self.geometry.camera_origin
        self.assertTrue(all(not p['positive'] and p['pixels'] is None for p in self.geometry.plates(behind)))
        self.assertFalse(valid_corners([[0, 0]] * 4))
        for radius in [0, -1, np.nan]:
            with self.assertRaises(ValueError):
                Geometry(radius=radius)


if __name__ == '__main__':
    unittest.main()
