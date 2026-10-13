"""LED evidence must constrain the rigid model without creating NN detections."""
import unittest

import cv2
import numpy as np

from image_evidence import extract_lightbars, refine_corners, refine_fit_with_lightbars
from reproject import Geometry


def detection(corners):
    corners = np.asarray(corners, dtype=float)
    return dict(corners=corners, center=corners.mean(0), confidence=.9,
                box=np.r_[corners.min(0), corners.max(0)], class_id=2)


class ImageEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.geometry = Geometry()
        # A recorded near-frontal NN board whose slight corner skew produced
        # a roughly 60 px error at the unobserved left plate.
        self.detections = [detection([[634.999947, 613.487755], [638.676453, 659.510376],
                                     [746.439491, 656.263092], [743.430061, 610.514236]])]
        self.seed = np.array([4.528627235, .061796652, -.530782445, -2.854447826])
        self.bars = [np.array(b) for b in [
            [[507.004581, 602.501143], [496.710504, 643.677623]],
            [[641.273757, 613.945525], [639.850129, 659.058385]],
            [[741.991446, 611.920841], [743.788388, 657.127660]],
            [[847.055828, 593.693457], [859.044776, 634.987000]]]]

    def fit(self, detections):
        fit = self.geometry.fit(self.seed, [(0, detections[0]['corners'])], max_rms=4)
        self.assertIsNotNone(fit)
        return fit

    def test_missing_evidence_preserves_observations_and_fit(self):
        refined, used = refine_corners(self.detections, [])
        self.assertIsNot(refined[0], self.detections[0])
        np.testing.assert_array_equal(refined[0]['corners'], self.detections[0]['corners'])
        self.assertEqual(used, [])
        fit = self.fit(refined)
        self.assertIs(refine_fit_with_lightbars(self.geometry, fit, refined, [], []), fit)
        self.assertEqual(extract_lightbars(np.zeros((1080, 1440, 3), np.uint8), []), [])

    def test_recorded_leds_correct_remote_plate_and_keep_fixed_geometry(self):
        raw_corners = self.detections[0]['corners'].copy()
        refined, used = refine_corners(self.detections, self.bars)
        self.assertEqual(used, [1, 2])
        np.testing.assert_array_equal(self.detections[0]['corners'], raw_corners)
        fit = self.fit(refined)
        fixed = refine_fit_with_lightbars(self.geometry, fit, refined, self.bars, used)
        self.assertIsNot(fixed, fit)
        self.assertEqual({s['plate'] for s in fixed['side_support']}, {1, 3})
        self.assertLess(max(s['rms'] for s in fixed['side_support']), 5)
        self.assertLess(fixed['errors'][0], 3)
        self.assertLess(abs(fixed['state'][3] + 3.08), .01)
        for p in fixed['plates']:
            self.assertAlmostEqual(np.linalg.norm(p['position'] - fixed['state'][:3]), .25, places=12)
        self.assertEqual(self.geometry.pitch, np.pi / 12)
        # Auxiliary evidence is permitted for nearly tangential, back-facing
        # boards: filtering by front-facing plane would lose this correction.
        self.assertFalse(fixed['plates'][1]['front'])

    def test_single_or_short_side_led_cannot_invent_frontal_yaw(self):
        refined, used = refine_corners(self.detections, self.bars)
        fit = self.fit(refined)
        self.assertIs(refine_fit_with_lightbars(self.geometry, fit, refined,
                                               [self.bars[0]], []), fit)
        short = [b.copy() for b in self.bars]
        for index in (0, 3):
            middle = short[index].mean(0)
            short[index] = middle + (short[index] - middle) * .45
        self.assertIs(refine_fit_with_lightbars(self.geometry, fit, refined, short, used), fit)
        spurious = [b + [0, 120] for b in self.bars]
        self.assertIs(refine_fit_with_lightbars(self.geometry, fit, refined, spurious, []), fit)

    def test_image_region_excludes_floor_and_round_blue_contours(self):
        image = np.zeros((1080, 1440, 3), np.uint8)
        for bar in self.bars:
            cv2.line(image, tuple(np.rint(bar[0]).astype(int)), tuple(np.rint(bar[1]).astype(int)),
                     (255, 90, 20), 5)
        cv2.line(image, (640, 810), (640, 860), (255, 90, 20), 5)
        cv2.circle(image, (580, 620), 8, (255, 90, 20), -1)
        found = extract_lightbars(image, self.detections)
        self.assertEqual(len(found), 4)
        self.assertTrue(all(b[:, 1].max() < 720 for b in found))
        for actual, expected in zip(found, self.bars):
            self.assertLess(np.linalg.norm(actual.mean(0) - expected.mean(0)), 1)

    def test_long_tilted_halo_does_not_replace_nn_led(self):
        d = detection([[685.51, 611.48], [685.78, 653.43],
                       [794.72, 655.31], [794.68, 612.83]])
        halo = np.array([[692.29, 612.47], [682.09, 661.62]])
        refined, used = refine_corners([d], [halo])
        self.assertEqual(used, [])
        np.testing.assert_array_equal(refined[0]['corners'], d['corners'])


if __name__ == '__main__':
    unittest.main()
