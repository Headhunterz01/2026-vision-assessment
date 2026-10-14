"""Image-motion-adaptive smoothing, with quarter-turn armor reindexing."""
import math

import cv2
import numpy as np


def smoothing_alpha(dt, motion_px):
    # Measured image motion drives the cutoff, not noisy PnP yaw. Thus a
    # mirror-branch jump on a still image cannot disable smoothing.
    if motion_px is None:
        return 1.
    cutoff = min(80., .6 + 40. * max(0., motion_px - .08))
    return 1. - math.exp(-2. * math.pi * cutoff * dt)


class VisualMotion:
    def __init__(self):
        self.gray = None
        self.detections = []
        self.stamp = None

    def update(self, image, detections, stamp):
        if image is None:
            return None
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, fx=.5, fy=.5)
        motion = None
        if self.gray is not None and self.stamp is not None and 0 < stamp-self.stamp <= .3:
            mask = np.zeros_like(gray)
            for d in self.detections:
                points = np.rint(np.asarray(d['corners']) * .5).astype(np.int32)
                cv2.fillConvexPoly(mask, points, 255)
            mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
            features = cv2.goodFeaturesToTrack(self.gray, 60, .015, 3, mask=mask)
            if features is not None and len(features) >= 6:
                tracked, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, features, None,
                    winSize=(17, 17), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .01))
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, tracked, None,
                    winSize=(17, 17), maxLevel=3)
                valid = ((status.ravel() != 0) & (back_status.ravel() != 0)
                         & (np.linalg.norm((back-features).reshape(-1, 2), axis=1) < .5))
                if valid.sum() >= 6:
                    displacement = (tracked-features).reshape(-1, 2)[valid] * 2.
                    motion = float(np.median(np.linalg.norm(displacement, axis=1)))
        self.gray, self.detections, self.stamp = gray, detections, stamp
        return motion


class CornerSmoother:
    def __init__(self):
        self.previous = []
        self.stamp = None

    def update(self, detections, stamp, motion_px):
        dt = stamp-self.stamp if self.stamp is not None else 0.
        alpha = smoothing_alpha(dt, motion_px) if 0 < dt <= .3 else 1.
        output, used = [], set()
        for d in detections:
            item = dict(d)
            corners = np.asarray(d['corners'], np.float64).copy()
            if alpha < 1 and self.previous:
                candidates = [(np.linalg.norm(np.mean(p['corners'], axis=0)-corners.mean(0)), i, p)
                              for i, p in enumerate(self.previous) if i not in used]
                if candidates:
                    distance, i, previous = min(candidates, key=lambda p: p[0])
                    # Never smear a different armor into a lost/replaced one.
                    scale = np.linalg.norm(corners[0]-corners[2])
                    if distance < max(12., .25*scale) and np.max(np.linalg.norm(
                            corners-np.asarray(previous['corners']), axis=1)) < max(15., .3*scale):
                        corners = np.asarray(previous['corners']) + alpha*(corners-previous['corners'])
                        used.add(i)
            item['corners'], item['center'] = corners, corners.mean(0)
            output.append(item)
        self.previous, self.stamp = output, stamp
        return output


class StateSmoother:
    def __init__(self):
        self.state = None
        self.stamp = None

    def reset(self):
        self.state = self.stamp = None

    def update(self, raw_state, stamp, motion_px):
        raw = np.asarray(raw_state, np.float64)
        dt = stamp-self.stamp if self.stamp is not None else 0.
        if self.state is None or not 0 < dt <= .3:
            self.state, self.stamp = raw.copy(), stamp
            return raw.copy(), 1.
        # Four identical plates have a phase period pi/2. Resolve the current
        # anchor label against the prior before smoothing, then restore it.
        phase = (raw[3]-self.state[3] + math.pi/4) % (math.pi/2) - math.pi/4
        aligned = raw.copy()
        aligned[3] = self.state[3]+phase
        offset = raw[3]-aligned[3]
        alpha = smoothing_alpha(dt, motion_px)
        # A different robot must start a fresh state even if the image is still.
        if np.linalg.norm(raw[:3]-self.state[:3]) > .5:
            alpha = 1.
        self.state += alpha*(aligned-self.state)
        self.stamp = stamp
        result = self.state.copy()
        result[3] = math.atan2(math.sin(result[3]+offset), math.cos(result[3]+offset))
        return result, alpha
