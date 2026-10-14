"""Refine NN armor corners and constrain neighboring plates with blue LEDs.

The neural network still selects the armor and supplies all pose observations.
Image evidence is restricted to its chassis region. An isolated LED supplies
only two endpoints of an adjacent plate; it never creates a new detected armor.
The geometry object owns the immutable radius, pitch, calibration and model.
"""
from copy import deepcopy
from itertools import combinations, product

import cv2
import numpy as np
from scipy.optimize import least_squares

EDGES = ((0, 1), (3, 2))


def _rms(points, target):
    return float(np.sqrt(np.mean(np.sum((np.asarray(points) - target) ** 2, axis=1))))


def _length(points):
    return float(np.linalg.norm(np.asarray(points)[1] - np.asarray(points)[0]))


def _direction(points):
    delta = np.asarray(points)[1] - np.asarray(points)[0]
    return delta / max(float(np.linalg.norm(delta)), 1e-9)


def extract_lightbars(image, detections):
    """Return top/bottom LED endpoints inside an NN-derived chassis region.

    No NN observation means no region and no LED detections. The region stops
    above ground reflections; lengths adapt to the NN board's image height.
    """
    if not detections or image is None or image.ndim != 3:
        return []
    corners = [np.asarray(d['corners'], dtype=float) for d in detections]
    corners = [q for q in corners if q.shape == (4, 2) and np.isfinite(q).all()]
    if not corners:
        return []
    height = float(np.median([(_length(q[[0, 1]]) + _length(q[[3, 2]])) / 2 for q in corners]))
    width = max(float(np.linalg.norm(q[3] - q[0])) for q in corners)
    if height < 4 or width < 4:
        return []
    points = np.concatenate(corners)
    rows, cols = image.shape[:2]
    x0 = max(0, int(np.floor(points[:, 0].min() - 2.5 * width)))
    x1 = min(cols, int(np.ceil(points[:, 0].max() + 2.5 * width)) + 1)
    y0 = max(0, int(np.floor(points[:, 1].min() - height)))
    y1 = min(rows, int(np.ceil(points[:, 1].max() + .8 * height)) + 1)
    if x1 <= x0 or y1 <= y0:
        return []
    blue, green, red = cv2.split(image[y0:y1, x0:x1])
    blue = blue.astype(np.int16)
    # Saturated LED cores and their blue halo remain connected; a high red/green
    # difference avoids the broad blue flare that biases principal directions.
    mask = ((blue > 90) & (blue - red.astype(np.int16) > 45)
            & (blue - green.astype(np.int16) > 10)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bars = []
    for contour in contours:
        if len(contour) < 8 or cv2.contourArea(contour) < 12:
            continue
        cloud = contour[:, 0, :].astype(float)
        center = cloud.mean(0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(cloud.T))
        direction = eigenvectors[:, 1]
        if direction[1] < 0:
            direction = -direction
        distance = (cloud - center) @ direction
        length = float(np.ptp(distance))
        if not max(8., .35 * height) < length < 1.8 * height:
            continue
        if direction[1] < .75 or eigenvalues[1] / max(eigenvalues[0], .01) < 4:
            continue
        endpoints = np.array([center + direction * distance.min(),
                              center + direction * distance.max()]) + [x0, y0]
        bars.append(endpoints)
    return sorted(bars, key=lambda b: (float(b[:, 0].mean()), float(b[:, 1].mean())))


def refine_corners(detections, bars):
    """Copy NN detections and refine nearby sides; return copies and used IDs.

    Corner order remains TL, BL, BR, TR. Each LED is assigned at most once.
    Unmatched corners retain their NN positions, including on missing LED frames.
    """
    refined = deepcopy(detections)
    proposals = []
    for detection_id, d in enumerate(refined):
        corners = np.asarray(d['corners'], dtype=float).copy()
        d['corners'] = corners
        d['nn_corners'] = corners.copy()
        d['lightbar_refined'] = False
        d['lightbar_ids'] = []
        for edge in EDGES:
            target = corners[list(edge)]
            height = _length(target)
            for bar_id, bar in enumerate(bars):
                bar = np.asarray(bar, dtype=float)
                if bar.shape != (2, 2) or not np.isfinite(bar).all():
                    continue
                error = _rms(bar, target)
                ratio = _length(bar) / max(height, 1e-9)
                # A blue halo can extend well beyond the physical LED. The
                # NN side supplies an independent size/direction check; do not
                # replace it with a long, tilted flare connected to the LED.
                endpoint_shift = np.max(np.linalg.norm(bar-target, axis=1))
                if (error < min(10., .2 * height) and .82 < ratio < 1.18
                        and endpoint_shift < min(10., .2 * height)
                        and _direction(bar) @ _direction(target) > .985):
                    proposals.append((error, detection_id, edge, bar_id))
    used_bars, used_edges = set(), set()
    for _, detection_id, edge, bar_id in sorted(proposals):
        if bar_id in used_bars or (detection_id, edge) in used_edges:
            continue
        d = refined[detection_id]
        d['corners'][list(edge)] = np.asarray(bars[bar_id])
        d['lightbar_refined'] = True
        d['lightbar_ids'].append(int(bar_id))
        used_bars.add(bar_id)
        used_edges.add((detection_id, edge))
    for d in refined:
        d['center'] = d['corners'].mean(0)
    return refined, sorted(used_bars)


def refine_fit_with_lightbars(geometry, fit, detections, bars, used_ids):
    """Fit NN observations plus validated edges of unobserved P1/P3 plates.

    Return the original object when evidence fails. Both unknown edges are
    considered, and a bar/plate is used only once. A lone frontal NN plate needs
    LED support from both sides, preventing a stray LED from inventing its yaw.
    """
    if fit is None or not bars or not fit.get('observations'):
        return fit
    observations = [(int(i), np.asarray(q, dtype=float)) for i, q in fit['observations']]
    observed_indices = {i for i, _ in observations}
    primary = next((q for i, q in observations if i == 0), observations[0][1])
    height = (_length(primary[[0, 1]]) + _length(primary[[3, 2]])) / 2
    if height < 4:
        return fit
    original_state = np.asarray(fit['state'], dtype=float)
    plates = geometry.plates(original_state)
    used = set(used_ids)
    primary_mid = primary.mean(0)
    choices = {i: [] for i in (1, 3) if i not in observed_indices}
    for bar_id, raw_bar in enumerate(bars):
        bar = np.asarray(raw_bar, dtype=float)
        if bar_id in used or bar.shape != (2, 2) or not np.isfinite(bar).all():
            continue
        if not .70 * height < _length(bar) < 1.35 * height:
            continue
        # A bar close to any observed side must not be reassigned as a neighbor,
        # even when its corner refinement failed a stricter angle/length gate.
        if any(_rms(bar, q[list(edge)]) < 25 for _, q in observations for edge in EDGES):
            continue
        for plate_id, candidates in choices.items():
            p = plates[plate_id]
            if not p['positive'] or p['pixels'] is None:
                continue
            for edge in EDGES:
                predicted = p['pixels'][list(edge)]
                length_ratio = _length(predicted) / max(_length(bar), 1e-9)
                if (_rms(predicted, bar) < min(85., 2 * height)
                        and .65 < length_ratio < 1.4
                        and _direction(predicted) @ _direction(bar) > .9):
                    candidates.append((bar_id, edge, bar))
    choices = {i: sorted(c, key=lambda e: _rms(plates[i]['pixels'][list(e[1])], e[2]))[:6]
               for i, c in choices.items() if c}
    if not choices:
        return fit
    single = len(observations) == 1
    # A frontal plane is weakly observable. Require two independent side LEDs.
    if single and len(choices) < 2:
        return fit
    candidates = []
    plate_ids = sorted(choices)
    sizes = [2] if single else list(range(len(plate_ids), 0, -1))
    for size in sizes:
        for selected in combinations(plate_ids, size):
            for edges in product(*(choices[i] for i in selected)):
                if len({edge[0] for edge in edges}) != len(edges):
                    continue
                support = [(i, *edge) for i, edge in zip(selected, edges)]
                if single:
                    mids = [bar.mean(0)[0] for _, _, _, bar in support]
                    if not min(mids) < primary_mid[0] < max(mids):
                        continue
                count = 8 * len(observations) + 4 * len(support)

                def residual(state):
                    projected = geometry.plates(state)
                    values = []
                    for i, q in observations:
                        p = projected[i]
                        if not p['positive'] or p['pixels'] is None:
                            return np.full(count, 1e6)
                        values.extend((p['pixels'] - q).ravel())
                    for i, _, edge, bar in support:
                        p = projected[i]
                        if not p['positive'] or p['pixels'] is None:
                            return np.full(count, 1e6)
                        values.extend((.5 * (p['pixels'][list(edge)] - bar)).ravel())
                    return np.asarray(values)

                optimized = least_squares(residual, original_state, method='lm', max_nfev=100,
                                          ftol=1e-6, xtol=1e-6, gtol=1e-6)
                state = optimized.x
                if not np.isfinite(state).all():
                    continue
                new_plates = geometry.plates(state)
                errors = np.array([_rms(new_plates[i]['pixels'], q) for i, q in observations])
                side_errors = [_rms(new_plates[i]['pixels'][list(edge)], bar)
                               for i, _, edge, bar in support]
                # Allow existing multi-NN residual floors, but enforce a strict
                # primary gate for single-board corrections with external LEDs.
                if errors[0] > 4 or max(errors) > 8 or max(side_errors) > 8:
                    continue
                if np.linalg.norm(state[:3] - original_state[:3]) > .4:
                    continue
                yaw_delta = np.arctan2(np.sin(state[3] - original_state[3]),
                                       np.cos(state[3] - original_state[3]))
                if abs(yaw_delta) > .5:
                    continue
                state[3] = np.arctan2(np.sin(state[3]), np.cos(state[3]))
                score = float(np.mean(residual(state) ** 2))
                candidates.append((score, state.copy(), new_plates, errors, support, side_errors))
        if candidates:
            break
    if not candidates:
        return fit
    _, state, new_plates, errors, support, side_errors = min(candidates, key=lambda c: c[0])
    result = dict(fit, state=state, plates=new_plates, errors=errors)
    result['side_support'] = [dict(plate=int(i), edge=list(edge), bar=bar.tolist(), rms=float(error))
                              for (i, _, edge, bar), error in zip(support, side_errors)]
    return result
