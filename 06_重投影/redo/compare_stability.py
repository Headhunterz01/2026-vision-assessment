#!/usr/bin/env python3
"""Compare initial/final stationary geometry and moving observation alignment.

Model board IDs are cyclically realigned before corner step comparisons. The
stationary windows are supported by source-video optical flow; moving corner
steps are motion diagnostics, not estimates of jitter. Image residuals measure
agreement with the same observations used by the estimator, not pose accuracy.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

WINDOWS = {'initial_stationary': (0., 8.),
           'moving': (9., 55.8),
           'final_stationary': (60., 63.)}


def summary(values):
    a = np.asarray(values, float).ravel()
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return dict(n=int(len(a)), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(np.max(a)))


def load_records(path):
    records = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
    return records


def corner_rms(a, b):
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - b) ** 2, axis=-1))))


def detection_points(record, stage):
    # Old logs contain raw NN data in detections; new logs explicitly preserve
    # raw_detections while detections contains the corners actually fitted.
    field = 'raw_detections' if stage == 'raw' else 'refined_detections'
    detections = record.get(field, record.get('detections', []))
    result = []
    for detection in detections:
        key = 'raw_corners' if stage == 'raw' else 'refined_corners'
        points = np.asarray(detection.get(key, detection.get('corners')), float)
        if points.shape == (4, 2) and np.isfinite(points).all():
            result.append(dict(corners=points, center=points.mean(0)))
    return result


def area(detection):
    p = detection['corners']
    return abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))
               - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def observation_residuals(record, stage):
    model = record.get('model')
    if model is None:
        return []
    used = detection_points(record, 'used')
    observations = detection_points(record, stage)
    if not used or not observations:
        return []
    pairs = [(0, model.get('anchor', 0))]
    pairs.extend((match[0], match[1]) for match in model.get('matches', []))
    errors = []
    for plate_index, detection_index in pairs:
        if not 0 <= detection_index < len(used):
            continue
        if not 0 <= plate_index < len(model['plates']):
            continue
        projection = np.asarray(model['plates'][plate_index]['pixels'], float)
        detection = used[detection_index]
        if stage == 'raw':
            # Preserve geometric correspondence if refinements remove or
            # reorder detections; classes are not permanent physical IDs.
            detection = min(observations, key=lambda d: np.linalg.norm(
                d['center'] - detection['center']))
            if np.linalg.norm(detection['center'] - used[detection_index]['center']) > 45:
                continue
        if projection.shape == (4, 2) and np.isfinite(projection).all():
            errors.append(dict(plate=int(plate_index),
                               rms=corner_rms(projection, detection['corners'])))
    return errors


def support_residuals(record):
    """Independently compare projected light-bar edges with recorded endpoints."""
    model = record.get('model')
    if model is None:
        return []
    result = []
    for support in model.get('side_support', []):
        if not isinstance(support, dict):
            continue
        index, edge = support.get('plate'), support.get('edge')
        if not isinstance(index, int) or not 0 <= index < len(model['plates']):
            continue
        if edge not in ([0, 1], [3, 2]):
            continue
        bar = np.asarray(support.get('bar'), float)
        pixels = np.asarray(model['plates'][index].get('pixels'), float)
        if bar.shape != (2, 2) or pixels.shape != (4, 2):
            continue
        if not np.isfinite(bar).all() or not np.isfinite(pixels).all():
            continue
        result.append(dict(plate=index, edge=edge, actual_bar_pixels=bar.tolist(),
                           projected_bar_pixels=pixels[edge].tolist(),
                           recomputed_rms_px=corner_rms(pixels[edge], bar),
                           logged_rms_px=support.get('rms')))
    return result


def collect(records, start, stop):
    rr = [r for r in records if start <= r['time'] < stop]
    aa, previous = [], None
    for record in rr:
        model = record.get('model')
        if model is None:
            continue
        p = np.asarray([p['pixels'] for p in model['plates']], float)
        if p.shape != (4, 4, 2) or not np.isfinite(p).all():
            continue
        if previous is not None:
            p = min((np.roll(p, k, axis=0) for k in range(4)),
                    key=lambda candidate: np.mean((candidate - previous) ** 2))
        aa.append((record, p))
        previous = p
    model_step, yaw_step, yaw_rate, center_step, center_pixel_step = [], [], [], [], []
    raw_step, used_step = [], []
    for (a, pa), (b, pb) in zip(aa, aa[1:]):
        dt = b['time'] - a['time']
        if not 0 < dt < .05:
            continue
        model_step.append(corner_rms(pb, pa))
        angle = (b['model']['yaw'] - a['model']['yaw'] + math.pi/4) % (math.pi/2) - math.pi/4
        yaw_step.append(abs(angle) * 180 / math.pi)
        yaw_rate.append(abs(angle) * 180 / math.pi / dt)
        center_step.append(np.linalg.norm(np.asarray(b['model']['center']) - a['model']['center']))
        if 'center_pixel' in a['model'] and 'center_pixel' in b['model']:
            center_pixel_step.append(np.linalg.norm(np.asarray(b['model']['center_pixel'])
                                                   - a['model']['center_pixel']))
        for stage, values in [('raw', raw_step), ('used', used_step)]:
            da, db = detection_points(a, stage), detection_points(b, stage)
            if not da or not db:
                continue
            anchor = max(da, key=area)
            current = min(db, key=lambda d: np.linalg.norm(d['center'] - anchor['center']))
            if np.linalg.norm(current['center'] - anchor['center']) < 30:
                values.append(corner_rms(current['corners'], anchor['corners']))
    result = dict(window_seconds=[start, stop], frames=len(rr), valid_model_frames=len(aa),
                  model_corner_step_px=summary(model_step), raw_NN_corner_step_px=summary(raw_step),
                  used_corner_step_px=summary(used_step), yaw_step_deg=summary(yaw_step),
                  yaw_rate_deg_per_s=summary(yaw_rate), center_step_m=summary(center_step),
                  center_pixel_step_px=summary(center_pixel_step))
    for stage, prefix in [('raw', 'raw_NN'), ('used', 'used_observation')]:
        residuals = [error for record in rr for error in observation_residuals(record, stage)]
        result[f'{prefix}_anchor_fit_rms_px'] = summary([e['rms'] for e in residuals if e['plate'] == 0])
        result[f'{prefix}_all_observed_fit_rms_px'] = summary([e['rms'] for e in residuals])
    support = [s for record in rr for s in support_residuals(record)]
    result['blue_bar_endpoint_fit_rms_px'] = summary([s['recomputed_rms_px'] for s in support])
    result['measured_image_motion_px'] = summary([r.get('motion_px', r['model'].get('motion_px')) for r in rr
                                                if r.get('model') is not None
                                                and isinstance(r.get('motion_px', r['model'].get('motion_px')), (int, float))])
    if aa:
        centers = np.asarray([r['model']['center'] for r, _ in aa])
        yaw = np.unwrap(np.asarray([r['model']['yaw'] for r, _ in aa]) * 4) / 4
        pixels = np.asarray([p for _, p in aa])
        result.update(yaw_std_deg=float(np.std(yaw) * 180/math.pi),
                      yaw_range_deg=float(np.ptp(yaw) * 180/math.pi),
                      center_range_m=np.ptp(centers, axis=0).tolist(),
                      model_corner_position_std_px=float(np.sqrt(np.mean(
                          np.var(pixels, axis=0).sum(axis=2)))))
    return result


def compare(paths):
    report = dict(notes=[
        'Source-video optical flow confirms stationary windows 0–8 s and 60–63 s; rotation begins about 8.9 s.',
        'Model corners are cyclically aligned over the four boards before step comparisons.',
        'Moving corner steps and Yaw rates describe actual motion plus estimation noise; they are not stationary jitter.',
        'NN and blue-bar endpoint residuals reuse observations fitted by the estimator and do not establish independent pose accuracy.',
        'Blue-bar RMS is independently recomputed from logged endpoints and final model projection; source extraction is not rerun.',
    ], outputs={})
    late_records = []
    for path in paths:
        records = load_records(path)
        output = {name: collect(records, start, stop) for name, (start, stop) in WINDOWS.items()}
        late = next((r for r in records if r['frame'] == 6000), None)
        late_records.append(late)
        if late is not None:
            supports = support_residuals(late)
            output['late_frame_6000'] = dict(time=late['time'], status=late.get('status'),
                raw_NN_fit_rms_px=observation_residuals(late, 'raw'),
                used_observation_fit_rms_px=observation_residuals(late, 'used'),
                blue_bar_endpoint_fit_rms_px=summary([s['recomputed_rms_px'] for s in supports]),
                blue_bar_support_details=supports)
        report['outputs'][str(Path(path).resolve())] = output
    if len(paths) == 2:
        baseline, fixed = (report['outputs'][str(Path(p).resolve())] for p in paths)
        report['stationary_improvement'] = {}
        for window in ('initial_stationary', 'final_stationary'):
            improvements = {}
            for metric in ('model_corner_step_px', 'yaw_step_deg', 'center_pixel_step_px'):
                before, after = baseline[window][metric], fixed[window][metric]
                if before is not None and after is not None and before['p95'] > 0:
                    improvements[metric + '_p95_reduction_percent'] = float(100 * (1 - after['p95']/before['p95']))
            before = baseline[window].get('model_corner_position_std_px')
            after = fixed[window].get('model_corner_position_std_px')
            if before is not None and after is not None and before > 0:
                improvements['model_corner_position_std_reduction_percent'] = float(100 * (1-after/before))
            report['stationary_improvement'][window] = improvements
        a, b = late_records
        if a is not None and b is not None and a.get('model') and b.get('model'):
            # Compare identical source endpoints, correcting quarter-turn IDs.
            shift = round((b['model']['yaw']-a['model']['yaw'])/(math.pi/2)) % 4
            comparisons = []
            for support in b['model'].get('side_support', []):
                index, edge, bar = support['plate'], support['edge'], np.asarray(support['bar'])
                old = np.asarray(a['model']['plates'][(index+shift)%4]['pixels'])[edge]
                new = np.asarray(b['model']['plates'][index]['pixels'])[edge]
                comparisons.append(dict(plate=index, observed_edge=edge,
                                        before_endpoint_rms_px=corner_rms(old, bar),
                                        after_endpoint_rms_px=corner_rms(new, bar)))
            report['late_frame_same_LED_comparison'] = dict(frame=6000, comparisons=comparisons,
                limitation='The same image-derived endpoints are evaluated on both models; they constrain the new fit and are not independent pose truth.')
    return report


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', type=Path, nargs='*', help='One or two frames.jsonl paths; defaults to output and output_fixed')
    parser.add_argument('--output', type=Path, help='Save report JSON in addition to stdout')
    args = parser.parse_args()
    paths = args.paths or [here / 'output/frames.jsonl', here / 'output_fixed/frames.jsonl']
    if not 1 <= len(paths) <= 2:
        parser.error('Provide one baseline path or two baseline/fixed paths')
    report = compare(paths)
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end='')


if __name__ == '__main__':
    main()
