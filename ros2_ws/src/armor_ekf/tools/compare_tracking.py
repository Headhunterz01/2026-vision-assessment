"""Compare tracking on identical frames; no yaw/identity ground truth is implied."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def summary(rows):
    count = lambda key: sum(int(row.get(key, 0)) for row in rows)
    n = len(rows)
    observed = count('detection_valid')
    raw_steps, filtered_steps = [], []
    for a, b in zip(rows, rows[1:]):
        if a['detection_valid'] == b['detection_valid'] == '1' and a['track_id'] == b['track_id']:
            for key, values in [('observed_yaw', raw_steps), ('filtered_yaw', filtered_steps)]:
                delta = float(b[key])-float(a[key])
                values.append(abs(math.atan2(math.sin(delta), math.cos(delta))))
    errors = [float(r['reprojection_error']) for r in rows if r['detection_valid'] == '1']
    switches = count('target_switched')
    return {
        'frames': n, 'observed_frames': observed, 'observation_coverage': observed/n,
        'used_measurements': count('measurement_used'), 'update_fraction_when_observed': count('measurement_used')/observed,
        'ekf_rejection_events': count('ekf_rejected'), 'filter_resets': count('target_reset'),
        'explicit_visual_switches': switches, 'other_resets': count('target_reset')-switches,
        'median_reprojection_error_px': statistics.median(errors),
        'median_consecutive_raw_step_rad': statistics.median(raw_steps),
        'p95_consecutive_raw_step_rad': statistics.quantiles(raw_steps,n=20)[18],
        'median_consecutive_filtered_step_rad': statistics.median(filtered_steps),
        'p95_consecutive_filtered_step_rad': statistics.quantiles(filtered_steps,n=20)[18],
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('legacy_csv',type=Path)
    cli.add_argument('motion_csv',type=Path)
    cli.add_argument('output_dir',type=Path)
    args = cli.parse_args()
    a = list(csv.DictReader(args.legacy_csv.open()))
    b = list(csv.DictReader(args.motion_csv.open()))
    if not a or len(a) != len(b):
        raise ValueError('Comparison requires identical non-empty frame sequences')
    for old,new in zip(a,b):
        if old['frame'] != new['frame'] or abs(float(old['time'])-float(new['time'])) > 1e-8:
            raise ValueError('Input frame sequences differ')
    report = {'legacy':summary(a),'motion':summary(b),
              'limitations':'Same cached post-deduplication detector inputs, no yaw or permanent physical-identity ground truth. Camera yaw and angle steps include actual rotation. Switches split EKF segments; a lower reset count is not an identity-accuracy metric.'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'tracking_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    text = '# Tracking comparison\n\nBoth runs use identical detector outputs and source timestamps.\n\n'
    text += '| Metric | Legacy | Motion |\n|---|---:|---:|\n'
    labels = {'observed_frames':'Frames with selected armor pose','observation_coverage':'Selected-pose coverage',
              'used_measurements':'EKF measurement updates','update_fraction_when_observed':'Updates / selected poses',
              'ekf_rejection_events':'EKF rejection events','filter_resets':'Filter segment resets',
              'explicit_visual_switches':'Explicit visual-track switches','median_reprojection_error_px':'Median reprojection error (px)'}
    for key,label in labels.items():
        vals = [report[mode][key] for mode in ['legacy','motion']]
        if key in ['observation_coverage','update_fraction_when_observed']:
            rendered = [f'{v:.2%}' for v in vals]
        else: rendered = [f'{v:.4g}' for v in vals]
        text += f'| {label} | {rendered[0]} | {rendered[1]} |\n'
    text += '\n'+report['limitations']+'\n'
    (args.output_dir/'TRACKING_COMPARISON.md').write_text(text)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
