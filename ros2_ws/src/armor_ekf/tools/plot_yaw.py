"""Plot measured and filtered yaw in the declared frame; do not claim accuracy."""
import argparse
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

cli = argparse.ArgumentParser()
cli.add_argument('output_dir')
args = cli.parse_args()
folder = Path(args.output_dir)
with (folder/'yaw.csv').open() as stream:
    yaw_frame = next(csv.DictReader(stream)).get('frame_id', 'camera_optical_frame')
data = np.genfromtxt(folder/'yaw.csv', delimiter=',', names=True)
valid = data['detection_valid'].astype(bool)
initialized = data['filter_initialized'].astype(bool)
same_track = np.diff(data['track_id']) == 0
pairs = valid[1:] & valid[:-1] & same_track


def angle_diff(a):
    delta = np.diff(a)
    return np.arctan2(np.sin(delta), np.cos(delta))


raw_steps = angle_diff(data['observed_yaw'])[pairs]
filtered_steps = angle_diff(data['filtered_yaw'])[pairs]
metrics = {
    'yaw_frame': yaw_frame,
    'frames': len(data), 'selected_pose_frames': int(valid.sum()),
    'selected_target_observation_fraction': float(valid.mean()),
    'measurements_used': int(data['measurement_used'].sum()),
    'measurement_fraction_when_detected': float(data['measurement_used'].sum()/max(1,valid.sum())),
    'track_resets': int(data['target_reset'].sum()),
    'visual_track_switches': int(data['target_switched'].sum()) if 'target_switched' in data.dtype.names else None,
    'median_reprojection_error_px': float(np.median(data['reprojection_error'][valid])),
    'median_abs_raw_step_rad': float(np.median(np.abs(raw_steps))) if len(raw_steps) else None,
    'median_abs_filtered_step_rad': float(np.median(np.abs(filtered_steps))) if len(filtered_steps) else None,
    'note': 'No ground truth. Step metrics include actual motion and do not measure yaw accuracy.',
}
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout='constrained')
axes[0].scatter(data['time'][valid], np.rad2deg(data['observed_yaw'][valid]), s=3, alpha=.45, label='PnP observation')
# Break lines at track resets and angle wraps.
filtered = np.rad2deg(data['filtered_yaw']).copy()
filtered[~initialized | data['target_reset'].astype(bool)] = np.nan
filtered[1:][np.abs(np.diff(data['filtered_yaw'])) > np.pi] = np.nan
axes[0].plot(data['time'], filtered, lw=1, label='EKF')
axes[0].set_ylabel(f'{yaw_frame} yaw (deg)'); axes[0].legend(loc='upper right')
axes[0].set_title(f'YOLO Pose -> PnP -> {yaw_frame} -> EKF, test.mp4 (no ground-truth yaw)')
rates = data['yaw_rate'].copy()
rates[data['target_reset'].astype(bool)] = np.nan
axes[1].plot(data['time'], rates, lw=.8)
axes[1].set_ylabel('Yaw rate (rad/s)')
axes[2].plot(data['time'], data['confidence'], label='Detection confidence', lw=.8)
axes[2].scatter(data['time'][~valid], np.zeros((~valid).sum()), s=4, color='red', label='No selected armor observation')
axes[2].set_ylabel('Confidence'); axes[2].set_xlabel('Source video time (s)'); axes[2].legend()
for ax in axes:
    ax.grid(True, alpha=.2)
fig.savefig(folder/'yaw_plot.png', dpi=160)
(folder/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
print(json.dumps(metrics, indent=2))
