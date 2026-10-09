"""Independently validate all replayed world poses against the prepared matrix."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def rotation(row, prefix=''):
    x, y, z, w = (float(row[prefix+k]) for k in ('qx','qy','qz','qw'))
    assert abs(x*x+y*y+z*z+w*w-1) < 1e-8
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--world-dir', required=True)
    cli.add_argument('--camera-dir', required=True)
    cli.add_argument('--old-camera-dir', required=True)
    cli.add_argument('--direct-world-dir')
    args = cli.parse_args()
    folder = Path(args.world_dir)
    baseline = Path(__file__).resolve().parents[3]/'preparation/coordinate_transform/geometry_baseline.json'
    transform = np.array(json.loads(baseline.read_text())['T_odom_camera_optical_frame'])
    rows = read(folder/'yaw.csv')
    position_error = rotation_error = yaw_error = 0.0
    low_horizontal = 0
    for row in rows:
        assert row['frame_id'] == 'odom'
        if row['detection_valid'] == '1':
            camera = np.array([float(row['camera_'+k]) for k in ('x','y','z')])
            position = np.array([float(row[k]) for k in ('x','y','z')])
            expected_position = transform[:3,:3] @ camera + transform[:3,3]
            expected_rotation = transform[:3,:3] @ rotation(row,'camera_')
            actual_rotation = rotation(row)
            position_error = max(position_error,float(np.max(np.abs(position-expected_position))))
            rotation_error = max(rotation_error,float(np.max(np.abs(actual_rotation-expected_rotation))))
            expected_yaw = math.atan2(expected_rotation[1,0],expected_rotation[0,0])
            delta = float(row['observed_yaw'])-expected_yaw
            yaw_error = max(yaw_error,abs(math.atan2(math.sin(delta),math.cos(delta))))
            horizontal = np.hypot(expected_rotation[0,0],expected_rotation[1,0])
            assert horizontal >= 0.001
            low_horizontal += horizontal < 0.2
        else:
            assert math.isnan(float(row['observed_yaw']))
        if row['target_reset'] == '1':
            assert row['measurement_used'] == '1'
            assert abs(float(row['filtered_yaw'])-float(row['observed_yaw'])) < 1e-10
    assert max(position_error,rotation_error,yaw_error) < 1e-8
    camera = read(Path(args.camera_dir)/'yaw.csv')
    old = read(Path(args.old_camera_dir)/'yaw.csv')
    assert len(old) == len(camera) == len(rows)
    checked = ['frame','time','detection_valid','measurement_used','filter_initialized','target_reset',
               'track_id','class_id','observed_yaw','filtered_yaw','yaw_rate','yaw_variance','x','y','z',
               'reprojection_error','visual_track_id','target_switched','selected_detection_index']
    for a,b in zip(camera,old):
        for key in checked:
            x,y = float(a[key]),float(b[key])
            assert (math.isnan(x) and math.isnan(y)) or abs(x-y) < 1e-9, f'Camera regression: {key}'
    selection_changes = sum(a['selected_detection_index'] != b['selected_detection_index'] for a,b in zip(rows,camera))
    direct = None
    if args.direct_world_dir:
        direct_rows = read(Path(args.direct_world_dir)/'yaw.csv')
        fields = checked + ['camera_yaw','camera_x','camera_y','camera_z','qx','qy','qz','qw']
        assert len(direct_rows) <= len(rows)
        for a,b in zip(direct_rows,rows):
            assert a['frame_id'] == b['frame_id']
            for key in fields:
                x,y = float(a[key]),float(b[key])
                assert (math.isnan(x) and math.isnan(y)) or abs(x-y) < 1e-9, f'Direct/replay mismatch: {key}'
        direct = len(direct_rows)
    report = dict(status='PASS', frames=len(rows),
                  valid_world_poses=sum(r['detection_valid']=='1' for r in rows),
                  max_position_component_error_m=position_error,
                  max_rotation_component_error=rotation_error,max_yaw_error_rad=yaw_error,
                  camera_regression_frames=len(camera),camera_regression_checked_fields=checked,
                  selected_detection_changes_from_camera_mode=selection_changes,
                  direct_inference_replay_agreement_frames=direct,
                  world_observations_horizontal_normal_below_0_2=int(low_horizontal),
                  note='Transform consistency checks; video has no world yaw ground truth. Near-vertical normals have poorly conditioned yaw.')
    (folder/'replay_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
