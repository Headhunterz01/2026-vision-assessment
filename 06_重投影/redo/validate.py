"""Independent Brown-Conrady projection and recorded-output validation."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    summary = json.loads((args.output / 'summary.json').read_text())
    radius = summary['radius_m']
    gimbal = summary['gimbal_pitch_rad']
    angle = gimbal + .091
    c, s = np.cos(angle), np.sin(angle)
    # Explicit transform, independent of Geometry.plates/PnP.
    world_from_camera = np.array([[0, -s, c], [-1, 0, 0], [0, -c, -s]])
    cg, sg = np.cos(gimbal), np.sin(gimbal)
    origin = np.array([cg * .0461892 - sg * .0603874, 0,
                       -sg * .0461892 - cg * .0603874])
    object_points = np.array([[0, -.0675, .0275], [0, -.0675, -.0275],
                              [0, .0675, -.0275], [0, .0675, .0275]])
    max_projection = max_transform = max_radius = 0.
    count = valid = 0
    center_errors, corner_errors, anchor_errors = [], [], []
    previous_frame = -1
    with (args.output / 'frames.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row['frame'] == previous_frame + summary['stride'] if count else row['frame'] == 0
            previous_frame = row['frame']
            count += 1
            model = row['model']
            if model is None:
                continue
            valid += 1
            assert len(model['plates']) == 4
            center = np.array(model['center'])
            for i, p in enumerate(model['plates']):
                position = np.array(p['position'])
                rotation = np.array(p['rotation'])
                radial = position - center
                max_radius = max(max_radius, abs(np.linalg.norm(radial) - radius))
                assert abs(radial[2]) < 1e-12
                theta = model['yaw'] + i * np.pi / 2
                expected = center + radius * np.array([np.cos(theta), np.sin(theta), 0])
                np.testing.assert_allclose(position, expected, atol=1e-12)
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                assert abs(rotation[2, 0] - np.sin(np.pi / 12)) < 1e-12
                assert radial[:2] @ rotation[:2, 0] > 0
                camera_points = (object_points @ rotation.T + position - origin) @ world_from_camera
                max_transform = max(max_transform, float(np.max(np.abs(camera_points - p['camera_points']))))
                if not p['positive']:
                    assert p['pixels'] is None
                    continue
                x, y = camera_points[:, 0] / camera_points[:, 2], camera_points[:, 1] / camera_points[:, 2]
                r2 = x*x + y*y
                radial_scale = 1 - .029757126887136703*r2 + .371503828585697*r2*r2
                xd = x*radial_scale + 2*.0011898367108863406*x*y - .0025516396408119418*(r2 + 2*x*x)
                yd = y*radial_scale + .0011898367108863406*(r2 + 2*y*y) - 2*.0025516396408119418*x*y
                pixels = np.column_stack((3571.1890927935297*xd + 683.5999418028833,
                                          3569.0234714126823*yd + 565.5838917436471))
                max_projection = max(max_projection, float(np.max(np.abs(pixels - p['pixels']))))
            observations = [(0, model['anchor'])] + [(i, j) for i, j, _ in model['matches']]
            assert len({j for _, j in observations}) == len(observations)
            assert len({i for i, _ in observations}) == len(observations)
            for n, (i, j) in enumerate(observations):
                predicted = np.array(model['plates'][i]['pixels'])
                observed = np.array(row['detections'][j]['corners'])
                rms = float(np.sqrt(np.mean(np.sum((predicted - observed)**2, axis=1))))
                assert abs(rms - model['rms'][n]) < 1e-8
                assert rms <= (8 if model['matches'] else 3) + 1e-8
                if i == 0:
                    anchor_errors.append(rms)
                else:
                    center_errors.append(float(np.linalg.norm(predicted.mean(0) - observed.mean(0))))
                    corner_errors.append(rms)
    assert count == summary['frames'] and valid == summary['valid_model_frames']
    assert max_projection < 1e-7 and max_radius < 1e-12 and max_transform < 1e-12
    capture = cv2.VideoCapture(str(args.output / 'reprojection.mp4'))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == count
    assert abs(capture.get(cv2.CAP_PROP_FPS) - summary['output_fps']) < .01
    for frame in [0, count // 2, count - 1]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        assert ok and image.shape[:2] == (1080, 1440)
    capture.release()
    report = dict(status='PASS' if valid else 'NO_VALID_MODEL', frames=count, valid_models=valid,
                  max_independent_projection_error_px=max_projection,
                  max_world_camera_transform_error=max_transform, max_radius_error_m=max_radius,
                  matched_other_boards=len(center_errors),
                  other_board_center_median_px=float(np.median(center_errors)) if center_errors else None,
                  other_board_corner_rms_p95_px=float(np.percentile(corner_errors, 95)) if corner_errors else None,
                  anchor_rms_median_px=float(np.median(anchor_errors)) if anchor_errors else None,
                  limitation='Same-frame NN detections used for pose fitting; errors are fit residuals, not independent physical accuracy.')
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
