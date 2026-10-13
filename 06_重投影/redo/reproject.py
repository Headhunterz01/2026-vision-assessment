#!/usr/bin/env python3
"""YOLO pose -> planar PnP -> four-armor rigid model -> image reprojection.

Independent implementation; no dependency on the earlier armor_ekf pipeline.
All lengths are metres, angles radians. Armor local X points outward, Y left,
Z up; the model's keypoint order is TL, BL, BR, TR.
"""
import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from image_evidence import extract_lightbars, refine_corners, refine_fit_with_lightbars
from temporal import CornerSmoother, StateSmoother, VisualMotion

ROOT = Path(__file__).resolve().parents[2]
K = np.array([[3571.1890927935297, 0, 683.5999418028833],
              [0, 3569.0234714126823, 565.5838917436471], [0, 0, 1]])
D = np.array([-.029757126887136703, .371503828585697,
              .0011898367108863406, -.0025516396408119418, 0.])
OBJECT = np.array([[0, -.0675, .0275], [0, -.0675, -.0275],
                   [0, .0675, -.0275], [0, .0675, .0275]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


@dataclass
class Geometry:
    gimbal_pitch: float = 0.
    radius: float = .25
    pitch: float = math.pi / 12

    def __post_init__(self):
        if not np.isfinite([self.gimbal_pitch, self.radius, self.pitch]).all() or self.radius <= 0:
            raise ValueError('Geometry must be finite and radius positive')
        self.world_from_camera = ry(self.gimbal_pitch + .091) @ np.array(
            [[0, 0, 1.], [-1., 0, 0], [0, -1., 0]])
        self.camera_origin = ry(self.gimbal_pitch) @ np.array([.0461892, 0, -.0603874])

    def plates(self, state):
        """state=(center_x,center_y,center_z,outward_yaw_of_anchor)."""
        center, yaw = np.asarray(state[:3]), state[3]
        result = []
        for i in range(4):
            angle = yaw + i * math.pi / 2
            position = center + self.radius * np.array([np.cos(angle), np.sin(angle), 0])
            # PDF X points inward. Rz(pdf_yaw) Ry(+15deg) Rz(pi)
            # equals Rz(outward_yaw) Ry(-15deg) for Roll=0.
            rotation = rz(angle) @ ry(-self.pitch)
            camera_rotation = self.world_from_camera.T @ rotation
            camera_position = self.world_from_camera.T @ (position - self.camera_origin)
            points = OBJECT @ camera_rotation.T + camera_position
            positive = bool(np.all(points[:, 2] > .01))
            pixels = None
            if positive:
                pixels = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, D)[0].reshape(4, 2)
            result.append(dict(index=i, yaw=wrap(angle), position=position, rotation=rotation,
                               camera_position=camera_position, camera_rotation=camera_rotation,
                               camera_points=points, pixels=pixels, positive=positive,
                               front=bool(camera_rotation[:, 0] @ camera_position < 0)))
        return result

    def pnp_seeds(self, corners):
        """Return both physically visible IPPE solutions, never just the first."""
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(OBJECT, np.asarray(corners, np.float64), K, D,
                                               flags=cv2.SOLVEPNP_IPPE)
        seeds = []
        for rvec, tvec in zip(rvecs, tvecs):
            rotation = cv2.Rodrigues(rvec)[0]
            t = tvec.reshape(3)
            if rotation[:, 0] @ t >= 0 or np.min((OBJECT @ rotation.T + t)[:, 2]) <= .01:
                continue
            world_rotation = self.world_from_camera @ rotation
            yaw = np.arctan2(world_rotation[1, 0], world_rotation[0, 0])
            p = self.world_from_camera @ t + self.camera_origin
            center = p - self.radius * np.array([np.cos(yaw), np.sin(yaw), 0])
            seeds.append(np.r_[center, yaw])
        return seeds

    def fit(self, seed, observations, max_rms=3.):
        """Fit only center XYZ and yaw; pitch/roll/radius remain the given model.

        observations contains (plate_index, detected_corners). A second detected
        plate only resolves/refines pose; all four plate positions are generated
        from the fitted anchor using the same rigid model.
        """
        def residual(state):
            plates = self.plates(state)
            values = []
            for index, corners in observations:
                plate = plates[index]
                if not plate['positive'] or not np.isfinite(plate['pixels']).all():
                    return np.full(8 * len(observations), 1e6)
                values.extend((plate['pixels'] - corners).ravel())
            return np.asarray(values)
        fit = least_squares(residual, seed, method='lm', max_nfev=35,
                            ftol=1e-6, xtol=1e-6, gtol=1e-6)
        state = fit.x
        errors = residual(state).reshape(-1, 4, 2)
        rms = np.sqrt(np.mean(np.sum(errors ** 2, axis=2), axis=1))
        if not np.isfinite(state).all() or np.max(rms) > max_rms:
            return None
        state[3] = wrap(state[3])
        return dict(state=state, plates=self.plates(state), errors=rms,
                    observations=observations)


def valid_corners(points):
    points = np.asarray(points, np.float32)
    return (np.isfinite(points).all() and cv2.isContourConvex(points)
            and abs(cv2.contourArea(points)) >= 30
            and np.min(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)) >= 4)


class Detector:
    def __init__(self, model, confidence=.5, threads=2):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0]
        _, channels, self.height, self.width = self.input.shape
        if channels != 3 or not isinstance(self.height, int) or not isinstance(self.width, int):
            raise ValueError('Expected fixed model input [1,3,H,W]')
        if self.session.get_outputs()[0].shape[-1] != 14:
            raise ValueError('Expected YOLO26 end-to-end output [1,N,14]: xyxy,score,class,4xy')
        self.confidence = confidence

    def __call__(self, image):
        h, w = image.shape[:2]
        scale = min(self.width / w, self.height / h)
        nw, nh = int(w * scale + .5), int(h * scale + .5)
        left, top = (self.width - nw) // 2, (self.height - nh) // 2
        padded = np.full((self.height, self.width, 3), 114, np.uint8)
        padded[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        rows = self.session.run(None, {self.input.name: tensor})[0][0]
        detections = []
        for row in rows:
            if not np.isfinite(row).all() or not self.confidence <= row[4] <= 1 or not 0 <= row[5] <= 12:
                continue
            corners = (row[6:].reshape(4, 2) - [left, top]) / scale
            box = (row[:4] - [left, top, left, top]) / scale
            if not valid_corners(corners) or np.any(corners < 0) or np.any(corners >= [w, h]):
                continue
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append(dict(corners=corners, center=corners.mean(0),
                                   box=box, confidence=float(row[4]), class_id=int(row[5])))
        unique = []
        for detection in sorted(detections, key=lambda d: -d['confidence']):
            if any(np.linalg.norm(detection['center'] - d['center']) < 8 for d in unique):
                continue
            unique.append(detection)
        return unique


class DetectionCache:
    """Replay recorded NN output only; geometry is always recomputed."""
    def __init__(self, path, model, video, confidence):
        summary = json.loads((path.parent / 'summary.json').read_text())
        if Path(summary['source']).resolve() != video.resolve() or Path(summary['model']).resolve() != model.resolve():
            raise ValueError('Detection cache model/video differs from the requested input')
        if confidence != summary.get('confidence', .5):
            raise ValueError('Detection cache confidence differs from the requested threshold')
        self.stream = path.open()
        self.row = None

    def frame(self, index):
        while self.row is None or self.row['frame'] < index:
            line = self.stream.readline()
            if not line:
                raise ValueError(f'Detection cache has no source frame {index}')
            self.row = json.loads(line)
        if self.row['frame'] != index:
            raise ValueError(f'Detection cache has no source frame {index}; check stride')
        output = []
        for d in self.row.get('raw_detections', self.row['detections']):
            item = dict(d)
            for key in ['corners', 'center', 'box']:
                item[key] = np.asarray(item[key], np.float64)
            output.append(item)
        return output

    def close(self):
        self.stream.close()


def match_others(plates, detections, anchor):
    """One-to-one matching to visible model plates; class labels are not IDs."""
    pairs = []
    for j, d in enumerate(detections):
        if j == anchor:
            continue
        for p in plates[1:]:
            if not p['positive'] or not p['front']:
                continue
            distance = np.linalg.norm(p['pixels'].mean(0) - d['center'])
            corner_error = np.sqrt(np.mean(np.sum((p['pixels'] - d['corners']) ** 2, axis=1)))
            if distance < 45 and corner_error < 40:
                pairs.append((corner_error, j, p['index']))
    used_d, used_p, matches = set(), set(), []
    for error, j, i in sorted(pairs):
        if j not in used_d and i not in used_p:
            matches.append((i, j, error))
            used_d.add(j)
            used_p.add(i)
    return matches


class Estimator:
    def __init__(self, geometry):
        self.geometry = geometry
        self.last = None
        self.last_time = -1.
        self.motion = VisualMotion()
        self.corner_smoother = CornerSmoother()
        self.state_smoother = StateSmoother()
        self.processed_detections = []
        self.motion_px = None

    def estimate(self, detections, stamp, image=None):
        if stamp - self.last_time > .3 or stamp <= self.last_time:
            self.last = None
            self.state_smoother.reset()
        self.motion_px = self.motion.update(image, detections, stamp)
        bars, used_bars = [], set()
        if image is not None:
            bars = extract_lightbars(image, detections)
            detections, used_bars = refine_corners(detections, bars)
            detections = self.corner_smoother.update(detections, stamp, self.motion_px)
        self.processed_detections = detections
        if not detections:
            return None, 'NO DETECTION'
        candidates = []
        # Prefer the most frontal/large observed board; other detections support
        # its pose. Consider another anchor if the first has no valid fit.
        order = sorted(range(len(detections)), key=lambda j: -abs(cv2.contourArea(
            detections[j]['corners'].astype(np.float32))))
        for anchor in order:
            d = detections[anchor]
            seen = []
            for seed in self.geometry.pnp_seeds(d['corners']):
                fit = self.geometry.fit(seed, [(0, d['corners'])],
                                        max_rms=8. if len(detections) > 1 else 3.)
                if fit is None:
                    continue
                if any(np.linalg.norm(fit['state'][:3] - s[:3]) < 1e-4
                       and abs(wrap(fit['state'][3] - s[3])) < 1e-3 for s in seen):
                    continue
                seen.append(fit['state'].copy())
                if fit['plates'][0]['front'] is False:
                    continue
                matches = match_others(fit['plates'], detections, anchor)
                if matches:
                    observations = [(0, d['corners'])] + [(i, detections[j]['corners']) for i, j, _ in matches]
                    # Fixed equal heights and the given tilt are approximate
                    # for the filmed chassis. A joint fit allows up to 8px RMS
                    # per board; the single-board gate remains 3px.
                    refined = self.geometry.fit(fit['state'], observations, max_rms=8.)
                    if refined is not None:
                        fit = refined
                    else:
                        # Geometrically inconsistent detections must not be
                        # silently accepted as support for this candidate.
                        continue
                elif len(detections) > 1:
                    continue
                fit['anchor'] = anchor
                fit['matches'] = matches
                if image is not None:
                    fit = refine_fit_with_lightbars(self.geometry, fit, detections, bars, used_bars)
                score = float(np.mean(fit['errors']))
                if self.last is not None:
                    # Robot yaw is only defined modulo pi/2 when anchor changes.
                    delta = abs((fit['state'][3] - self.last[3] + math.pi / 4) % (math.pi / 2) - math.pi / 4)
                    center_delta = np.linalg.norm(fit['state'][:3] - self.last[:3])
                    score += min(center_delta, 1.) * 2 + min(delta, .8) * .3
                fit['score'] = score - .15 * len(matches)
                candidates.append(fit)
            if candidates:
                break
        if not candidates:
            return None, 'POSE / GEOMETRY REJECTED'
        candidates.sort(key=lambda c: c['score'])
        best = candidates[0]
        # A close mirror fit with a remote center and no supporting board is
        # unobservable on this frame. Suppress it instead of inventing a pose.
        if len(candidates) > 1 and not best['matches'] and self.last is None:
            second = candidates[1]
            separation = np.linalg.norm(best['state'][:3] - second['state'][:3])
            if separation > .12 and second['score'] - best['score'] < .25:
                return None, 'AMBIGUOUS SINGLE PLATE'
        best['raw_state'], best['raw_rms'] = best['state'].copy(), best['errors'].copy()
        best['state'], best['temporal_alpha'] = self.state_smoother.update(best['state'], stamp, self.motion_px)
        best['plates'] = self.geometry.plates(best['state'])
        best['errors'] = np.array([np.sqrt(np.mean(np.sum((best['plates'][i]['pixels'] - corners)**2, axis=1)))
                                  for i, corners in best['observations']])
        for support in best.get('side_support', []):
            pixels = best['plates'][support['plate']]['pixels'][support['edge']]
            support['rms'] = float(np.sqrt(np.mean(np.sum((pixels-support['bar'])**2, axis=1))))
        self.last = best['state'].copy()
        self.last_time = stamp
        camera_center = self.geometry.world_from_camera.T @ (best['state'][:3] - self.geometry.camera_origin)
        if camera_center[2] > .01:
            best['center_pixel'] = cv2.projectPoints(camera_center[None], np.zeros(3), np.zeros(3), K, D)[0].reshape(2)
        return best, 'MULTI PLATE' if best['matches'] else 'SINGLE PLATE'


def draw(image, detections, fit, status, debug=False):
    out = image.copy()
    cyan, green = (255, 255, 0), (80, 255, 160)
    if fit is not None:
        for p in fit['plates']:
            if not p['positive'] or not np.isfinite(p['pixels']).all():
                continue
            if np.max(np.abs(p['pixels'])) > 1e6:
                continue
            points = np.rint(p['pixels']).astype(np.int32)
            cv2.polylines(out, [points], True, cyan, 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(out, tuple(point), 3, cyan, -1, cv2.LINE_AA)
            if debug:
                cv2.putText(out, f"P{p['index']}" + (' back' if not p['front'] else ''),
                            tuple(points[0] + [0, -8]), cv2.FONT_HERSHEY_SIMPLEX, .45, cyan, 1, cv2.LINE_AA)
        if 'center_pixel' in fit:
            x, y = fit['center_pixel']
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                cv2.drawMarker(out, (int(round(x)), int(round(y))), (0, 0, 255),
                               cv2.MARKER_DIAMOND, 11, 2, cv2.LINE_AA)
    for d in detections:
        points = np.rint(d['corners']).astype(np.int32)
        cv2.polylines(out, [points], True, green, 1, cv2.LINE_AA)
        cv2.line(out, tuple(points[0]), tuple(points[2]), green, 1, cv2.LINE_AA)
        cv2.line(out, tuple(points[1]), tuple(points[3]), green, 1, cv2.LINE_AA)
        for p in points:
            cv2.circle(out, tuple(p), 3, (0, 255, 255), -1, cv2.LINE_AA)
    if debug:
        cv2.putText(out, f'{status} | detected {len(detections)} | cyan: model / green: refined NN',
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, cyan, 1, cv2.LINE_AA)
    return out


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


class VideoOutput:
    """Encode browser-compatible H.264 via the bundled imageio ffmpeg binary."""
    def __init__(self, path, size, fps):
        import imageio_ffmpeg
        self.generator = imageio_ffmpeg.write_frames(str(path), size, fps=fps,
            codec='libx264', pix_fmt_in='bgr24', pix_fmt_out='yuv420p',
            quality=7, macro_block_size=2, ffmpeg_log_level='error',
            output_params=['-preset', 'veryfast', '-movflags', '+faststart'])
        self.generator.send(None)

    def write(self, image):
        self.generator.send(np.ascontiguousarray(image))

    def close(self):
        self.generator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / '01_神经网络/yolo_pose/final_model/best.onnx')
    parser.add_argument('--video', type=Path, default=ROOT / '00_题目原件/videos/test.mp4')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'output_fixed')
    parser.add_argument('--profile', choices=['video', 'pdf'], default='video',
                        help='video: gimbal pitch 0rad diagnostic; pdf: exact 1.34rad extrinsics')
    parser.add_argument('--gimbal-pitch-rad', type=float)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=.5)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--detections-cache', type=Path,
                        help='Recorded frames.jsonl from the same NN model/video; re-solve all geometry')
    args = parser.parse_args()
    if args.stride < 1 or args.max_frames < 0 or args.threads < 1 or not 0 < args.confidence < 1:
        parser.error('Invalid stride, max-frames, threads or confidence')
    gimbal = args.gimbal_pitch_rad if args.gimbal_pitch_rad is not None else (0. if args.profile == 'video' else 1.34)
    geometry = Geometry(gimbal)
    detector = None if args.detections_cache else Detector(args.model, args.confidence, args.threads)
    detection_cache = DetectionCache(args.detections_cache, args.model, args.video, args.confidence) if args.detections_cache else None
    estimator = Estimator(geometry)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    w, h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != (1440, 1080):
        raise ValueError('Calibration is for 1440x1080; supply matching video/calibration')
    if not fps > 0:
        raise ValueError('Source video has no valid frame rate')
    args.output.mkdir(parents=True, exist_ok=True)
    video = VideoOutput(args.output / 'reprojection.mp4', (w, h), fps / args.stride)
    count = valid = multi = 0
    reasons, errors = {}, []
    started = time.monotonic()
    snapshot_frames = {0, 200, 450, 906, 978, 2000, 4000, 6000}
    csv_fields = ['frame', 'time', 'valid', 'plate', 'anchor_detection', 'center_x', 'center_y',
                  'center_z', 'yaw', 'world_x', 'world_y', 'world_z', 'front_facing'] + [
                  f'k{i}{axis}' for i in range(4) for axis in 'xy']
    try:
        with (args.output / 'frames.jsonl').open('w') as log, (args.output / 'poses.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, csv_fields)
            writer.writeheader()
            frame = -1
            while not args.max_frames or count < args.max_frames:
                if not capture.grab():
                    break
                frame += 1
                if frame % args.stride:
                    continue
                ok, image = capture.retrieve()
                if not ok:
                    raise RuntimeError(f'Failed decoding source frame {frame}')
                stamp = frame / fps
                raw_detections = detection_cache.frame(frame) if detection_cache else detector(image)
                fit, status = estimator.estimate(raw_detections, stamp, image)
                detections = estimator.processed_detections
                reasons[status] = reasons.get(status, 0) + 1
                data = dict(frame=frame, time=stamp, status=status, detections=detections,
                            raw_detections=raw_detections, motion_px=estimator.motion_px, model=None)
                if fit is not None:
                    valid += 1
                    multi += bool(fit['matches'])
                    errors.append(float(fit['errors'][0]))
                    state = fit['state']
                    data['model'] = dict(center=state[:3], yaw=state[3], anchor=fit['anchor'],
                                         rms=fit['errors'], matches=fit['matches'], plates=fit['plates'])
                    data['model'].update(side_support=fit.get('side_support', []),
                                         raw_state=fit['raw_state'], raw_rms=fit['raw_rms'],
                                         temporal_alpha=fit['temporal_alpha'])
                    if 'center_pixel' in fit:
                        data['model']['center_pixel'] = fit['center_pixel']
                    for p in fit['plates']:
                        row = dict(frame=frame, time=stamp, valid=1, plate=p['index'],
                                   anchor_detection=fit['anchor'], yaw=p['yaw'], front_facing=int(p['front']))
                        row.update(zip(['center_x', 'center_y', 'center_z'], state[:3]))
                        row.update(zip(['world_x', 'world_y', 'world_z'], p['position']))
                        if p['pixels'] is not None:
                            row.update(zip([f'k{i}{axis}' for i in range(4) for axis in 'xy'], p['pixels'].ravel()))
                        writer.writerow(row)
                else:
                    writer.writerow(dict(frame=frame, time=stamp, valid=0, plate=-1))
                annotated = draw(image, detections, fit, status, args.debug)
                video.write(annotated)
                log.write(json.dumps(data, default=serializable, allow_nan=False) + '\n')
                if frame in snapshot_frames:
                    cv2.imwrite(str(args.output / f'frame_{frame:04d}.png'), annotated)
                count += 1
                if count % 250 == 0:
                    print(f'frames={count} model={valid} speed={count / (time.monotonic()-started):.1f}fps', flush=True)
    finally:
        video.close()
        capture.release()
        if detection_cache:
            detection_cache.close()
    if not count:
        raise RuntimeError('No video frames were processed')
    report = dict(source=str(args.video.resolve()), model=str(args.model.resolve()),
                  profile=args.profile, gimbal_pitch_rad=gimbal, radius_m=.25,
                  inward_pitch_deg=15, roll_deg=0, width_m=.135, height_m=.055,
                  source_fps=fps, output_fps=fps/args.stride, stride=args.stride,
                  confidence=args.confidence,
                  inference_source='recorded NN detections' if detection_cache else 'direct ONNX inference',
                  frames=count, valid_model_frames=valid, multi_plate_frames=multi,
                  status_counts=reasons, processing_seconds=time.monotonic()-started,
                  anchor_rms_median_px=float(np.median(errors)) if errors else None,
                  anchor_rms_p95_px=float(np.percentile(errors, 95)) if errors else None,
                  calibration_note='video profile uses uncalibrated 0rad gimbal pitch for this test video; pdf uses 1.34rad',
                  visibility_note='All 4 model boards are drawn, including occluded/back-facing boards; no physical pose ground truth.')
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
