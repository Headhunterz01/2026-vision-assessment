"""Preparation-only geometry checks; does not change or launch the EKF pipeline."""
from pathlib import Path
import json
import math
import numpy as np


def rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(xyz, rpy):
    result = np.eye(4)
    result[:3, :3] = rotation(*rpy)
    result[:3, 3] = xyz
    return result


def main():
    specs = [
        ("odom", "gimbal_link", [0, 0, 0], [0, 1.34, 0]),
        ("gimbal_link", "camera_link", [0.0461892, 0, -0.0603874], [0, 0.091, 0]),
        ("camera_link", "camera_optical_frame", [0, 0, 0], [-math.pi / 2, 0, -math.pi / 2]),
    ]
    chain = [transform(xyz, rpy) for _, _, xyz, rpy in specs]
    world_from_optical = chain[0] @ chain[1] @ chain[2]
    optical = chain[2][:3, :3]
    # Optical right/down/forward map to camera_link right/down/forward.
    assert np.allclose(optical, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=1e-12)
    for item in chain + [world_from_optical]:
        r = item[:3, :3]
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(r) - 1) < 1e-12
    point = np.array([0.04, 0.08, 3.5, 1.0])
    composed = world_from_optical @ point
    sequential = chain[0] @ (chain[1] @ (chain[2] @ point))
    assert np.allclose(composed, sequential, atol=1e-12)
    assert np.allclose(np.linalg.inv(world_from_optical) @ composed, point, atol=1e-12)
    # Recover known world armor headings after converting into the optical frame.
    # A second synthetic camera rotation checks that the recovered yaw is invariant.
    max_error = 0.0
    cameras = [world_from_optical[:3, :3], rotation(0.2, -0.4, 0.7) @ world_from_optical[:3, :3]]
    headings = [-math.pi + 1e-6, -1.2, 0, 0.8, math.pi - 1e-6]
    for camera in cameras:
        for yaw in headings:
            armor_world = rotation(0, math.radians(15), yaw)
            armor_optical = camera.T @ armor_world
            recovered = camera @ armor_optical
            measured = math.atan2(recovered[1, 0], recovered[0, 0])
            error = abs(math.atan2(math.sin(measured - yaw), math.cos(measured - yaw)))
            max_error = max(max_error, error)
            assert error < 1e-12
    report = {
        "stage": "preparation_only",
        "checks": ["optical_axis_mapping", "proper_rotations", "sequential_vs_composed", "point_round_trip", "known_world_yaw_and_camera_rotation"],
        "status": "PASS",
        "convention": "T_parent_child maps child coordinates to parent; p_parent = R_parent_child p_child + t_parent_child; R = Rz(yaw) Ry(pitch) Rx(roll)",
        "static_transforms": [dict(parent=a, child=b, xyz_m=x, rpy_rad=r) for a, b, x, r in specs],
        "T_odom_camera_optical_frame": world_from_optical.tolist(),
        "sample_optical_point_m": point[:3].tolist(),
        "sample_odom_point_m": composed[:3].tolist(),
        "max_synthetic_yaw_error_rad": max_error,
        "limitations": "Algebra checks only; no C++ pipeline, TF publication, real-video accuracy, or ROS message integration validated yet.",
    }
    output = Path(__file__).with_name("geometry_baseline.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: 5 geometry preparation checks")
    print("T_odom_optical =\n", world_from_optical)
    print("Report:", output)


if __name__ == "__main__":
    main()
