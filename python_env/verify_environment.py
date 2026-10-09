"""Check Python dependencies and read one frame from each assessment video."""
from pathlib import Path

import cv2
import numpy
import serial
import yaml

print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"PyYAML: {yaml.__version__}")
print(f"pyserial: {serial.VERSION}")

root = Path(__file__).resolve().parent.parent / "00_题目原件" / "videos"
for name in ("armors_yolo_test.mp4", "test.mp4"):
    cap = cv2.VideoCapture(str(root / name))
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read video: {name}")
        print(f"{name}: shape={frame.shape}, fps={cap.get(cv2.CAP_PROP_FPS):.6f}, "
              f"frames={int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    finally:
        cap.release()
