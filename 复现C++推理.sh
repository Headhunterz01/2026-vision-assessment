#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ROOT}/01_神经网络/yolo_pose/YOLO_Pose_OpenVINO_C_Inference"
MODEL="${ROOT}/01_神经网络/yolo_pose/final_model/best.onnx"
VIDEO="${ROOT}/00_题目原件/videos/armors_yolo_test.mp4"
VENV="${TMPDIR:-/tmp}/vision-openvino-venv"
BUILD="${TMPDIR:-/tmp}/vision-openvino-build"
OUTPUT="${TMPDIR:-/tmp}/C++推理复现_$(date +%Y%m%d_%H%M%S)"

for path in "$REPO/CMakeLists.txt" "$MODEL" "$VIDEO"; do
  if [[ ! -f "$path" ]]; then
    echo "缺少所需文件：$path" >&2
    exit 1
  fi
done

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c 'import openvino' >/dev/null 2>&1; then
  "$VENV/bin/python" -m pip install openvino
fi
OV_CMAKE_DIR=$("$VENV/bin/python" -c 'from openvino.utils import get_cmake_path; print(get_cmake_path())')
OV_LIB_DIR=$("$VENV/bin/python" -c 'import openvino, pathlib; print(pathlib.Path(openvino.__file__).parent / "libs")')
cmake -S "$REPO" -B "$BUILD" -DOpenVINO_DIR="$OV_CMAKE_DIR"
cmake --build "$BUILD" -j2
mkdir -p "$OUTPUT"
export LD_LIBRARY_PATH="$OV_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$BUILD/yolo_pose_demo" --model "$MODEL" --video "$VIDEO" \
  --device CPU --input-size 640 --conf 0.25 --iou 0.45 --kpt-conf 0.5 \
  --headless --save-dir "$OUTPUT" --save-every 400 2>&1 | tee "$OUTPUT/run.log"
printf '结果目录：%s\n' "$OUTPUT"
