#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_alias="$HOME/armor_ekf_ws"
runtime_root="${ARMOR_EKF_RUNTIME:-$HOME/armor_ekf_runtime}"
# rosidl_adapter resolves its output paths; the actual build directory must be ASCII.
python3 - "$workspace" "$source_alias" <<'PY'
from pathlib import Path
import sys
source, alias = map(Path, sys.argv[1:])
if alias.is_symlink():
    if alias.resolve() != source.resolve():
        raise SystemExit(f'Existing alias points elsewhere: {alias}')
elif alias.exists():
    raise SystemExit(f'Existing path is not a symlink: {alias}')
else:
    alias.symlink_to(source, target_is_directory=True)
PY
sdk="$workspace/deps/onnxruntime-linux-x64-1.30.0"
if [[ ! -f "$sdk/include/onnxruntime_cxx_api.h" || ! -f "$sdk/lib/libonnxruntime.so" ]]; then
  mkdir -p "$workspace/deps"
  archive="$(mktemp --suffix=.tgz)"
  trap 'rm -f "$archive"' EXIT
  curl -fL --retry 2 'https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz' -o "$archive"
  tar -xzf "$archive" -C "$workspace/deps"
fi
set +u
source /opt/ros/lyrical/setup.bash
set -u
/usr/bin/colcon --log-base "$runtime_root/log" build --base-paths "$source_alias/src" \
  --build-base "$runtime_root/build" --install-base "$runtime_root/install" \
  --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
echo "Load the workspace with: source $runtime_root/install/setup.bash"
