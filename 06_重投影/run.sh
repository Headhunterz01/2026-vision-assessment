#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$task_dir/.venv/bin/python" ]]; then
  /usr/bin/python3 -m venv --system-site-packages "$task_dir/.venv"
  "$task_dir/.venv/bin/python" -m pip install -r "$task_dir/requirements.txt"
fi
if [[ "${1:-}" == ros ]]; then
  shift
  set +u
  source /opt/ros/lyrical/setup.bash
  set -u
  exec "$task_dir/.venv/bin/python" "$task_dir/src/ros_node.py" "$@"
fi
exec "$task_dir/.venv/bin/python" "$task_dir/src/reproject.py" "$@"
