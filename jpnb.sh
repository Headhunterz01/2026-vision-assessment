#!/usr/bin/env bash
set -euo pipefail

# Always start Jupyter from this script's directory (the repository root).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JUPYTER_BIN="${SCRIPT_DIR}/python_env/.venv/bin/jupyter"

if [[ ! -x "${JUPYTER_BIN}" ]]; then
  echo "Jupyter is not installed in the project environment: ${JUPYTER_BIN}" >&2
  echo "Install it with: python3 -m venv python_env/.venv && python_env/.venv/bin/pip install jupyter ipykernel" >&2
  exit 1
fi

cd -- "${SCRIPT_DIR}"
exec "${JUPYTER_BIN}" lab "$@"
