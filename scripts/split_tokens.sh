#!/usr/bin/env bash
set -euo pipefail

# Locate repo root relative to this script (scripts/..).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${REPO_ROOT}/configs/dataset/aliccp.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG_PATH="$1"
  shift
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python -m src.cli.main split-tokens --config "${CONFIG_PATH}" "$@"

