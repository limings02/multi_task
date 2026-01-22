#!/usr/bin/env bash
set -euo pipefail

# Locate repo root relative to this script (scripts/..).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default config; allow overriding by passing a config path as the first arg.
CONFIG_PATH="${REPO_ROOT}/configs/dataset/aliccp.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG_PATH="$1"
  shift
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python -m src.cli.main canonical --config "${CONFIG_PATH}" "$@"
