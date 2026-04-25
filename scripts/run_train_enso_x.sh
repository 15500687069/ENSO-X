#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-$ROOT/configs/enso_x_24_final.yaml}"

export ENSOX_DATA_ROOT="${ENSOX_DATA_ROOT:-$ROOT/data/ctefnet_data}"
export ENSOX_OUTPUT_ROOT="${ENSOX_OUTPUT_ROOT:-$ROOT/outputs/checkpoints}"

exec "$PYTHON_BIN" "$ROOT/train.py" --config "$CONFIG_PATH"
