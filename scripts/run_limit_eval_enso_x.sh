#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_limit_eval_enso_x.sh <checkpoint_path> [output_json]"
  exit 1
fi
CKPT_PATH="$1"
OUT_JSON="${2:-$ROOT/results/enso_x_limit_eval.json}"

export ENSOX_DATA_ROOT="${ENSOX_DATA_ROOT:-$ROOT/data/ctefnet_data}"

exec "$PYTHON_BIN" "$ROOT/scripts/evaluate_limit_enso_x.py" \
  --base-config "$ROOT/configs/enso_x_24_final.yaml" \
  --ckpt "$CKPT_PATH" \
  --data-root "$ENSOX_DATA_ROOT" \
  --output-json "$OUT_JSON"
