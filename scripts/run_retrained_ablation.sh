#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${ENSOX_ABLATION_BASE_CONFIG:-configs/enso_x_24_final.yaml}"
DATA_ROOT="${ENSOX_DATA_ROOT:?Please set ENSOX_DATA_ROOT to the processed ENSO-X data directory.}"
INIT_DIR="${ENSOX_ABLATION_INIT_DIR:-./outputs/ablation_init}"
CONFIG_DIR="${ENSOX_ABLATION_CONFIG_DIR:-configs/ablation_retrain}"
OUT_ROOT="${ENSOX_ABLATION_OUTPUT_ROOT:-./outputs/retrained_ablation}"
SEEDS="${ENSOX_ABLATION_SEEDS:-0 1 2}"
VARIANTS="${ENSOX_ABLATION_VARIANTS:-full no_memory no_local_lead_repair no_legal_analog no_reanalysis_repair}"
EPOCHS="${ENSOX_ABLATION_EPOCHS:-60}"
PYTHON="${ENSOX_PYTHON:-python}"

mkdir -p "$INIT_DIR" "$CONFIG_DIR" "$OUT_ROOT" results

export ENSOX_LEGAL_ANALOG_INIT="$INIT_DIR/legal_analog_train_only.npz"
export ENSOX_LEGAL_ANALOG_INIT_NO_MEMORY="$INIT_DIR/legal_analog_train_only_no_memory.npz"
export ENSOX_ABLATION_OUTPUT_ROOT="$OUT_ROOT"

"$PYTHON" scripts/audit_data_leakage_enso_x.py \
  --config "$BASE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --output-json results/enso_x_leakage_audit.json

"$PYTHON" scripts/build_legal_analog_init.py \
  --config "$BASE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --output "$ENSOX_LEGAL_ANALOG_INIT"

"$PYTHON" scripts/build_legal_analog_init.py \
  --config "$BASE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --output "$ENSOX_LEGAL_ANALOG_INIT_NO_MEMORY" \
  --zero-memory-input

"$PYTHON" scripts/prepare_retrained_ablation_configs.py \
  --base-config "$BASE_CONFIG" \
  --output-dir "$CONFIG_DIR" \
  --seeds $SEEDS \
  --variants $VARIANTS \
  --epochs "$EPOCHS"

while IFS= read -r cfg; do
  [ -z "$cfg" ] && continue
  exp="$(basename "$cfg" .yaml)"
  run_dir="$OUT_ROOT/$exp"
  mkdir -p "$run_dir"
  if [ -f "$run_dir/training_summary.json" ] && [ "${ENSOX_ABLATION_FORCE:-0}" != "1" ]; then
    echo "[Ablation] skip completed $exp"
    continue
  fi
  echo "[Ablation] run $exp"
  "$PYTHON" -u train.py --config "$cfg" 2>&1 | tee "$run_dir/train.log"
  rm -f "$run_dir/last.ckpt" "$run_dir/best_score.ckpt"
  "$PYTHON" scripts/summarize_retrained_ablation.py \
    --output-root "$OUT_ROOT" \
    --output-json results/enso_x_retrained_ablation_summary.json
done < "$CONFIG_DIR/manifest.txt"

"$PYTHON" scripts/summarize_retrained_ablation.py \
  --output-root "$OUT_ROOT" \
  --output-json results/enso_x_retrained_ablation_summary.json
