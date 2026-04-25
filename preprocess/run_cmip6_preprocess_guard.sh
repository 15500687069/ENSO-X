#!/usr/bin/env bash
set -u -o pipefail

# Guard runner for long CMIP6 preprocessing jobs.
# - Safe against SSH disconnect (run with nohup)
# - Auto-retry per model on failure
# - Resume by checking existing output count (10 files/model)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS=""
MAX_RETRY=20
RETRY_SLEEP_SEC=120

usage() {
  cat <<EOF
Usage: bash run_cmip6_preprocess_guard.sh [options]
Options:
  --root <path>            Project root (default: parent of this script)
  --models <csv>           Model list, e.g. "CESM2,CESM2-WACCM"
  --max-retry <n>          Max retries per model (default: 20)
  --retry-sleep <sec>      Sleep between retries (default: 120)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --max-retry) MAX_RETRY="$2"; shift 2 ;;
    --retry-sleep) RETRY_SLEEP_SEC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

RAW="$ROOT/data/cmip6_raw"
OUT="$ROOT/data/ctefnet_data/CMIP6var"
PY="$ROOT/preprocess/preprocess_cmip6_to_ensox.py"
LOGDIR="$ROOT/preprocess/logs"
STATE="$LOGDIR/preprocess_guard_state.log"
LOCKDIR="$LOGDIR/preprocess_guard.lock"
PIDFILE="$LOGDIR/preprocess_guard.pid"

mkdir -p "$LOGDIR"

if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[ERR] another guard may be running: $LOCKDIR"
  exit 1
fi

cleanup() {
  rm -rf "$LOCKDIR" "$PIDFILE" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "$$" > "$PIDFILE"

if [[ -z "$MODELS" ]]; then
  echo "[ERR] --models is required"
  exit 2
fi

count_outputs() {
  local model="$1"
  find "$OUT" -type f -name "${model}_ssp370_185001-210012.npz" 2>/dev/null | wc -l
}

echo "[$(date '+%F %T')] guard start root=$ROOT models=$MODELS max_retry=$MAX_RETRY retry_sleep=$RETRY_SLEEP_SEC" | tee -a "$STATE"

IFS=',' read -r -a MODEL_ARR <<< "$MODELS"

for raw_m in "${MODEL_ARR[@]}"; do
  m="$(echo "$raw_m" | xargs)"
  [[ -z "$m" ]] && continue

  cur="$(count_outputs "$m" | xargs)"
  if [[ "$cur" -ge 10 ]]; then
    echo "[$(date '+%F %T')] [SKIP] $m outputs=$cur/10" | tee -a "$STATE"
    continue
  fi

  attempt=0
  while true; do
    attempt=$((attempt + 1))
    echo "[$(date '+%F %T')] [RUN] model=$m attempt=$attempt outputs_before=$(count_outputs "$m" | xargs)/10" | tee -a "$STATE"

    model_log="$LOGDIR/preprocess_${m}_strict.log"
    python -u "$PY" \
      --raw-root "$RAW" \
      --out-root "$OUT" \
      --models "$m" \
      --wmean-fallback thetao_5 \
      --depth-max 300 >> "$model_log" 2>&1
    rc=$?

    now_count="$(count_outputs "$m" | xargs)"
    echo "[$(date '+%F %T')] [END] model=$m attempt=$attempt rc=$rc outputs_after=${now_count}/10" | tee -a "$STATE"

    if [[ "$rc" -eq 0 && "$now_count" -ge 10 ]]; then
      echo "[$(date '+%F %T')] [DONE] $m" | tee -a "$STATE"
      break
    fi

    if [[ "$attempt" -ge "$MAX_RETRY" ]]; then
      echo "[$(date '+%F %T')] [GIVEUP] $m after $attempt attempts" | tee -a "$STATE"
      break
    fi

    echo "[$(date '+%F %T')] [RETRY] $m sleep=${RETRY_SLEEP_SEC}s" | tee -a "$STATE"
    sleep "$RETRY_SLEEP_SEC"
  done
done

echo "[$(date '+%F %T')] guard finished" | tee -a "$STATE"
