#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Использование: ./run_jobs.sh <jobs_dir> [parallel_jobs] [master_log]"
  echo "Примеры:"
  echo "  ./run_jobs.sh jobs/01_xtb 8 xtb_master.log"
  echo "  ./run_jobs.sh jobs/02_r2scan 1 dft_master.log"
  exit 1
fi

JOBS_DIR="$1"
PARALLEL_JOBS="${2:-1}"
MASTER_LOG="${3:-run_jobs_master.log}"

ORCA_PATH=$(python3 - <<'PY'
import json
with open('settings.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)
print(cfg['orca_path'])
PY
)

if [ ! -x "$ORCA_PATH" ]; then
  echo "ORCA не найден или не исполняемый: $ORCA_PATH"
  exit 1
fi

mkdir -p .status

mapfile -t INPUTS < <(find "$JOBS_DIR" -type f -name "*.inp" | sort)
TOTAL="${#INPUTS[@]}"

if [ "$TOTAL" -eq 0 ]; then
  echo "В $JOBS_DIR нет .inp файлов"
  exit 1
fi

echo "[$(date '+%F %T')] START jobs_dir=$JOBS_DIR total=$TOTAL parallel=$PARALLEL_JOBS" | tee -a "$MASTER_LOG"

run_one() {
  local idx="$1"
  local total="$2"
  local inp="$3"

  local inp_abs inp_dir inp_file base out status_file
  inp_abs="$(readlink -f "$inp")"
  inp_dir="$(dirname "$inp_abs")"
  inp_file="$(basename "$inp_abs")"
  base="${inp_file%.inp}"
  out="$inp_dir/${base}.out"
  status_file=".status/${base}.running"

  if [ -f "$out" ] && grep -q "ORCA TERMINATED NORMALLY" "$out"; then
    echo "[$(date '+%F %T')] [SKIP $idx/$total] $inp" | tee -a "$MASTER_LOG"
    return 0
  fi

  echo "$inp_abs" > "$status_file"
  echo "[$(date '+%F %T')] [RUN  $idx/$total] $inp" | tee -a "$MASTER_LOG"

  set +e
  (
    cd "$inp_dir"
    "$ORCA_PATH" "$inp_file" > "${base}.out" 2>&1
  )
  rc=$?
  set -e

  rm -f "$status_file"

  if [ "$rc" -eq 0 ] && grep -q "ORCA TERMINATED NORMALLY" "$out"; then
    echo "[$(date '+%F %T')] [DONE $idx/$total] $inp" | tee -a "$MASTER_LOG"
  else
    echo "[$(date '+%F %T')] [FAIL $idx/$total] $inp rc=$rc" | tee -a "$MASTER_LOG"
    return "$rc"
  fi
}

running_jobs_count() {
  jobs -rp | wc -l | awk '{print $1}'
}

fail=0
for i in "${!INPUTS[@]}"; do
  idx=$((i+1))
  inp="${INPUTS[$i]}"

  while [ "$(running_jobs_count)" -ge "$PARALLEL_JOBS" ]; do
    sleep 2
  done

  run_one "$idx" "$TOTAL" "$inp" &
done

for job in $(jobs -rp); do
  if ! wait "$job"; then
    fail=1
  fi
done

echo "[$(date '+%F %T')] FINISH jobs_dir=$JOBS_DIR fail=$fail" | tee -a "$MASTER_LOG"
exit "$fail"
