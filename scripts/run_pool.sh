#!/bin/bash
# Run a job file with several workers in parallel, several per GPU if memory allows.
#
#   scripts/run_pool.sh <jobs.jsonl> "<gpu list, one entry per worker>"
#
#   scripts/run_pool.sh jobs/parity_sam.jsonl "0 0 1 1"          # SAM ViT-H: 2 per A100 80GB
#   scripts/run_pool.sh jobs/medsam_primary.jsonl "0 0 0 0 0 0 1 1 1 1 1 1"   # MedSAM: 6 per GPU
#
# Worker k (0-based) of N runs lines k, k+N, k+2N, ... so no two workers ever take
# the same line, and every worker gets a mix of datasets/arms. Completed runs
# (test_metrics.txt present) are skipped, so re-launching resumes where it stopped.
# Measured peak memory at batch size 1 (smoke run 2026-10-14): SAM ViT-H 27.9 GB,
# MedSAM 9.3 GB per run.
#
# Run inside tmux so it survives disconnecting:
#   tmux new -s parity      (detach: Ctrl-b then d;  re-attach: tmux attach -t parity)
set -uo pipefail
JOBS="${1:?usage: run_pool.sh <jobs.jsonl> \"<gpu per worker>\"}"
read -r -a GPUS <<< "${2:?give one GPU id per worker, e.g. \"0 0 1 1\"}"
N=${#GPUS[@]}
TOTAL=$(grep -c . "$JOBS")
NAME=$(basename "$JOBS" .jsonl)
mkdir -p logs
echo "$(date '+%F %T') pool start: $JOBS ($TOTAL jobs), $N workers on GPUs: ${GPUS[*]}" | tee -a "logs/pool_${NAME}.log"

worker() {
  local k=$1 gpu=$2
  for ((i = k; i < TOTAL; i += N)); do
    echo "$(date '+%F %T') worker $k gpu $gpu line $i start" >> "logs/pool_${NAME}.log"
    if CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_uat.py --jobs "$JOBS" --index "$i" \
         > "logs/${NAME}_${i}.log" 2>&1; then
      echo "$(date '+%F %T') worker $k gpu $gpu line $i done" >> "logs/pool_${NAME}.log"
    else
      echo "$(date '+%F %T') worker $k gpu $gpu line $i FAILED (logs/${NAME}_${i}.log)" | tee -a "logs/pool_${NAME}.log"
    fi
  done
}

for ((k = 0; k < N; k++)); do
  worker "$k" "${GPUS[$k]}" &
  sleep 20   # stagger start-up (model loading)
done
wait
echo "$(date '+%F %T') pool finished: $JOBS" | tee -a "logs/pool_${NAME}.log"
grep -c " done" "logs/pool_${NAME}.log" | xargs echo "lines completed (this and earlier launches):"
grep FAILED "logs/pool_${NAME}.log" || echo "no failures"
