#!/bin/bash
# Run a range of lines from a job file sequentially on one GPU (no SLURM).
#
#   scripts/run_local.sh jobs/parity_sam.jsonl 0            # GPU 0, all jobs
#   scripts/run_local.sh jobs/parity_sam.jsonl 1 0 129      # GPU 1, lines 0..129
#
# Completed runs (test_metrics.txt present) are skipped, so the script can be
# re-launched after an interruption. Logs go to logs/local_<jobfile>_<line>.log
set -uo pipefail
JOBS="${1:?usage: run_local.sh <jobs.jsonl> <gpu> [start] [end]}"
GPU="${2:?gpu id}"
N=$(grep -c . "$JOBS")
START="${3:-0}"
END="${4:-$((N - 1))}"
mkdir -p logs
NAME=$(basename "$JOBS" .jsonl)
echo "running lines $START..$END of $JOBS ($N total) on GPU $GPU"
for i in $(seq "$START" "$END"); do
  echo "=== line $i ==="
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_uat.py --jobs "$JOBS" --index "$i" \
      > "logs/local_${NAME}_${i}.log" 2>&1 || echo "line $i FAILED (see logs/local_${NAME}_${i}.log)"
done
