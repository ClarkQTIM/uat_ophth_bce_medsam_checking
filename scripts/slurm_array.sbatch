#!/bin/bash
# SLURM array launcher: one array task = one line of a job file.
#
#   sbatch --array=0-259%15 scripts/slurm_array.sbatch jobs/parity_sam.jsonl
#
# Partition/account/env lines mirror the Alpine settings used for the paper's
# gamma ablation (bash_scripts/run_experiments.sh); edit for your cluster.
#SBATCH --job-name=paper_crisis
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
JOBS="${1:?usage: sbatch --array=0-N scripts/slurm_array.sbatch <jobs.jsonl>}"

module purge || true
module load anaconda || true
source activate "${CONDA_ENV:-cclark_dl4}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"
mkdir -p logs

echo "host=$(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/run_uat.py --jobs "$JOBS" --index "$SLURM_ARRAY_TASK_ID"
