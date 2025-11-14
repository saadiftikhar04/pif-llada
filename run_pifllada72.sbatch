#!/bin/bash -l
#SBATCH -J pifllada72
#SBATCH -p nvidia
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --time=2-23:59:59
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

# Caches & performance
export HF_HOME=/scratch/si2356/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1        # you installed hf_transfer, enable fast path
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs
echo "== $(date) :: node=$HOSTNAME :: job=${SLURM_JOB_ID} =="

# Optional: quick diag prints to .out
conda run -n pif-llada --no-capture-output python - <<'PY'
import os, sys, torch
print("PY:", sys.executable)
print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_HUB_ENABLE_HF_TRANSFER:", os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"))
print("CUDA available:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())
PY

# Requeue on preemption (if your cluster signals it)
trap 'echo "$(date) :: Preemption; requeue ${SLURM_JOB_ID}"; scontrol requeue ${SLURM_JOB_ID}; exit 0' SIGUSR1 SIGTERM

# Run workload
conda run -n pif-llada --no-capture-output python -u run_pif_llada_ENHANCED.py
status=$?
echo "== $(date) :: exit code: $status =="
exit $status

