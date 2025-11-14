#!/bin/bash -l
#SBATCH -J pifllada
#SBATCH -p nvidia
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -t 23:59:59
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

# --- Safety: don't use nounset until after .bashrc ---
# If your site uses modules, -l above loads them. Otherwise uncomment:
# module load cuda/12.1
# module load anaconda

# Conda
# If -l doesn't bring conda in, fall back to manual init without failing:
[[ -f ~/.bashrc ]] && source ~/.bashrc || true
conda activate pif-llada

# Caches & perf
export HF_HOME=/scratch/si2356/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Now safe to tighten bash
set -euo pipefail

echo "== $(date) :: Node=$HOSTNAME :: JobID=${SLURM_JOB_ID} =="

# Requeue on preemption if your cluster signals it
trap 'echo "$(date) :: Preemption; requeue ${SLURM_JOB_ID}"; scontrol requeue ${SLURM_JOB_ID}; exit 0' SIGUSR1 SIGTERM

python -u run_pif_llada_ENHANCED.py
status=$?
echo "== $(date) :: Exit code: $status =="
exit $status
