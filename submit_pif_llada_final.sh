#!/bin/bash
#SBATCH --job-name=pif-llada-final
#SBATCH --partition=nvidia
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=experiments/harmbench_final/slurm_%j.out
#SBATCH --error=experiments/harmbench_final/slurm_%j.err

echo "=================================="
echo "PiF-LLaDA FINAL FIX"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=================================="

cd /scratch/si2356/pif-llada-minimal
source ~/.bashrc
conda activate pif-llada
export PYTHONPATH=/scratch/si2356/pif-llada-minimal/dllm:$PYTHONPATH

nvidia-smi

mkdir -p experiments/harmbench_final

echo ""
echo "Running FINAL fixed experiment..."
python run_pif_llada_FINAL.py 2>&1 | tee experiments/harmbench_final/run_${SLURM_JOB_ID}.log

echo ""
echo "=================================="
echo "Completed: $(date)"
echo "=================================="

if [ -f experiments/harmbench_final/summary.json ]; then
    echo ""
    echo "RESULTS:"
    cat experiments/harmbench_final/summary.json
fi

