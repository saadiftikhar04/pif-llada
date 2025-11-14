#!/bin/bash
#SBATCH --job-name=pif-llada
#SBATCH --partition=nvidia
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --output=experiments/harmbench_corrected/slurm_%j.out
#SBATCH --error=experiments/harmbench_corrected/slurm_%j.err

# Print job info
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=================================="
echo ""

# Setup environment
cd /scratch/si2356/pif-llada-minimal
source ~/.bashrc
conda activate pif-llada
export PYTHONPATH=/scratch/si2356/pif-llada-minimal/dllm:$PYTHONPATH

# Verify GPU
echo "GPU Check:"
nvidia-smi
echo ""

# Apply generation fixes (idempotent)
echo "Applying generation parameter fixes..."
if [ -f run_pif_llada_CORRECTED.py.backup ]; then
    echo "  Already applied, using existing version"
else
    ./apply_generation_fix.sh
fi
echo ""

# Create output directory
mkdir -p experiments/harmbench_corrected

# Run experiment with full logging
echo "Starting PiF-LLaDA experiment..."
echo "Results will be saved to: experiments/harmbench_corrected/"
echo ""

python run_pif_llada_CORRECTED.py 2>&1 | tee experiments/harmbench_corrected/run_${SLURM_JOB_ID}.log

# Record completion
echo ""
echo "=================================="
echo "Job completed at: $(date)"
echo "=================================="
echo ""

# Display final results
if [ -f experiments/harmbench_corrected/summary.json ]; then
    echo "FINAL RESULTS:"
    cat experiments/harmbench_corrected/summary.json
    echo ""
    echo "✓ Full results: experiments/harmbench_corrected/pif_llada_corrected_results.json"
    echo "✓ Summary: experiments/harmbench_corrected/summary.json"
    echo "✓ Log: experiments/harmbench_corrected/run_${SLURM_JOB_ID}.log"
else
    echo "WARNING: Summary file not created - check logs for errors"
fi

