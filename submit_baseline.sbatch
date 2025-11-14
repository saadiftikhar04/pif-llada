#!/bin/bash
#SBATCH --job-name=llada_baseline
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Load modules
module purge
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate pif-llada

# Set model path
export LLADA_MODEL_PATH="/scratch/si2356/models/llada-1.5-8b"

# Run baseline
echo "Starting LLaDA baseline on HarmBench..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python run_baseline.py

echo "Baseline complete!"
