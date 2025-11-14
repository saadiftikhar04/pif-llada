# PiF-LLaDA Minimal

Toy implementation of Prompt Importance Flattening for masked diffusion LLMs.

## Setup
```bash
conda activate pif-llada
```

## Quick Test (Without Model)
```bash
python toy_example.py
```

## With LLaDA Model
```bash
export LLADA_MODEL_PATH=/scratch/si2356/DiffuLLaMA/models/llada-1.5-8b
python toy_example.py
```

## Start Baseline
```bash
sbatch submit_baseline.sbatch
squeue -u $USER
```

## Directory Structure
```
pif-llada-minimal/
├── src/
│   ├── llada_io/          # Model interface
│   └── pif_attack/        # Attack components
├── data/                  # Prompts & datasets
├── configs/               # Configuration files
├── experiments/           # Results
└── logs/                  # SLURM outputs
```
