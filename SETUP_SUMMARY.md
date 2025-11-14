# PiF-LLaDA Setup Summary

## ✅ Completed
1. **Environment**: `pif-llada` conda environment with PyTorch, Transformers, etc.
2. **LLaDA Model**: Downloaded to `/scratch/si2356/models/llada-1.5-8b` (15GB)
3. **dllm Framework**: Cloned and installed
4. **Code Structure**: 
   - `src/llada_io/`: Model wrappers
   - `src/pif_attack/`: PiF attack components
   - `toy_example.py`: Working toy example (keyword-based)
   - `test_llada_simple.py`: GPU test script

## 📍 Current Status
- Toy example works with keyword-based probe
- Ready to test with real LLaDA model on GPU

## 🚀 Next Steps

### 1. Test LLaDA on GPU
```bash
srun --gres=gpu:1 --mem=32GB --time=1:00:00 --pty bash
cd /scratch/si2356/pif-llada-minimal
conda activate pif-llada
python test_llada_simple.py
```

### 2. Run Toy Example with Real Model
Once GPU test works, update toy_example.py to use real model

### 3. Start HarmBench Baseline
Submit baseline job to run in background

### 4. Begin Writing Paper
While experiments run, start paper structure

## 📁 Key Paths
- **Project**: `/scratch/si2356/pif-llada-minimal`
- **Model**: `/scratch/si2356/models/llada-1.5-8b`
- **dllm**: `/scratch/si2356/pif-llada-minimal/dllm`
- **Environment**: `conda activate pif-llada`

## 🎯 Goal
Get one toy example working with T=4 diffusion steps, then expand from there.
