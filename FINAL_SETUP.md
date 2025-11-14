# PiF-LLaDA: Setup Complete! ✅

## Working Configuration

### Environment
- **Location**: `/scratch/si2356/pif-llada-minimal`
- **Conda env**: `pif-llada`
- **GPU**: NVIDIA A100-PCIE-40GB
- **PYTHONPATH**: Must be set: `export PYTHONPATH=/scratch/si2356/pif-llada-minimal/dllm:$PYTHONPATH`

### Model
- **Path**: `/scratch/si2356/models/llada-1.5-8b`
- **Framework**: dllm
- **Diffusion steps**: T=4 (working!)
- **mask_token_id**: 126336 (from config)

### Verified Working
✅ LLaDA model loads on A100
✅ Generation works with T=4 diffusion steps
✅ Toy PiF example runs (keyword-based probe)
✅ Environment fully configured

## Quick Start Commands

### Get GPU node:
```bash
srun -p nvidia --qos=nvidias --gres=gpu:1 --mem=32GB -t 30:00 --pty bash
```

### Activate environment:
```bash
cd /scratch/si2356/pif-llada-minimal
conda activate pif-llada
export PYTHONPATH=/scratch/si2356/pif-llada-minimal/dllm:$PYTHONPATH
```

### Test LLaDA:
```bash
python test_llada_dllm.py
```

### Run PiF toy example:
```bash
python toy_example.py  # Keyword-based
python test_pif_with_llada.py  # With real model loaded
```

## Next Steps

### 1. Implement LLaDA-based Intent Probe
Update `src/llada_io/intent_probe.py` to use LLaDA's diffusion scoring instead of keywords.

### 2. Run Full Toy Example
Execute PiF attack with LLaDA scoring on one prompt with T=4.

### 3. Start HarmBench Baseline
Submit batch job to evaluate baseline LLaDA performance on HarmBench.

### 4. Scale Up
Once toy example works end-to-end, expand to full HarmBench dataset.

### 5. Begin Paper Writing
Start drafting while experiments run in background.

## Key Files
- `test_llada_dllm.py` - Verified working LLaDA test
- `test_pif_with_llada.py` - PiF attack with real model
- `toy_example.py` - Simple keyword-based example
- `src/llada_io/model.py` - Model wrapper
- `src/pif_attack/attack.py` - PiF attack implementation

## Important Notes
- Always set PYTHONPATH before running
- Use T=4 for speed (can increase later)
- mask_token_id must be set from model.config
- GPU session times out after allocated time

## Time Estimate
- Single prompt PiF attack: ~30 seconds
- HarmBench baseline (200 prompts): ~2-3 hours
- Full ablation studies: ~8-10 hours

🎯 **Goal**: Get ONE complete PiF attack working end-to-end with real LLaDA scoring, then expand!
