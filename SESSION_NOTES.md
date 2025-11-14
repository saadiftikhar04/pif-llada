# Session Notes - November 5, 2025

## ✅ Achievements Today
1. Set up complete PiF-LLaDA environment
2. Downloaded LLaDA-1.5-8B model (15GB)
3. Installed and configured dllm framework
4. **Successfully tested LLaDA with T=4 diffusion steps on A100 GPU**
5. PiF attack pipeline working with keyword-based probe
6. All dependencies installed and working

## 📊 Current Status
- **Model**: LLaDA-1.5-8B loaded and generating successfully
- **Hardware**: NVIDIA A100-PCIE-40GB
- **Framework**: dllm with masked diffusion
- **PiF Attack**: Implemented, using keyword probe (needs LLaDA-based probe)

## 🎯 Next Session Goals

### Priority 1: Implement LLaDA-based Intent Probe
Replace keyword matching with actual LLaDA diffusion scoring for intent detection.

### Priority 2: Test End-to-End PiF Attack
Run complete PiF attack with LLaDA scoring on 1 harmful prompt.

### Priority 3: Start HarmBench Baseline
Submit batch job to evaluate baseline LLaDA performance.

### Priority 4: Paper Structure
Begin drafting paper sections while experiments run.

## 🔧 Technical Details

### Working Setup
- PYTHONPATH must include: `/scratch/si2356/pif-llada-minimal/dllm`
- mask_token_id: 126336 (from model config)
- Diffusion steps: T=4 (takes ~5 seconds per generation)

### Files Ready
- `test_llada_dllm.py` ✅ Working
- `test_pif_with_llada.py` ✅ Working  
- `toy_example.py` ✅ Working
- `src/pif_attack/attack.py` ✅ Implemented
- `src/llada_io/intent_probe.py` ⚠️ Needs LLaDA integration

### Time Estimates
- Single generation: ~5 seconds (T=4)
- Single PiF attack: ~30-60 seconds
- HarmBench baseline (200 prompts): ~2-3 hours
- Full experiments: ~8-10 hours

## 📝 Important Commands

### Start GPU Session
```bash
srun -p nvidia --qos=nvidias --gres=gpu:1 --mem=32GB -t 2:00:00 --pty bash
cd /scratch/si2356/pif-llada-minimal
conda activate pif-llada
export PYTHONPATH=/scratch/si2356/pif-llada-minimal/dllm:$PYTHONPATH
```

### Run Tests
```bash
python test_llada_dllm.py           # Test generation
python test_pif_with_llada.py       # Test PiF attack
```

## 💡 Key Insights
1. LLaDA requires special diffusion-based generation (can't use standard .generate())
2. mask_token_id must be set from model.config
3. PYTHONPATH is critical for dllm imports
4. T=4 is fast enough for development, can increase for final runs
5. Keyword probe works but needs to be replaced with LLaDA-based scoring

## 📚 References
- LLaDA Model: https://huggingface.co/GSAI-ML/LLaDA-1.5
- dllm Framework: https://github.com/ZHZisZZ/dllm
- HarmBench: https://github.com/centerforaisafety/HarmBench

---
**Next session**: Implement LLaDA-based intent probe and test end-to-end attack!
