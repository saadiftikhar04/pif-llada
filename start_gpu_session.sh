#!/bin/bash
# Quick start script for PiF-LLaDA GPU sessions

echo "Requesting GPU node..."
srun -p nvidia --qos=nvidias --gres=gpu:1 --mem=32GB -t 2:00:00 --pty bash << 'INNER'
cd /scratch/si2356/pif-llada-minimal
conda activate pif-llada
export PYTHONPATH=/scratch/si2356/pif-llada-minimal/dllm:$PYTHONPATH
echo ""
echo "✅ Ready! You're on a GPU node with environment activated."
echo ""
echo "Quick tests:"
echo "  python test_llada_dllm.py          # Test LLaDA generation"
echo "  python test_pif_with_llada.py      # Test PiF attack"
echo "  python toy_example.py              # Toy example"
echo ""
bash
INNER
