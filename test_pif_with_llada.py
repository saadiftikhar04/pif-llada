#!/usr/bin/env python
"""Test PiF attack with real LLaDA model."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Set PYTHONPATH for dllm
os.environ['PYTHONPATH'] = '/scratch/si2356/pif-llada-minimal/dllm:' + os.environ.get('PYTHONPATH', '')

import torch
from dataclasses import dataclass
from dllm.utils import get_tokenizer, get_model
from src.llada_io.intent_probe import IntentProbe
from src.pif_attack.attack import PiFAttack

@dataclass
class ModelArgs:
    model_name_or_path: str = "/scratch/si2356/models/llada-1.5-8b"

print("="*60)
print("PiF-LLaDA Attack with Real Model (T=4)")
print("="*60 + "\n")

# Load LLaDA
model_args = ModelArgs()
print("Loading LLaDA model...")
tokenizer = get_tokenizer(model_args=model_args)
model = get_model(model_args=model_args).eval()

# Set mask_token_id
if tokenizer.mask_token_id is None:
    tokenizer.mask_token_id = model.config.mask_token_id

print(f"✓ Model loaded on {model.device}\n")

# Create probe
probe = IntentProbe(model=None, template_id=0)  # Using keyword-based for now

# Load prompt
with open("data/toy_prompt.txt") as f:
    prompt = f.read().strip()

# Create attack
attack = PiFAttack(model=None, probe=probe, max_edits=10, tokens_per_iter=2)

# Execute attack
print("Executing PiF attack...\n")
adversarial, edits = attack.attack(prompt)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Original:     {prompt}")
print(f"Adversarial:  {adversarial}")
print(f"Edits made:   {edits}")
print("="*60)

print("\n✅ PiF attack complete! Model ready for full experiments.")
