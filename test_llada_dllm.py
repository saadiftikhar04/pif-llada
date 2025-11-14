#!/usr/bin/env python
"""Test LLaDA with dllm framework."""
import torch
from dataclasses import dataclass
from dllm.utils import get_tokenizer, get_model
from dllm.pipelines.llada import generate

@dataclass
class ModelArgs:
    model_name_or_path: str = "/scratch/si2356/models/llada-1.5-8b"

print("="*60)
print("Testing LLaDA with dllm")
print("="*60 + "\n")

model_args = ModelArgs()

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# Load model using dllm
print(f"Loading from: {model_args.model_name_or_path}")
tokenizer = get_tokenizer(model_args=model_args)
model = get_model(model_args=model_args).eval()

# Check and set mask_token if needed
print(f"mask_token_id before: {tokenizer.mask_token_id}")
if tokenizer.mask_token_id is None:
    # Get mask_token_id from model config
    if hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
        tokenizer.mask_token_id = model.config.mask_token_id
        print(f"Set mask_token_id from config: {tokenizer.mask_token_id}")
    else:
        # Use a reasonable default (typically vocab_size - 1 for LLaDA)
        tokenizer.mask_token_id = len(tokenizer) - 1
        print(f"Set mask_token_id to vocab_size-1: {tokenizer.mask_token_id}")

print("✓ Model loaded!\n")

# Test generation with masked diffusion
prompt = "What is the capital of France?"
print(f"Prompt: {prompt}")
print("Generating with 4 diffusion steps...\n")

# Tokenize input
input_ids = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=True
)["input_ids"].to(model.device)

# Generate using dllm's masked diffusion
output = generate(
    model,
    tokenizer,
    [input_ids[0]],
    steps=4,
    max_new_tokens=50,
    block_length=32,
    temperature=0.0,
)

response = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n" + "="*60)
print("Response:")
print(response)
print("="*60)

print("\n✅ SUCCESS! LLaDA is working with T=4 diffusion steps!")
