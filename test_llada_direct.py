#!/usr/bin/env python
"""Test LLaDA directly with HuggingFace transformers."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("="*60)
print("Testing LLaDA Direct Load")
print("="*60 + "\n")

model_path = "/scratch/si2356/models/llada-1.5-8b"

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# Load model
print(f"Loading from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("✓ Model loaded!\n")

# Test
prompt = "What is the capital of France?"
print(f"Prompt: {prompt}\n")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("="*60)
print("Response:")
print(response)
print("="*60)
