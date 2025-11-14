#!/usr/bin/env python
"""Simple LLaDA test on GPU."""
import os
os.environ["LLADA_MODEL_PATH"] = "/scratch/si2356/models/llada-1.5-8b"

import torch
import dllm
from dllm.utils import get_tokenizer, get_model
from dllm.pipelines.llada import generate

def main():
    print("="*60)
    print("Testing LLaDA with dllm")
    print("="*60 + "\n")
    
    model_path = "/scratch/si2356/models/llada-1.5-8b"
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Load model
    print(f"Loading model from: {model_path}")
    tokenizer = get_tokenizer(model_path)
    model = get_model(
        model_name_or_path=model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    ).eval()
    
    print("✓ Model loaded!\n")
    
    # Test generation
    prompt = "What is the capital of France?"
    print(f"Prompt: {prompt}")
    print("Generating with 4 diffusion steps...\n")
    
    # Tokenize
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True
    )["input_ids"].to(model.device)
    
    # Generate
    output = generate(
        model=model,
        tokenizer=tokenizer,
        input_ids_list=[input_ids[0]],
        steps=4,
        max_new_tokens=50,
        temperature=0.0,
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print("="*60)
    print("Response:")
    print(response)
    print("="*60)

if __name__ == "__main__":
    main()
