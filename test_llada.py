#!/usr/bin/env python
"""Test LLaDA model loading and generation."""
import os
os.environ["LLADA_MODEL_PATH"] = "/scratch/si2356/models/llada-1.5-8b"

from src.llada_io.model import LLaDAModel

def main():
    print("="*60)
    print("Testing LLaDA Model")
    print("="*60 + "\n")
    
    # Load model
    model = LLaDAModel(num_steps=4)
    
    # Test prompt
    prompt = "What is the capital of France?"
    print(f"Prompt: {prompt}")
    print("Generating...\n")
    
    # Generate
    response = model.generate(prompt, max_length=100)
    
    print("="*60)
    print("Response:")
    print(response)
    print("="*60)

if __name__ == "__main__":
    main()
