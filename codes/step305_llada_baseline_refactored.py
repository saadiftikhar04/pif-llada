#!/usr/bin/env python
"""
Step 3.05: LLaDA Baseline (Refactored)
Using src/lldms.py
"""

import sys
sys.path.append('src')

from lldms import LLaDA
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Step 3.05: LLaDA Baseline')
    parser.add_argument('--prompt', type=str,
                        default="What is the capital of France?",
                        help='Prompt to test')
    parser.add_argument('--model', type=str,
                        default='GSAI-ML/LLaDA-1.5',
                        help='LLaDA model')
    parser.add_argument('--max-tokens', type=int, default=128,
                        help='Max tokens to generate')
    parser.add_argument('--steps', type=int, default=128,
                        help='Number of denoising steps')
    parser.add_argument('--output', type=str, default='step305_result.json',
                        help='Output file')
    args = parser.parse_args()
    
    print("="*80)
    print("STEP 3.05: LLaDA Baseline (Refactored)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print()
    
    # Load LLaDA using class
    llada = LLaDA(args.model)
    
    # Generate response
    print("[GENERATING]")
    response, elapsed = llada.infer(
        args.prompt,
        max_new_tokens=args.max_tokens,
        steps=args.steps,
        return_time=True
    )
    
    print(f"Response: {response}")
    print(f"Time: {elapsed:.2f}s\n")
    
    # Save results
    results = {
        'model': args.model,
        'prompt': args.prompt,
        'response': response,
        'time': elapsed,
        'max_tokens': args.max_tokens,
        'steps': args.steps
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
