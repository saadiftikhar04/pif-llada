#!/usr/bin/env python
"""Simplified HarmBench baseline - no heavy dependencies."""
import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from dllm.utils import get_tokenizer, get_model
from dllm.pipelines.llada import generate
from src.llada_io.simple_classifier import SimpleRefusalClassifier

@dataclass
class ModelArgs:
    model_name_or_path: str = "/scratch/si2356/models/llada-1.5-8b"

def main():
    print("="*70)
    print("HarmBench Baseline Evaluation - LLaDA with T=4")
    print("Using Simple Refusal Classifier")
    print("="*70 + "\n")
    
    # Load model
    model_args = ModelArgs()
    print("Loading LLaDA model...")
    tokenizer = get_tokenizer(model_args=model_args)
    model = get_model(model_args=model_args).eval()
    
    if tokenizer.mask_token_id is None:
        tokenizer.mask_token_id = model.config.mask_token_id
    
    print(f"✓ Model loaded on {model.device}\n")
    
    # Load classifier
    classifier = SimpleRefusalClassifier()
    print("✓ Simple refusal classifier ready\n")
    
    # Load prompts
    prompt_file = Path("data/harmbench_harmful.jsonl")
    if not prompt_file.exists():
        print(f"Error: {prompt_file} not found!")
        print("Run the dataset download first.")
        return
    
    prompts = []
    with open(prompt_file) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    
    print(f"Loaded {len(prompts)} prompts from HarmBench\n")
    
    # Evaluate
    results = []
    successful_attacks = 0
    
    for i, prompt_data in enumerate(tqdm(prompts, desc="Evaluating")):
        prompt = prompt_data['prompt']
        
        try:
            # Tokenize
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True
            )["input_ids"].to(model.device)
            
            # Generate with T=4
            output = generate(
                model,
                tokenizer,
                [input_ids[0]],
                steps=4,
                max_new_tokens=100,
                block_length=32,
                temperature=0.0,
            )
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Classify: harmful = not a refusal
            is_harmful = classifier.is_harmful(response)
            
            results.append({
                'prompt': prompt,
                'response': response,
                'harmful': is_harmful,
                'behavior': prompt_data.get('behavior', ''),
                'category': prompt_data.get('category', ''),
            })
            
            if is_harmful:
                successful_attacks += 1
                
        except Exception as e:
            print(f"\nError on prompt {i}: {e}")
            results.append({
                'prompt': prompt,
                'response': f"ERROR: {str(e)}",
                'harmful': False,
                'behavior': prompt_data.get('behavior', ''),
                'category': prompt_data.get('category', ''),
            })
    
    # Save results
    output_dir = Path("experiments/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "harmbench_baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compute metrics
    total = len(results)
    asr = (successful_attacks / total * 100) if total > 0 else 0
    
    print(f"\n{'='*70}")
    print("BASELINE RESULTS")
    print(f"{'='*70}")
    print(f"Total prompts:      {total}")
    print(f"Harmful responses:  {successful_attacks}")
    print(f"ASR (Attack Success Rate): {asr:.2f}%")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Save summary
    summary = {
        'total_prompts': total,
        'successful_attacks': successful_attacks,
        'asr': asr,
        'model': model_args.model_name_or_path,
        'diffusion_steps': 4,
        'classifier': 'simple_refusal',
    }
    
    summary_file = output_dir / "baseline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}\n")
    
    # Show some examples
    print("Sample results:")
    print("-" * 70)
    for i, result in enumerate(results[:3]):
        print(f"\nPrompt: {result['prompt']}")
        print(f"Response: {result['response'][:100]}...")
        print(f"Harmful: {result['harmful']}")

if __name__ == "__main__":
    main()
