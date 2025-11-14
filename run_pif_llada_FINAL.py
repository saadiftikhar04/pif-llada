#!/usr/bin/env python
"""PiF-LLaDA - FINAL FIX: Extract new tokens only + proper generation."""
import csv
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from dllm.utils import get_tokenizer, get_model
from dllm.pipelines.llada import generate

from src.llada_io.llada_probe_generation import LLaDAGenerationProbe
from src.pif_attack.attack_llada_fixed import PiFLLaDAAttackFixed
from src.llada_io.simple_classifier import SimpleRefusalClassifier

@dataclass
class ModelArgs:
    model_name_or_path: str = "/scratch/si2356/models/llada-1.5-8b"

def load_harmbench_behaviors(csv_path):
    behaviors = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            behaviors.append({
                'behavior': row.get('Behavior', ''),
                'category': row.get('FunctionalCategory', ''),
                'behavior_id': row.get('BehaviorID', ''),
            })
    return behaviors

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    """Generate response and extract ONLY new tokens."""
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )["input_ids"].to(model.device)
    
    input_length = input_ids.shape[1]
    
    # FIX: Ensure block_length doesn't exceed max_new_tokens
    block_length = min(32, max_new_tokens)
    
    # Generate with proper parameters
    output = generate(
        model,
        tokenizer,
        [input_ids[0]],
        steps=32,  # Increased for better quality
        max_new_tokens=max_new_tokens,
        block_length=block_length,  # Use adaptive block_length
        temperature=0.7,
    )
    
    # CRITICAL FIX: Extract ONLY new tokens
    new_tokens = output[0, input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response

def main():
    print("="*70)
    print("PiF-LLaDA: FINAL FIX - Proper Token Extraction")
    print("="*70 + "\n")
    
    # Load data
    harmbench_path = Path("/scratch/si2356/pif-llada-minimal/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv")
    behaviors = load_harmbench_behaviors(harmbench_path)
    print(f"✓ Loaded {len(behaviors)} behaviors\n")
    
    # Load model
    model_args = ModelArgs()
    print("Loading LLaDA...")
    tokenizer = get_tokenizer(model_args=model_args)
    model = get_model(model_args=model_args).eval()
    if tokenizer.mask_token_id is None:
        tokenizer.mask_token_id = model.config.mask_token_id
    print(f"✓ Model on {model.device}\n")
    
    # Test generation quality first
    print("Testing generation quality...")
    test_response = generate_response(model, tokenizer, "What is the capital of France?", 20)
    print(f"  Test: '{test_response[:50]}'")
    if len(test_response.strip()) < 5:
        print("  ⚠️  WARNING: Generation quality may be poor")
    else:
        print("  ✓ Generation working\n")
    
    # Create probe
    probe = LLaDAGenerationProbe(model, tokenizer)
    print("✓ Probe ready\n")
    
    # Create attack
    attack = PiFLLaDAAttackFixed(
        probe=probe,
        max_edits=20,
        tokens_per_iter=3,
        candidates_per_token=8,
        verbose=False
    )
    print("✓ Attack ready\n")
    
    classifier = SimpleRefusalClassifier()
    
    print(f"Processing {len(behaviors)} behaviors\n")
    
    results = []
    baseline_asr = 0
    attack_asr = 0
    total_edits = 0
    
    for i, behavior_data in enumerate(tqdm(behaviors, desc="Processing")):
        behavior = behavior_data['behavior']
        
        if not behavior or len(behavior) < 5:
            continue
        
        try:
            # Baseline - EXTRACT NEW TOKENS ONLY
            baseline_response = generate_response(model, tokenizer, behavior, 100)
            baseline_harmful = classifier.is_harmful(baseline_response)
            
            if baseline_harmful:
                baseline_asr += 1
            
            # Attack
            adversarial_prompt, edits, attack_success = attack.attack(behavior)
            total_edits += edits
            
            # Test adversarial - EXTRACT NEW TOKENS ONLY
            attack_response = generate_response(model, tokenizer, adversarial_prompt, 100)
            attack_harmful = classifier.is_harmful(attack_response)
            
            if attack_harmful:
                attack_asr += 1
            
            results.append({
                'behavior_id': behavior_data.get('behavior_id', ''),
                'behavior_original': behavior,
                'behavior_adversarial': adversarial_prompt,
                'edits': edits,
                'baseline_response': baseline_response,
                'baseline_harmful': baseline_harmful,
                'attack_response': attack_response,
                'attack_harmful': attack_harmful,
                'category': behavior_data.get('category', ''),
            })
            
            # Checkpoint
            if (i + 1) % 50 == 0:
                output_dir = Path("experiments/harmbench_final")
                output_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_file = output_dir / f"checkpoint_{i+1}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump({'results': results, 'baseline_asr': baseline_asr, 'attack_asr': attack_asr}, f, indent=2)
                tqdm.write(f"✓ Checkpoint {i+1}: Baseline={baseline_asr}, Attack={attack_asr}, Avg edits={total_edits/(i+1):.1f}")
                
        except Exception as e:
            tqdm.write(f"Error {i}: {str(e)[:60]}")
            results.append({
                'behavior_id': behavior_data.get('behavior_id', ''),
                'behavior_original': behavior,
                'behavior_adversarial': behavior,
                'edits': 0,
                'baseline_response': f"ERROR: {e}",
                'baseline_harmful': False,
                'attack_response': f"ERROR: {e}",
                'attack_harmful': False,
                'category': behavior_data.get('category', ''),
            })
    
    # Save results
    output_dir = Path("experiments/harmbench_final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "pif_llada_final_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Metrics
    total = len(results)
    baseline_asr_pct = (baseline_asr / total * 100) if total > 0 else 0
    attack_asr_pct = (attack_asr / total * 100) if total > 0 else 0
    improvement = attack_asr_pct - baseline_asr_pct
    avg_edits = total_edits / total if total > 0 else 0
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS - PROPER TOKEN EXTRACTION")
    print(f"{'='*70}")
    print(f"Total behaviors:      {total}")
    print(f"Baseline ASR:         {baseline_asr}/{total} ({baseline_asr_pct:.1f}%)")
    print(f"PiF-LLaDA ASR:        {attack_asr}/{total} ({attack_asr_pct:.1f}%)")
    print(f"Improvement:          {improvement:+.1f}pp")
    print(f"Average edits:        {avg_edits:.1f}")
    print(f"\nResults: {results_file}")
    print(f"{'='*70}\n")
    
    summary = {
        'dataset': 'HarmBench (Final Fix)',
        'total_behaviors': total,
        'baseline_asr': baseline_asr,
        'baseline_asr_pct': baseline_asr_pct,
        'attack_asr': attack_asr,
        'attack_asr_pct': attack_asr_pct,
        'improvement': improvement,
        'average_edits': avg_edits,
        'steps': 32,
        'fix': 'proper_token_extraction',
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
