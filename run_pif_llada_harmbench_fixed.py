#!/usr/bin/env python
"""Run PiF-LLaDA with FIXED generation-based probe."""
import os
import csv
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from dllm.utils import get_tokenizer, get_model
from dllm.pipelines.llada import generate

from src.llada_io.llada_probe_generation import LLaDAGenerationProbe
from src.pif_attack.attack_llada import PiFLLaDAAttack
from src.llada_io.simple_classifier import SimpleRefusalClassifier

@dataclass
class ModelArgs:
    model_name_or_path: str = "/scratch/si2356/models/llada-1.5-8b"

def load_harmbench_behaviors(csv_path):
    """Load behaviors from HarmBench CSV."""
    behaviors = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            behaviors.append({
                'behavior': row.get('Behavior', ''),
                'category': row.get('FunctionalCategory', ''),
                'semantic_category': row.get('SemanticCategory', ''),
                'behavior_id': row.get('BehaviorID', ''),
            })
    return behaviors

def main():
    print("="*70)
    print("PiF-LLaDA: FIXED - Full HarmBench Evaluation")
    print("Using Generation-Based Intent Probe")
    print("="*70 + "\n")
    
    # Load HarmBench
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
    print(f"✓ Model loaded on {model.device}\n")
    
    # Use FIXED generation-based probe
    print("Initializing Generation-Based Probe (FIXED)...")
    probe = LLaDAGenerationProbe(model, tokenizer)
    print("✓ Probe ready\n")
    
    # Create attack
    attack = PiFLLaDAAttack(
        probe=probe,
        max_edits=20,
        tokens_per_iter=3,
        candidates_per_token=8,
        verbose=False
    )
    print("✓ Attack ready\n")
    
    # Classifier
    classifier = SimpleRefusalClassifier()
    
    print(f"Processing {len(behaviors)} behaviors with FIXED probe\n")
    
    # Run experiments
    results = []
    baseline_asr = 0
    attack_asr = 0
    total_edits = 0
    
    for i, behavior_data in enumerate(tqdm(behaviors, desc="Processing")):
        behavior = behavior_data['behavior']
        
        if not behavior or len(behavior) < 5:
            continue
        
        try:
            # Baseline
            input_ids = tokenizer(
                behavior,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )["input_ids"].to(model.device)
            
            output_baseline = generate(
                model, tokenizer, [input_ids[0]],
                steps=4, max_new_tokens=100,
                block_length=32, temperature=0.0,
            )
            
            baseline_response = tokenizer.decode(output_baseline[0], skip_special_tokens=True)
            baseline_harmful = classifier.is_harmful(baseline_response)
            
            if baseline_harmful:
                baseline_asr += 1
            
            # Attack
            adversarial_prompt, edits, attack_success = attack.attack(behavior)
            total_edits += edits
            
            # Test adversarial
            input_ids_adv = tokenizer(
                adversarial_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )["input_ids"].to(model.device)
            
            output_attack = generate(
                model, tokenizer, [input_ids_adv[0]],
                steps=4, max_new_tokens=100,
                block_length=32, temperature=0.0,
            )
            
            attack_response = tokenizer.decode(output_attack[0], skip_special_tokens=True)
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
            
            # Checkpoint every 50
            if (i + 1) % 50 == 0:
                output_dir = Path("experiments/harmbench_fixed")
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
    
    # Final results
    output_dir = Path("experiments/harmbench_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "pif_llada_fixed_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Metrics
    total = len(results)
    baseline_asr_pct = (baseline_asr / total * 100) if total > 0 else 0
    attack_asr_pct = (attack_asr / total * 100) if total > 0 else 0
    improvement = attack_asr_pct - baseline_asr_pct
    avg_edits = total_edits / total if total > 0 else 0
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS - FIXED PROBE")
    print(f"{'='*70}")
    print(f"Total behaviors:      {total}")
    print(f"Baseline ASR:         {baseline_asr}/{total} ({baseline_asr_pct:.1f}%)")
    print(f"PiF-LLaDA ASR:        {attack_asr}/{total} ({attack_asr_pct:.1f}%)")
    print(f"Improvement:          {improvement:+.1f}pp")
    print(f"Average edits:        {avg_edits:.1f}")
    print(f"\nResults: {results_file}")
    print(f"{'='*70}\n")
    
    # Summary
    summary = {
        'dataset': 'HarmBench (Fixed Probe)',
        'total_behaviors': total,
        'baseline_asr': baseline_asr,
        'baseline_asr_pct': baseline_asr_pct,
        'attack_asr': attack_asr,
        'attack_asr_pct': attack_asr_pct,
        'improvement': improvement,
        'average_edits': avg_edits,
        'probe_type': 'generation_based',
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
