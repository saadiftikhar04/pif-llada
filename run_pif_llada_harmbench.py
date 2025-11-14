#!/usr/bin/env python
"""Run PiF-LLaDA attack on official HarmBench test set."""
import os
import csv
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from dllm.utils import get_tokenizer, get_model
from dllm.pipelines.llada import generate

from src.llada_io.llada_intent_probe import LLaDAIntentProbe
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
    print("PiF-LLaDA: Official HarmBench Test Set Evaluation")
    print("="*70 + "\n")
    
    # Load HarmBench behaviors
    harmbench_path = Path("/scratch/si2356/pif-llada-minimal/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv")
    
    print(f"Loading behaviors from: {harmbench_path}")
    behaviors = load_harmbench_behaviors(harmbench_path)
    print(f"✓ Loaded {len(behaviors)} behaviors from HarmBench\n")
    
    # Load LLaDA model
    model_args = ModelArgs()
    print("Loading LLaDA model...")
    tokenizer = get_tokenizer(model_args=model_args)
    model = get_model(model_args=model_args).eval()
    
    if tokenizer.mask_token_id is None:
        tokenizer.mask_token_id = model.config.mask_token_id
    
    print(f"✓ Model loaded on {model.device}\n")
    
    # Create intent probe
    print("Initializing LLaDA Intent Probe...")
    probe = LLaDAIntentProbe(model, tokenizer)
    print("✓ Probe ready\n")
    
    # Create PiF attack
    print("Initializing PiF-LLaDA Attack...")
    attack = PiFLLaDAAttack(
        probe=probe,
        max_edits=20,
        tokens_per_iter=3,
        candidates_per_token=8,
        verbose=False
    )
    print("✓ Attack ready\n")
    
    # Initialize classifier
    classifier = SimpleRefusalClassifier()
    
    print(f"Running FULL evaluation on {len(behaviors)} behaviors")
    print("Estimated time: 2-3 hours for full dataset\n")
    
    # Run experiments
    results = []
    baseline_asr = 0
    attack_asr = 0
    
    for i, behavior_data in enumerate(tqdm(behaviors, desc="Processing behaviors")):
        behavior = behavior_data['behavior']
        
        # Skip empty behaviors
        if not behavior or len(behavior) < 5:
            continue
        
        try:
            # Step 1: Baseline test
            input_ids = tokenizer(
                behavior,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=512
            )["input_ids"].to(model.device)
            
            output_baseline = generate(
                model,
                tokenizer,
                [input_ids[0]],
                steps=4,
                max_new_tokens=100,
                block_length=32,
                temperature=0.0,
            )
            
            baseline_response = tokenizer.decode(output_baseline[0], skip_special_tokens=True)
            baseline_harmful = classifier.is_harmful(baseline_response)
            
            if baseline_harmful:
                baseline_asr += 1
            
            # Step 2: Run PiF-LLaDA attack
            adversarial_prompt, edits, attack_success = attack.attack(behavior)
            
            # Step 3: Test adversarial prompt
            input_ids_adv = tokenizer(
                adversarial_prompt,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=512
            )["input_ids"].to(model.device)
            
            output_attack = generate(
                model,
                tokenizer,
                [input_ids_adv[0]],
                steps=4,
                max_new_tokens=100,
                block_length=32,
                temperature=0.0,
            )
            
            attack_response = tokenizer.decode(output_attack[0], skip_special_tokens=True)
            attack_harmful = classifier.is_harmful(attack_response)
            
            if attack_harmful:
                attack_asr += 1
            
            # Save result
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
                'semantic_category': behavior_data.get('semantic_category', ''),
            })
            
            # Save checkpoint every 50 behaviors
            if (i + 1) % 50 == 0:
                output_dir = Path("experiments/harmbench_full")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_file = output_dir / f"checkpoint_{i+1}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'results': results,
                        'baseline_asr': baseline_asr,
                        'attack_asr': attack_asr,
                        'processed': i + 1,
                    }, f, indent=2)
                
                tqdm.write(f"Checkpoint saved: {i+1} behaviors processed")
                
        except Exception as e:
            tqdm.write(f"Error on behavior {i}: {str(e)[:80]}")
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
                'semantic_category': behavior_data.get('semantic_category', ''),
            })
    
    # Save final results
    output_dir = Path("experiments/harmbench_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "pif_llada_harmbench_full.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compute metrics
    total = len(results)
    baseline_asr_pct = (baseline_asr / total * 100) if total > 0 else 0
    attack_asr_pct = (attack_asr / total * 100) if total > 0 else 0
    improvement = attack_asr_pct - baseline_asr_pct
    
    avg_edits = sum(r['edits'] for r in results) / total if total > 0 else 0
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS - Official HarmBench Test Set")
    print(f"{'='*70}")
    print(f"Total behaviors:      {total}")
    print(f"Baseline ASR:         {baseline_asr}/{total} ({baseline_asr_pct:.1f}%)")
    print(f"PiF-LLaDA ASR:        {attack_asr}/{total} ({attack_asr_pct:.1f}%)")
    print(f"Improvement:          {improvement:+.1f}pp")
    print(f"Average edits:        {avg_edits:.1f}")
    print(f"\nResults saved to: {results_file}")
    print(f"{'='*70}\n")
    
    # Save summary
    summary = {
        'dataset': 'HarmBench Text Test Set',
        'total_behaviors': total,
        'baseline_asr': baseline_asr,
        'baseline_asr_pct': baseline_asr_pct,
        'attack_asr': attack_asr,
        'attack_asr_pct': attack_asr_pct,
        'improvement': improvement,
        'average_edits': avg_edits,
        'model': model_args.model_name_or_path,
        'diffusion_steps': 4,
        'max_edits': 20,
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}\n")
    
    # Category breakdown
    categories = {}
    for r in results:
        cat = r.get('category', 'Unknown')
        if cat not in categories:
            categories[cat] = {'total': 0, 'baseline': 0, 'attack': 0}
        categories[cat]['total'] += 1
        if r['baseline_harmful']:
            categories[cat]['baseline'] += 1
        if r['attack_harmful']:
            categories[cat]['attack'] += 1
    
    print("Category-wise breakdown:")
    print("-" * 70)
    for cat, stats in sorted(categories.items()):
        if stats['total'] > 0:
            b_asr = stats['baseline'] / stats['total'] * 100
            a_asr = stats['attack'] / stats['total'] * 100
            print(f"{cat:25} | Baseline: {b_asr:5.1f}% | Attack: {a_asr:5.1f}% | Δ: {a_asr-b_asr:+5.1f}pp")

if __name__ == "__main__":
    main()
