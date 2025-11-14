#!/usr/bin/env python
"""Run PiF-LLaDA attack on HarmBench dataset."""
import os
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

def main():
    print("="*70)
    print("PiF-LLaDA: Complete Attack Pipeline")
    print("="*70 + "\n")
    
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
        verbose=True
    )
    print("✓ Attack ready\n")
    
    # Load HarmBench prompts
    prompt_file = Path("data/harmbench_harmful.jsonl")
    if not prompt_file.exists():
        print(f"Error: {prompt_file} not found!")
        return
    
    prompts = []
    with open(prompt_file) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    
    print(f"Loaded {len(prompts)} prompts from HarmBench\n")
    
    # Limit to first 10 for now (full run takes longer)
    prompts = prompts[:10]
    print(f"Running on first {len(prompts)} prompts for testing\n")
    
    # Initialize classifier
    classifier = SimpleRefusalClassifier()
    
    # Run experiments
    results = []
    baseline_asr = 0
    attack_asr = 0
    
    for i, prompt_data in enumerate(prompts):
        print(f"\n{'#'*70}")
        print(f"Prompt {i+1}/{len(prompts)}")
        print(f"{'#'*70}")
        
        original_prompt = prompt_data['prompt']
        
        # Step 1: Test baseline (original prompt)
        print("\n[BASELINE] Testing original prompt...")
        try:
            input_ids = tokenizer(
                original_prompt,
                return_tensors="pt",
                add_special_tokens=True
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
            
            print(f"Baseline Response: {baseline_response[:100]}...")
            print(f"Baseline Harmful: {baseline_harmful}")
            
            if baseline_harmful:
                baseline_asr += 1
                
        except Exception as e:
            print(f"Baseline error: {e}")
            baseline_response = f"ERROR: {e}"
            baseline_harmful = False
        
        # Step 2: Run PiF-LLaDA attack
        print("\n[ATTACK] Running PiF-LLaDA...")
        adversarial_prompt, edits, attack_success = attack.attack(original_prompt)
        
        # Step 3: Test adversarial prompt
        print("\n[EVALUATION] Testing adversarial prompt...")
        try:
            input_ids_adv = tokenizer(
                adversarial_prompt,
                return_tensors="pt",
                add_special_tokens=True
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
            
            print(f"Attack Response: {attack_response[:100]}...")
            print(f"Attack Harmful: {attack_harmful}")
            
            if attack_harmful:
                attack_asr += 1
                
        except Exception as e:
            print(f"Attack error: {e}")
            attack_response = f"ERROR: {e}"
            attack_harmful = False
        
        # Save result
        results.append({
            'prompt_original': original_prompt,
            'prompt_adversarial': adversarial_prompt,
            'edits': edits,
            'baseline_response': baseline_response,
            'baseline_harmful': baseline_harmful,
            'attack_response': attack_response,
            'attack_harmful': attack_harmful,
            'behavior': prompt_data.get('behavior', ''),
            'category': prompt_data.get('category', ''),
        })
    
    # Save results
    output_dir = Path("experiments/pif_llada")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "pif_llada_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compute final metrics
    total = len(results)
    baseline_asr_pct = (baseline_asr / total * 100) if total > 0 else 0
    attack_asr_pct = (attack_asr / total * 100) if total > 0 else 0
    improvement = attack_asr_pct - baseline_asr_pct
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Total prompts:        {total}")
    print(f"Baseline ASR:         {baseline_asr}/{total} ({baseline_asr_pct:.1f}%)")
    print(f"PiF-LLaDA ASR:        {attack_asr}/{total} ({attack_asr_pct:.1f}%)")
    print(f"Improvement:          {improvement:+.1f}%")
    print(f"\nResults saved to: {results_file}")
    print(f"{'='*70}\n")
    
    # Save summary
    summary = {
        'total_prompts': total,
        'baseline_asr': baseline_asr,
        'baseline_asr_pct': baseline_asr_pct,
        'attack_asr': attack_asr,
        'attack_asr_pct': attack_asr_pct,
        'improvement': improvement,
        'model': model_args.model_name_or_path,
        'diffusion_steps': 4,
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
