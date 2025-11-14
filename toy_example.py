#!/usr/bin/env python
"""Toy example: Single prompt PiF attack."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.llada_io.model import LLaDAModel
from src.llada_io.intent_probe import IntentProbe
from src.pif_attack.attack import PiFAttack

def main():
    print("\n" + "="*60)
    print("PiF-LLaDA TOY EXAMPLE")
    print("="*60)
    
    # Check for model path
    model_path = os.getenv("LLADA_MODEL_PATH")
    if model_path and os.path.exists(model_path):
        print(f"\nLoading LLaDA from: {model_path}")
        model = LLaDAModel(model_path, num_steps=4)
    else:
        print("\nNo LLaDA model found, using placeholder probe")
        model = None
    
    # Create probe
    probe = IntentProbe(model, template_id=0)
    
    # Load prompt
    with open("data/toy_prompt.txt") as f:
        prompt = f.read().strip()
    
    # Create attack
    attack = PiFAttack(model, probe, max_edits=10, tokens_per_iter=2)
    
    # Execute
    adversarial, edits = attack.attack(prompt)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Original:     {prompt}")
    print(f"Adversarial:  {adversarial}")
    print(f"Edits made:   {edits}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
