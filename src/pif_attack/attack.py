"""PiF-LLaDA attack implementation."""
import numpy as np
from .perceived_importance import PIComputer
from .synonyms import SynonymGenerator

class PiFAttack:
    def __init__(self, model, probe, max_edits=10, tokens_per_iter=2):
        self.model = model
        self.probe = probe
        self.pi_computer = PIComputer(probe)
        self.syn_gen = SynonymGenerator()
        self.max_edits = max_edits
        self.tokens_per_iter = tokens_per_iter
    
    def attack(self, prompt):
        """Execute PiF attack."""
        current_prompt = prompt
        edits_made = 0
        
        print(f"\n{'='*60}")
        print(f"Original: {prompt}")
        print(f"Score: {self.probe.score(prompt):.3f}")
        print(f"{'='*60}\n")
        
        while edits_made < self.max_edits:
            # Compute PI
            pi_scores = self.pi_computer.compute(current_prompt)
            pi_scores.sort(key=lambda x: x[2])  # Sort by PI (ascending)
            
            # Select low-PI tokens
            candidates = pi_scores[:self.tokens_per_iter]
            
            print(f"Iteration {edits_made + 1}:")
            print(f"  Low-PI tokens: {[(tok, f'{pi:.3f}') for _, tok, pi in candidates]}")
            
            # Try synonyms
            best_prompt = current_prompt
            best_score = self.probe.score(current_prompt)
            
            for idx, token, pi in candidates:
                synonyms = self.syn_gen.get_synonyms(token)
                print(f"  Synonyms for '{token}': {synonyms[:3]}")
                
                tokens = current_prompt.split()
                for syn in synonyms:
                    candidate = " ".join(tokens[:idx] + [syn] + tokens[idx+1:])
                    score = self.probe.score(candidate)
                    
                    # Lower score = more neutral/benign
                    if score < best_score - 0.1:  # Small improvement threshold
                        best_score = score
                        best_prompt = candidate
            
            if best_prompt == current_prompt:
                print("  No improvement found.\n")
                break
            
            current_prompt = best_prompt
            edits_made += 1
            print(f"  → {current_prompt}")
            print(f"  → Score: {best_score:.3f}\n")
        
        return current_prompt, edits_made
