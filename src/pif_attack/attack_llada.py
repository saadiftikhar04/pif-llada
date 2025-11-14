"""PiF-LLaDA Attack with LLaDA-based intent probe."""
import numpy as np
from .perceived_importance_llada import PIComputerLLaDA
from .synonyms import SynonymGenerator

class PiFLLaDAAttack:
    """PiF attack using LLaDA intent probe."""
    
    def __init__(self, probe, max_edits=20, tokens_per_iter=3, 
                 candidates_per_token=8, verbose=True):
        self.probe = probe
        self.pi_computer = PIComputerLLaDA(probe, mode="delete")
        self.syn_gen = SynonymGenerator(max_candidates=candidates_per_token)
        self.max_edits = max_edits
        self.tokens_per_iter = tokens_per_iter
        self.verbose = verbose
    
    def attack(self, prompt: str):
        """Execute PiF-LLaDA attack.
        
        Args:
            prompt: Original harmful prompt
            
        Returns:
            (adversarial_prompt, num_edits, success)
        """
        current_prompt = prompt
        edits_made = 0
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"PiF-LLaDA Attack")
            print(f"{'='*70}")
            print(f"Original: {prompt}")
            print(f"Initial Score: {self.probe.score(prompt):.3f}")
            print(f"{'='*70}\n")
        
        while edits_made < self.max_edits:
            # Compute PI for current prompt
            pi_scores = self.pi_computer.compute(current_prompt, verbose=False)
            
            # Sort by PI (ascending - low PI first)
            pi_scores.sort(key=lambda x: x[2])
            
            # Select lowest-PI neutral tokens (skip harmful keywords)
            candidates = self._select_neutral_tokens(pi_scores, current_prompt)[:self.tokens_per_iter]
            
            if not candidates:
                if self.verbose:
                    print("No neutral tokens to modify.")
                break
            
            if self.verbose:
                print(f"Iteration {edits_made + 1}:")
                print(f"  Low-PI tokens: {[(tok, f'{pi:.3f}') for _, tok, pi in candidates]}")
            
            # Try synonyms
            best_prompt = current_prompt
            best_score = self.probe.score(current_prompt)
            
            tokens = current_prompt.split()
            
            for idx, token, pi in candidates:
                synonyms = self.syn_gen.get_synonyms(token)
                
                if self.verbose and synonyms:
                    print(f"  Trying synonyms for '{token}': {synonyms[:3]}")
                
                for syn in synonyms:
                    # Create candidate
                    candidate_tokens = tokens.copy()
                    candidate_tokens[idx] = syn
                    candidate = " ".join(candidate_tokens)
                    
                    # Score
                    score = self.probe.score(candidate)
                    
                    # Lower score = more neutral (objective: minimize harmful intent)
                    if score < best_score - 0.1:
                        best_score = score
                        best_prompt = candidate
            
            if best_prompt == current_prompt:
                if self.verbose:
                    print("  No improvement found.\n")
                break
            
            current_prompt = best_prompt
            tokens = current_prompt.split()  # Update for next iteration
            edits_made += 1
            
            if self.verbose:
                print(f"  → {current_prompt}")
                print(f"  → Score: {best_score:.3f}\n")
        
        final_score = self.probe.score(current_prompt)
        initial_score = self.probe.score(prompt)
        success = final_score < initial_score * 0.7  # 30% reduction in harmful intent
        
        if self.verbose:
            print(f"{'='*70}")
            print(f"Attack Complete")
            print(f"{'='*70}")
            print(f"Original:     {prompt}")
            print(f"Adversarial:  {current_prompt}")
            print(f"Edits:        {edits_made}")
            print(f"Score:        {initial_score:.3f} → {final_score:.3f}")
            print(f"Success:      {success}")
            print(f"{'='*70}\n")
        
        return current_prompt, edits_made, success
    
    def _select_neutral_tokens(self, pi_scores, prompt):
        """Select neutral (non-harmful) tokens with low PI."""
        harmful_keywords = {
            "bomb", "weapon", "explosive", "hack", "virus",
            "poison", "kill", "attack", "harm", "illegal"
        }
        
        neutral_candidates = []
        for idx, token, pi in pi_scores:
            if token.lower() not in harmful_keywords:
                neutral_candidates.append((idx, token, pi))
        
        return neutral_candidates

