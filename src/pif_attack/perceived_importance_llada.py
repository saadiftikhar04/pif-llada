"""Compute Perceived Importance using LLaDA intent probe."""
import numpy as np
from tqdm import tqdm

class PIComputerLLaDA:
    """Compute PI scores using leave-one-out with LLaDA probe."""
    
    def __init__(self, probe, mode="delete"):
        self.probe = probe
        self.mode = mode
    
    def compute(self, prompt: str, verbose=False) -> list:
        """Compute PI for each token.
        
        Args:
            prompt: Input prompt
            verbose: Show progress
            
        Returns:
            List of (index, token, pi_score) tuples
        """
        tokens = prompt.split()
        base_score = self.probe.score(prompt)
        
        pi_scores = []
        iterator = tqdm(range(len(tokens)), desc="Computing PI") if verbose else range(len(tokens))
        
        for i in iterator:
            # Occlude token i
            if self.mode == "mask":
                occluded = " ".join(tokens[:i] + ["[MASK]"] + tokens[i+1:])
            elif self.mode == "delete":
                occluded = " ".join(tokens[:i] + tokens[i+1:])
            else:  # neutral
                occluded = " ".join(tokens[:i] + ["something"] + tokens[i+1:])
            
            # Score occluded prompt
            occluded_score = self.probe.score(occluded)
            
            # PI = absolute difference
            pi = abs(base_score - occluded_score)
            
            pi_scores.append((i, tokens[i], pi))
        
        return pi_scores

