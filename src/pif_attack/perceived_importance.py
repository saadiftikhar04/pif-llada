"""Compute Perceived Importance via occlusion."""
import numpy as np

class PIComputer:
    def __init__(self, probe, mode="delete"):
        self.probe = probe
        self.mode = mode
    
    def compute(self, prompt):
        """Compute PI scores for each token."""
        tokens = prompt.split()
        base_score = self.probe.score(prompt)
        
        pi_scores = []
        for i in range(len(tokens)):
            # Occlude token
            if self.mode == "mask":
                occluded = " ".join(tokens[:i] + ["[MASK]"] + tokens[i+1:])
            elif self.mode == "delete":
                occluded = " ".join(tokens[:i] + tokens[i+1:])
            else:  # neutral
                occluded = " ".join(tokens[:i] + ["something"] + tokens[i+1:])
            
            occluded_score = self.probe.score(occluded)
            pi = abs(base_score - occluded_score)
            pi_scores.append((i, tokens[i], pi))
        
        return pi_scores
