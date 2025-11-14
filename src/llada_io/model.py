"""Model wrapper - uses LLaDA if available, otherwise placeholder."""
import os
import torch

try:
    from .llada_model import LLaDAModel as _LLaDAModel
    LLADA_AVAILABLE = True
except ImportError:
    LLADA_AVAILABLE = False
    _LLaDAModel = None

class LLaDAModel:
    """Wrapper that uses real LLaDA if available."""
    
    def __init__(self, model_path=None, device="cuda", num_steps=4):
        if model_path is None:
            model_path = os.getenv("LLADA_MODEL_PATH", "/scratch/si2356/models/llada-1.5-8b")
        
        if LLADA_AVAILABLE and os.path.exists(model_path):
            print("Using real LLaDA model")
            self.model = _LLaDAModel(model_path, num_steps, device)
            self.tokenizer = self.model.tokenizer
        else:
            print("Warning: LLaDA not available, using placeholder")
            self.model = None
            self.tokenizer = None
    
    def generate(self, prompt, max_length=512):
        if self.model:
            return self.model.generate(prompt, max_length)
        return f"[Placeholder response to: {prompt}]"
    
    def get_logits(self, prompt):
        if self.model:
            # Simplified - real implementation would return logits
            return torch.randn(1, 10)
        return torch.randn(1, 10)
    
    def score_prompt(self, prompt):
        if self.model:
            return self.model.score_prompt(prompt)
        return 0.0
