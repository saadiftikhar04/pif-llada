"""LLaDA-based Intent Probe using diffusion model scoring."""
import torch
import numpy as np

class LLaDAIntentProbe:
    """Intent probe using LLaDA's internal representations."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
        # Lexicons for intent classification
        self.harmful_words = [
            "bomb", "weapon", "explosive", "poison", "hack", 
            "virus", "malware", "attack", "kill", "harm",
            "illegal", "dangerous", "instructions", "tutorial"
        ]
        
        self.benign_words = [
            "help", "information", "learn", "educational",
            "legal", "safe", "ethical", "proper"
        ]
    
    def score(self, prompt: str) -> float:
        """Score harmful intent of prompt.
        
        Higher score = more harmful intent detected.
        Uses both keyword matching and token embeddings.
        
        Args:
            prompt: Input text
            
        Returns:
            Intent score (higher = more harmful)
        """
        # Keyword-based component (fast)
        keyword_score = self._keyword_score(prompt)
        
        # Token-level component (using embeddings)
        embedding_score = self._embedding_score(prompt)
        
        # Weighted combination
        final_score = 0.7 * keyword_score + 0.3 * embedding_score
        
        return final_score
    
    def _keyword_score(self, prompt: str) -> float:
        """Keyword-based scoring."""
        score = 0.0
        prompt_lower = prompt.lower()
        
        for word in self.harmful_words:
            if word in prompt_lower:
                score += 1.0
        
        for word in self.benign_words:
            if word in prompt_lower:
                score -= 0.5
        
        return max(0.0, score)
    
    def _embedding_score(self, prompt: str) -> float:
        """Embedding-based scoring using model."""
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True
            ).to(self.device)
            
            # Get embeddings (use first layer)
            with torch.no_grad():
                embeddings = self.model.model.embed_tokens(inputs['input_ids'])
                
                # Compute mean embedding
                mean_emb = embeddings.mean(dim=1)
                
                # Simple heuristic: norm of embedding as proxy for "intensity"
                score = mean_emb.norm().item() / 10.0
            
            return score
            
        except Exception as e:
            # Fallback to 0 if error
            return 0.0

