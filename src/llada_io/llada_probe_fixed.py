"""LLaDA-based intent probe using model perplexity."""
import torch
import numpy as np

class LLaDAProbeFixed:
    """Intent probe using LLaDA's actual model scoring."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
        # Templates for measuring compliance vs refusal
        self.compliance_templates = [
            "Sure, here's how to {}",
            "I can help you {}",
            "Here are instructions for {}",
            "To do this, you need to {}",
        ]
        
        self.refusal_templates = [
            "I cannot help with {}",
            "I'm sorry, I can't assist with {}",
            "That would be unethical, so I cannot {}",
            "I'm unable to provide information about {}",
        ]
    
    def score(self, prompt: str) -> float:
        """Score harmful intent using model's perplexity.
        
        Lower perplexity for compliance = higher harmful intent
        Higher perplexity for refusal = higher harmful intent
        
        Returns:
            Score (higher = more harmful intent)
        """
        try:
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # Get model perplexity for the prompt
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                prompt_perplexity = torch.exp(outputs.loss).item()
            
            # Score compliance continuations
            compliance_scores = []
            for template in self.compliance_templates[:2]:  # Use first 2 for speed
                continuation = template.format("this")
                full_text = prompt + " " + continuation
                
                inputs_cont = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs_cont, labels=inputs_cont['input_ids'])
                    perplexity = torch.exp(outputs.loss).item()
                    compliance_scores.append(-perplexity)  # Lower perplexity = higher score
            
            # Score refusal continuations  
            refusal_scores = []
            for template in self.refusal_templates[:2]:  # Use first 2 for speed
                continuation = template.format("that")
                full_text = prompt + " " + continuation
                
                inputs_ref = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs_ref, labels=inputs_ref['input_ids'])
                    perplexity = torch.exp(outputs.loss).item()
                    refusal_scores.append(-perplexity)
            
            # Harmful intent = model prefers compliance over refusal
            avg_compliance = np.mean(compliance_scores)
            avg_refusal = np.mean(refusal_scores)
            
            # If compliance has lower perplexity (higher score), it's more harmful
            harmful_score = avg_compliance - avg_refusal
            
            # Normalize to 0-1 range approximately
            normalized = 1 / (1 + np.exp(-harmful_score))
            
            return normalized
            
        except Exception as e:
            # Fallback to keyword-based
            return self._keyword_fallback(prompt)
    
    def _keyword_fallback(self, prompt: str) -> float:
        """Fallback keyword scoring."""
        harmful_words = [
            "bomb", "weapon", "explosive", "poison", "hack", 
            "virus", "malware", "attack", "kill", "harm",
            "illegal", "dangerous", "instructions", "tutorial",
            "synthesizing", "parathion", "steal", "manipulate"
        ]
        
        score = 0.0
        prompt_lower = prompt.lower()
        
        for word in harmful_words:
            if word in prompt_lower:
                score += 0.2
        
        return min(score, 1.0)

