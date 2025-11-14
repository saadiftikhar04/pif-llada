"""LLaDA model wrapper using dllm framework."""
import torch
from dllm.pipelines.llada import generate, LLaDAModelLM
from dllm.utils import get_tokenizer, get_model

class LLaDAModel:
    def __init__(self, model_path="/scratch/si2356/models/llada-1.5-8b", num_steps=4, device="cuda"):
        """Initialize LLaDA model with dllm framework.
        
        Args:
            model_path: Path to LLaDA checkpoint
            num_steps: Number of diffusion steps (default: 4 for speed)
            device: Device to run on
        """
        self.model_path = model_path
        self.num_steps = num_steps
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"Loading LLaDA from: {model_path}")
        print(f"Diffusion steps: {num_steps}")
        print(f"Device: {self.device}")
        
        # Load tokenizer using dllm utility
        self.tokenizer = get_tokenizer(model_path)
        
        # Load model using dllm utility
        self.model = get_model(
            model_name_or_path=model_path,
            device_map=self.device if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        print("✓ LLaDA model loaded successfully!")
    
    def generate(self, prompt, max_length=512, temperature=1.0):
        """Generate response using masked diffusion.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        # Use dllm's generate function
        outputs = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=[prompt],
            num_steps=self.num_steps,
            max_new_tokens=max_length,
            temperature=temperature,
            device=self.device,
        )
        
        return outputs[0] if outputs else ""
    
    def score_prompt(self, prompt):
        """Score a prompt (for intent probe).
        
        This is a simplified scoring - in the full implementation,
        we'd use the diffusion model's likelihood.
        """
        # For now, use tokenizer to get token count as a proxy
        tokens = self.tokenizer(prompt, return_tensors="pt")
        return float(tokens['input_ids'].shape[1])

