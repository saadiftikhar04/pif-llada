#!/usr/bin/env python
"""
Standard LLM Implementation  
Reusable class for causal language models (Falcon3, LLaMA, etc.)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, List
import time
import importlib.util


class LLM:
    """
    Standard Causal Language Model Wrapper
    
    Usage:
        llm = LLM("tiiuae/Falcon3-7B-Instruct")
        response = llm.infer("What is the capital of France?")
    """
    
    def __init__(
        self, 
        model_name: str,
        device: str = "auto",
        load_in_8bit: bool = True,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize LLM
        
        Args:
            model_name: HuggingFace model ID
            device: Device placement ('auto', 'cuda', 'cpu')
            load_in_8bit: Use 8-bit quantization if available
            system_prompt: Default system prompt for chat models
        """
        print(f"Loading model: {model_name}...")
        
        self.model_name = model_name
        self.system_prompt = system_prompt or self._get_default_system_prompt(model_name)
        
        # Robust Model Loading Logic (matching Step 2)
        use_8bit = False
        if load_in_8bit and importlib.util.find_spec("bitsandbytes") is not None:
            try:
                import bitsandbytes as bnb
                use_8bit = True
            except ImportError:
                pass
        
        model_loaded = False
        if use_8bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device,
                    quantization_config=quantization_config,
                    output_hidden_states=True,
                    trust_remote_code=True
                )
                model_loaded = True
                print("Successfully loaded in 8-bit.")
            except Exception:
                print("8-bit loading failed. Falling back to Float16.")
        
        # Fallback to Float16
        if not model_loaded:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float16,
                output_hidden_states=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("Loaded in Float16.")
        
        self.model.eval()
        
        # Tokenizer setup (matching Step 2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Handle missing special tokens
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.mask_token_id = self.tokenizer.mask_token_id
        self.device = self.model.device
        print(f"Model loaded and initialized.\n")
    
    def _get_default_system_prompt(self, model_name: str) -> str:
        """Get default system prompt based on model"""
        if "falcon" in model_name.lower():
            return "You are a helpful friendly assistant Falcon3 from TII, try to follow instructions as much as possible."
        elif "llama" in model_name.lower():
            return "You are a helpful assistant."
        else:
            return "You are a helpful assistant."
    
    def infer(
        self, 
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 1.0,
        do_sample: bool = False,
        system_prompt: Optional[str] = None,
        return_time: bool = False
    ) -> str:
        """
        Generate response using standard LLM generation
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Use sampling vs greedy
            system_prompt: Override default system prompt
            return_time: Return (response, time) tuple
        
        Returns:
            Generated text response (or tuple if return_time=True)
        """
        system = system_prompt or self.system_prompt
        
        # Format with chat template if available (matching Step 2)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            text = f"User: {prompt}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        elapsed = time.time() - start_time
        
        # Decode only new tokens (matching Step 2)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0].strip()
        
        if return_time:
            return response, elapsed
        return response
    
    def batch_infer(
        self,
        prompts: List[str],
        max_new_tokens: int = 300,
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens per generation
            system_prompt: Override default system prompt
        
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            response = self.infer(
                prompt, 
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt
            )
            responses.append(response)
        return responses
    
    def get_logits(self, text: str, return_hidden_states: bool = False):
        """
        Get model logits for input text
        
        Args:
            text: Input text
            return_hidden_states: Return hidden states
        
        Returns:
            Logits tensor (and optionally hidden states)
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=return_hidden_states)
        
        if return_hidden_states:
            return outputs.logits, outputs.hidden_states
        return outputs.logits
    
    def __repr__(self):
        return f"LLM(model={self.model_name}, device={self.device})"
