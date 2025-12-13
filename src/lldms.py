#!/usr/bin/env python
"""
LLaDA Diffusion Model Implementation
Reusable class for all LLaDA experiments
"""

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List
import time


def add_gumbel_noise(logits, temperature):
    """Official LLaDA Gumbel noise implementation"""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Official LLaDA transfer token calculation"""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, 
        device=mask_index.device, 
        dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def llada_generate(model, prompt, attention_mask=None, steps=128, gen_length=128, 
                   block_length=32, temperature=0., cfg_scale=0., 
                   remasking='low_confidence', mask_id=126336):
    """
    Official LLaDA generation function
    
    Args:
        model: LLaDA model
        prompt: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask
        steps: Number of denoising steps (default 128)
        gen_length: Length of generated sequence (default 128)
        block_length: Block size for generation (default 32)
        temperature: Temperature for Gumbel noise (default 0)
        cfg_scale: Classifier-free guidance scale (default 0)
        remasking: Remasking strategy ('low_confidence' or 'random')
        mask_id: Mask token ID (default 126336 for LLaDA)
    
    Returns:
        Generated token IDs [batch_size, seq_len + gen_length]
    """
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length), 
        mask_id, 
        dtype=torch.long
    ).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones(
                (prompt.shape[0], gen_length), 
                dtype=attention_mask.dtype, 
                device=model.device
            )
        ], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length: 
               prompt.shape[1] + (num_block + 1) * block_length:] == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)
            
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


class LLaDA:
    """
    LLaDA Diffusion Model Wrapper
    
    Usage:
        llada = LLaDA("GSAI-ML/LLaDA-1.5")
        response = llada.infer("What is the capital of France?")
    """
    
    def __init__(self, model_name: str = "GSAI-ML/LLaDA-1.5", device: str = "auto"):
        """
        Initialize LLaDA model
        
        Args:
            model_name: HuggingFace model ID
            device: Device placement ('auto', 'cuda', 'cpu')
        """
        print(f"Loading LLaDA: {model_name}...")
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
            output_hidden_states=True
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Ensure left padding for LLaDA
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'
        
        # Setup mask token
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        
        self.mask_token_id = 126336  # LLaDA's mask token
        self.device = self.model.device
        
        print(f"✓ LLaDA loaded on {self.device}\n")
    
    def infer(
        self, 
        prompt: str, 
        max_new_tokens: int = 128,
        steps: int = 128,
        block_length: int = 32,
        temperature: float = 0.,
        cfg_scale: float = 0.,
        return_time: bool = False
    ) -> str:
        """
        Generate response using LLaDA
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate (default 128)
            steps: Number of denoising steps (default 128)
            block_length: Block size (default 32)
            temperature: Gumbel temperature (default 0)
            cfg_scale: Classifier-free guidance (default 0)
            return_time: Return (response, time) tuple
        
        Returns:
            Generated text response (or tuple if return_time=True)
        """
        # Apply chat template
        message = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            message, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Encode
        encoded = self.tokenizer(
            [formatted], 
            add_special_tokens=False, 
            padding=True, 
            return_tensors="pt"
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Generate
        start_time = time.time()
        output = llada_generate(
            self.model,
            input_ids,
            attention_mask,
            steps=steps,
            gen_length=max_new_tokens,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            mask_id=self.mask_token_id
        )
        elapsed = time.time() - start_time
        
        # Decode (only new tokens)
        response = self.tokenizer.batch_decode(
            output[:, input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0].strip()
        
        if return_time:
            return response, elapsed
        return response
    
    def batch_infer(
        self, 
        prompts: List[str],
        max_new_tokens: int = 128,
        steps: int = 128,
        block_length: int = 32
    ) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens per generation
            steps: Denoising steps
            block_length: Block size
        
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            response = self.infer(
                prompt, 
                max_new_tokens=max_new_tokens,
                steps=steps,
                block_length=block_length
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
        return f"LLaDA(device={self.device}, mask_id={self.mask_token_id})"
