import random, numpy as np, torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

def set_seed(seed: int):
    if seed is None: return
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def apply_chat_template(tokenizer: PreTrainedTokenizerBase, user_text: str, system_text: str|None=None):
    msgs = []
    if system_text:
        msgs.append({"role":"system","content":system_text})
    msgs.append({"role":"user","content":user_text})
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return text

def ensure_tokens(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

def llada_chat_generate(model, tokenizer, user_text, system_text=None, seed=None,
                        max_new_tokens=256, temperature=0.2, top_p=0.9,
                        repetition_penalty=1.1, no_repeat_ngram_size=3):
    set_seed(seed)
    ensure_tokens(model, tokenizer)
    prompt = apply_chat_template(tokenizer, user_text, system_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.batch_decode(gen_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    return out
