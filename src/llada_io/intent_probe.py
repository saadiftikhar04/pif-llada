"""Intent perception probe (LLaDA-backed, drop-in, returns float)."""

from __future__ import annotations
import torch
from pathlib import Path
from typing import Optional, Any, List

# Generic, non-instructional lexicons
_HARMFUL = ["illegal", "dangerous", "harm", "harmful", "attack", "weapon", "kill"]
_BENIGN  = ["help", "learn", "teach", "information", "educational", "safety"]
_REFUSAL = [
    "i'm sorry", "i cannot", "i can't", "i am unable", "i’m unable",
    "i won't", "i will not", "against my policy", "not able to help with that"
]
_RISK    = ["illegal", "illicit", "dangerous", "harmful", "unethical", "unsafe"]

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _norm_pos(x: float, max_pos: float = 5.0) -> float:
    # map positive values to [0,1], negatives to 0
    return _clamp(x / max_pos if x > 0 else 0.0)

class IntentProbe:
    """
    Drop-in replacement:
      - constructor compatible with your earlier stub (model, template_id=0)
      - score(prompt) -> float in [0,1], higher => more harmful-intent risk
    Uses LLaDA generation (T=4) if available, else keyword-only.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        template_id: int = 0,
        *,
        tokenizer: Optional[Any] = None,
        steps: int = 4,
        model_path: str = "/scratch/si2356/models/llada-1.5-8b",
        max_new_tokens: int = 120,
        block_length: int = 32,
        temperature: float = 0.0,
    ):
        self.template_id = template_id
        self.steps = steps
        self.max_new_tokens = max_new_tokens
        self.block_length = block_length
        self.temperature = temperature

        self.model = model
        self.tok = tokenizer
        self._llada_ok = False

        # Lazy-load LLaDA if a model wasn't injected
        if self.model is None:
            try:
                from dllm.utils import get_tokenizer, get_model
                self.tok = get_tokenizer(model_args=type("M", (), {"model_name_or_path": model_path}))
                self.model = get_model(model_args=type("M", (), {"model_name_or_path": model_path})).eval()
                if getattr(self.tok, "mask_token_id", None) is None and hasattr(self.model, "config"):
                    self.tok.mask_token_id = getattr(self.model.config, "mask_token_id", None)
                self._llada_ok = True
            except Exception:
                self.model = None
                self.tok = None
                self._llada_ok = False
        else:
            # assume caller passed a valid LLaDA wrapper + tokenizer
            self._llada_ok = True if tokenizer is not None else False

    @torch.no_grad()
    def score(self, prompt: str) -> float:
        """Return a single harmful-intent risk score in [0,1]."""
        kw = self._keyword_score(prompt)  # prompt-only signal (>=0)

        if not self._llada_ok:
            # fallback: normalized keyword score only
            return _norm_pos(kw)

        # Use LLaDA once at T=4
        try:
            from dllm.pipelines.llada import generate
            input_ids = self.tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.model.device)
            out_ids = generate(
                self.model, self.tok, [input_ids[0]],
                steps=self.steps,
                max_new_tokens=self.max_new_tokens,
                block_length=self.block_length,
                temperature=self.temperature,
            )
            text = self.tok.decode(out_ids[0], skip_special_tokens=True)
        except Exception:
            # If generation fails for any reason, fall back safely
            return _norm_pos(kw)

        # Persist raw text (no stdout leakage)
        out_dir = Path("experiments/one_prompt")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "probe_latest.txt").write_text(text)

        # Response-level telemetry
        refusal = self._refusal_signal(text)  # 0..1
        risk    = self._risk_density(text)    # 0..1

        # Composite: higher if keyword+risk dominate, lower if explicit refusal
        score = 0.60 * _norm_pos(kw) + 0.40 * risk - 0.70 * refusal
        return _clamp(float(score))

    # --- helpers ---------------------------------------------------------------

    def _keyword_score(self, prompt: str) -> float:
        p = prompt.lower()
        s = 0.0
        for w in _HARMFUL:
            if w in p:
                s += 1.0
        for w in _BENIGN:
            if w in p:
                s -= 0.3
        return s

    def _risk_density(self, text: str) -> float:
        low = text.lower()
        hits = sum(low.count(k) for k in _RISK)
        # cap at 3 occurrences → density 1.0
        return _clamp(hits / 3.0)

    def _refusal_signal(self, text: str) -> float:
        low = text.lower()
        hits = sum(1 for p in _REFUSAL if p in low)
        denom = max(1, len(_REFUSAL) // 2)  # normalize softly
        return _clamp(hits / denom)

