#!/usr/bin/env python
import argparse, json, time
from pathlib import Path

import torch
from dllm.utils import get_tokenizer, get_model
from dllm.pipelines.llada import generate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to HarmBench JSONL (harmful split).")
    ap.add_argument("--out", default="experiments/asr/completions.jsonl", help="Where to write completions JSONL.")
    ap.add_argument("--model", default="/scratch/si2356/models/llada-1.5-8b")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="-1 = till EOF")
    args = ap.parse_args()

    data_fp = Path(args.data)
    assert data_fp.exists() and data_fp.stat().st_size > 0, f"{data_fp} missing or empty"

    out_fp = Path(args.out); out_fp.parent.mkdir(parents=True, exist_ok=True)

    # Load model/tokenizer once (LLaDA)
    tok = get_tokenizer(model_args=type("M", (), {"model_name_or_path": args.model}))
    model = get_model(model_args=type("M", (), {"model_name_or_path": args.model})).eval()
    if tok.mask_token_id is None:
        tok.mask_token_id = getattr(model.config, "mask_token_id", None)

    lines = data_fp.read_text().splitlines()
    N = len(lines)
    lo = max(0, args.start); hi = N if args.end < 0 else min(N, args.end)

    with out_fp.open("w") as fout:
        for i in range(lo, hi):
            row = json.loads(lines[i])
            prompt = row.get("prompt", row.get("text", ""))

            # Generate once with LLaDA (T=4 dev default)
            t0 = time.time()
            input_ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(model.device)
            out_ids = generate(
                model, tok, [input_ids[0]],
                steps=args.steps, max_new_tokens=args.max_new_tokens,
                block_length=args.block_length, temperature=args.temperature
            )
            gen = tok.decode(out_ids[0], skip_special_tokens=True)
            _ = time.time() - t0  # timing available if you want to log later

            # HarmBench-completions format: one JSON per line with prompt/response
            fout.write(json.dumps({
                "prompt": prompt,
                "response": gen,
                "behavior": row.get("behavior", ""),
                "category": row.get("category", ""),
                "idx": i
            }) + "\n")

    print(f"[GEN] wrote completions for {hi-lo} items → {out_fp}")

if __name__ == "__main__":
    main()

