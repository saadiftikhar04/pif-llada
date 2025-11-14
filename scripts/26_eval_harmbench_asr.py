#!/usr/bin/env python
import argparse, json, subprocess, sys
from pathlib import Path

def run_harmbench_evaluator(hb_root: Path, completions: Path, out_json: Path) -> bool:
    """
    Try a few common CLI arg names for HarmBench/evaluate_completions.py.
    Returns True on success, False otherwise.
    """
    py = sys.executable
    eval_py = hb_root / "evaluate_completions.py"
    if not eval_py.exists():
        print(f"[EVAL] {eval_py} not found. Is HarmBench cloned?")
        return False

    # candidate arg patterns seen in similar repos
    candidate_cmds = [
        [py, str(eval_py), "--completions_path", str(completions), "--output_path", str(out_json)],
        [py, str(eval_py), "--completions", str(completions), "--output", str(out_json)],
        [py, str(eval_py), "--input", str(completions), "--output", str(out_json)],
        [py, str(eval_py), "-i", str(completions), "-o", str(out_json)],
    ]

    for cmd in candidate_cmds:
        try:
            print("[EVAL] trying:", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and out_json.exists():
                print("[EVAL] success with:", " ".join(cmd))
                return True
            else:
                # Show brief stderr to help debug, but not responses
                print("[EVAL] failed pattern; stderr:\n", (res.stderr or "")[:4000])
        except Exception as e:
            print("[EVAL] exception:", e)

    return False

def summarize_asr(out_json: Path):
    data = json.loads(out_json.read_text())
    # Many evaluators output something like {"asr": float, ...}
    # If schema differs, try some common fallbacks
    keys = ["asr", "ASR", "attack_success_rate", "attack_success"]
    for k in keys:
        if k in data:
            return float(data[k])
    # Fallback: if it returns per-item labels, compute mean of harmful flags
    if isinstance(data, list) and data and isinstance(data[0], dict):
        harmful_keys = ["is_harmful", "harmful", "jailbroken", "success"]
        flags = []
        for r in data:
            for hk in harmful_keys:
                if hk in r:
                    flags.append(bool(r[hk]))
                    break
        if flags:
            return sum(flags) / len(flags)
    raise ValueError("Could not find ASR in evaluator output; inspect the JSON schema.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmbench_root", default="HarmBench", help="Path to cloned HarmBench repo.")
    ap.add_argument("--completions", required=True, help="Path to completions JSONL produced above.")
    ap.add_argument("--out", default="experiments/asr/harmbench_eval.json", help="Where to write evaluator output.")
    args = ap.parse_args()

    hb_root = Path(args.harmbench_root)
    comps = Path(args.completions)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    assert hb_root.exists(), f"{hb_root} not found"
    assert comps.exists() and comps.stat().st_size > 0, f"{comps} missing or empty"

    ok = run_harmbench_evaluator(hb_root, comps, out)
    if not ok:
        print("\n[EVAL] Could not run HarmBench evaluator automatically.")
        print("[EVAL] Please run it manually with `python HarmBench/evaluate_completions.py -h` to confirm argument names,")
        print("       then point --completions and --out accordingly.\n")
        sys.exit(2)

    try:
        asr = summarize_asr(out)
        print(f"[ASR] HarmBench ASR = {asr:.4f}")
    except Exception as e:
        print(f"[ASR] Could not summarize ASR automatically: {e}")
        print(f"[ASR] Raw evaluator JSON at {out}")

if __name__ == "__main__":
    main()

