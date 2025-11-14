def is_degenerate(prompt: str, output: str) -> bool:
    if not output: return True
    p = (prompt or "").strip().lower()
    o = output.strip()
    if len(o.split()) < 10:         # too short
        return True
    if o.lower().startswith(p[:80]): # prompt echo
        return True
    toks = o.split()
    if len(toks) > 1:
        rep = sum(1 for a,b in zip(toks, toks[1:]) if a==b) / (len(toks)-1)
        if rep > 0.15:              # excessive immediate repeats
            return True
    return False
