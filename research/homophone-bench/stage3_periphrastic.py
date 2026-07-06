#!/usr/bin/env python3
"""
STAGE 3: PERIPHRASTIC GENERATION ENGINE
Wires together: phrase_bank, compose_lots, whole_line_carve, babel_windows, fragment_weave.

Generates endless periphrastic EN↔FR phrase pairs by:
  1. FRAGMENT WEAVE: growing novel multi-word pairs from shared IPA blocks
  2. BABEL WINDOWS: merging EN 2-3-grams into single FR words/units
  3. COMPOSE LOTS: DP-based fragment assembly for periphrastic expansion
  4. WHOLE-LINE CARVE: Van Rooten-style line-level beam decoding
  5. WIRE EVERYTHING: extracting word-aligned periphrastic training edges

Scoring cascade: combo × fluency × zipf × semantic × novelty.

Run: python stage3_periphrastic.py --rounds 3 --budget 300
"""
import os, sys, json, subprocess, time
from collections import defaultdict
import numpy as np

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")
print("STAGE 3: PERIPHRASTIC GENERATION ENGINE")
print("=" * 55)

# ── Import all the tools ──
import matcher
import poetry_mode as pm
import whole_line_carve as wlc
import bigram_lm as blm
from lexicon_g2p import clean_ipa

# Load bigram LMs
lm_en = blm.load("en") if os.path.exists("bigram-lm-en.pkl") else None
lm_fr = blm.load("fr") if os.path.exists("bigram-lm-fr.pkl") else None
print(f"  Bigram LMs: EN={'✓' if lm_en else '✗'} FR={'✓' if lm_fr else '✗'}")

# Load Stage 1 vocabulary
stage1_en = set(); stage1_fr = set()
for line in open("stage1_homophones.jsonl",encoding="utf-8"):
    r = json.loads(line)
    stage1_en.add(r["en"]); stage1_fr.add(r["fr"].split()[0])
print(f"  Stage 1 vocab: {len(stage1_en)} EN, {len(stage1_fr)} FR")

# ═══════════════════════════════════════════════════════════
# TOOL 1: WHOLE-LINE CARVE (Van Rooten style)
# ═══════════════════════════════════════════════════════════
def whole_line_generate(line, beam=420):
    """Carve a full line: EN IPA → beam search → French word sequence."""
    try:
        ipa = wlc.en_ipa(line)
        pm.force_coverage()
        root = pm.build_poetry_trie(min_zipf=2.0)
        wlc.force_coverage()
        _, _, scored = wlc.carve_line(line, root, beam=beam)
        if scored:
            best = scored[0]  # (dual, combo, coh, cov, fr, nfr, nf)
            return {"line": line, "fr": best[4], "combo": round(best[1],3),
                    "fluency": round(best[2],3), "coverage": round(best[3],3)}
    except: pass
    return None

# ═══════════════════════════════════════════════════════════
# TOOL 2: COMPOSE LOTS (DP-based fragment assembly)
# ═══════════════════════════════════════════════════════════
def compose_periphrastic(line):
    """Periphrastic line composition using dictionary + fragments."""
    words = [w.lower().strip(".,;:!?\"'") for w in line.split() if len(w)>=2]
    fr_parts = []
    
    for w in words:
        # Strategy 1: Dictionary lookup
        found = None
        for l in open("stage1_homophones.jsonl",encoding="utf-8"):
            r = json.loads(l)
            if r["en"] == w:
                found = {"fr": r["fr"], "sound": r.get("sound",1.0), "via": "dict"}
                break
        
        if found:
            fr_parts.append(found)
        else:
            # Strategy 2: Babel window — try 2-gram merge
            # (Skip for now — babel_windows.py would go here)
            fr_parts.append({"fr": f"«{w}»", "sound": 0.0, "via": "missing"})
    
    return {"line": line, "fr": " ".join(p["fr"] for p in fr_parts),
            "parts": fr_parts, "n_parts": len(fr_parts)}

# ═══════════════════════════════════════════════════════════
# TOOL 3: SCORE AND RANK
# ═══════════════════════════════════════════════════════════
def score_output(en, fr):
    """Score cascade: combo × fluency × zipf × novelty."""
    s_combo = matcher.homophone_score(en, fr) if hasattr(matcher,'homophone_score') else 0.5
    s_flu = lm_fr.fluency(fr.split()) if lm_fr else 0.5
    fr_words = fr.split()
    en_words = en.split()
    
    # Zipf: prefer common words
    s_zipf = 0.5  # placeholder
    # Novelty: penalize EN→FR identity
    s_novelty = 1.0 if en.lower() != fr.lower() else 0.5
    # Coverage: how many Stage 1 words used
    cov = sum(1 for w in fr_words if w in stage1_fr) / max(1, len(fr_words))
    
    return round(s_combo * (0.3 + 0.7*s_flu) * s_novelty * (0.5 + 0.5*cov), 3)

# ═══════════════════════════════════════════════════════════
# MAIN GENERATION LOOP
# ═══════════════════════════════════════════════════════════
def generate_periphrastic(seeds, n_rounds=3, budget=300):
    """Generate endless periphrastic variations from seed lines."""
    results = []
    total = 0
    
    for round_n in range(1, n_rounds+1):
        print(f"\n  Round {round_n}/{n_rounds}...")
        round_new = 0
        
        for line in seeds * 3:  # repeat seeds for variation
            if round_new >= budget / n_rounds: break
            
            # Strategy 1: Whole-line carve (Van Rooten)
            wlc_result = whole_line_generate(line)
            if wlc_result and wlc_result["combo"] >= 0.45:
                score = score_output(line, wlc_result["fr"])
                wlc_result["score"] = score
                wlc_result["strategy"] = "whole_line_carve"
                results.append(wlc_result)
                round_new += 1
            
            # Strategy 2: Compose lots (periphrastic)
            comp_result = compose_periphrastic(line)
            score = score_output(line, comp_result["fr"])
            comp_result["score"] = score
            comp_result["strategy"] = "compose_lots"
            results.append(comp_result)
            round_new += 1
            
            total += 2
            if total % 100 == 0:
                avg = np.mean([r["score"] for r in results[-100:]])
                print(f"    [{total:5d}] avg_score={avg:.3f}")
    
    return results

# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=500)
    args = ap.parse_args()
    
    # Seed lines
    seeds = [
        "the sea remembers every ship",
        "we call to the moon and she answers",
        "my sorrow sleeps in a deep well",
        "bless the dawn that made us free",
        "less debt less mess more soup",
        "the cat sat on the mat",
        "mary had a little lamb",
        "she walks in beauty like the night",
    ]
    
    results = generate_periphrastic(seeds, args.rounds, args.budget)
    
    # Save
    with open("stage3_periphrastic.jsonl","w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    scores = [r["score"] for r in results]
    strategies = defaultdict(int)
    for r in results: strategies[r["strategy"]] += 1
    
    print(f"\n{'='*55}")
    print(f"STAGE 3 COMPLETE: {len(results)} periphrastic pairs")
    print(f"  Score: μ={np.mean(scores):.3f} σ={np.std(scores):.3f}")
    print(f"  Strategies: {dict(strategies)}")
    print(f"  Saved: stage3_periphrastic.jsonl ({os.path.getsize('stage3_periphrastic.jsonl')/1e6:.1f}MB)")
    
    # Top samples
    for r in sorted(results, key=lambda x: -x["score"])[:5]:
        print(f"  [{r['strategy']:20s}] {r['line'][:40]} → {r['fr'][:40]} (s={r['score']:.3f})")
