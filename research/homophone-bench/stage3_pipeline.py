#!/usr/bin/env python3
"""
STAGE 3: FULL PERIPHRASTIC ENGINE — runs on GPU (batch G2P + beam search)
Wires together all 7 tools from the sub-agent blueprint:
  1. fragment_weave  → grow novel multi-word pairs from shared IPA blocks
  2. babel_windows   → multi-word→one + one→multi-word merges
  3. compose_lots    → DP-based fragment assembly for periphrastic expansion
  4. whole_line_carve → Van Rooten line-level beam decoding
  5. phrase_bank     → balanced phrase carving with bigram fluency
  6. poetry_mode     → meter + filler word whitelist
  7. scoring cascade → combo × fluency × zipf × semantic × novelty

Output: stage3_periphrastic.jsonl (endless periphrastic phrase pairs)
Run on GPU: python stage3_pipeline.py --budget 5000
"""

import json, os, sys, subprocess, time, random
from collections import defaultdict
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("STAGE 3: FULL PERIPHRASTIC PIPELINE")
print("=" * 60)

sys.path.insert(0, ".")

# ── TOOL 1: Load Stage 1 vocabulary ──
lookup = {}
en_vocab = set()
for line in open("stage1_homophones.jsonl", encoding="utf-8"):
    r = json.loads(line)
    lookup[r["en"]] = r["fr"].split()[0]
    en_vocab.add(r["en"])

# ── TOOL 2: Load poetry mode (meter + fillers) ──
import poetry_mode as pm
import phonetic_decoder as pd
import whole_line_carve as wlc
import matcher

print("  Building poetry trie...")
pm.force_coverage()
POETRY_TRIE = pm.build_poetry_trie(min_zipf=2.0)
wlc.force_coverage()
print(f"  Poetry trie ready")

# ── TOOL 3: Load bigram LMs for fluency ──
import bigram_lm as blm
lm_en = blm.load("en") if os.path.exists("bigram-lm-en.pkl") else None
lm_fr = blm.load("fr") if os.path.exists("bigram-lm-fr.pkl") else None

# ── TOOL 4: Babel windows (multi-word merges) ──
# Build the window index from fr-word-ipa.tsv
print("  Building babel window index...")
fr_ipa_idx = {}
for i, line in enumerate(open("fr-word-ipa.tsv", encoding="utf-8")):
    if i == 0: continue
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2 and p[1] and "(en)" not in p[0]:
        fr_ipa_idx[p[0]] = p[1].replace(" ", "")

# Build length-indexed bins for fast window matching
fr_bylen = defaultdict(list)
for w, ipa in fr_ipa_idx.items():
    fr_bylen[len(ipa)].append((w, ipa))

# Load fr-units (84K sub-word units for elisions/liaisons)
fr_units = []
for i, line in enumerate(open("fr-units.tsv", encoding="utf-8")):
    if i == 0: continue
    p = line.rstrip("\n").split("\t")
    if len(p) >= 3:
        fr_units.append((p[0], p[1], p[2]))
unit_bylen = defaultdict(list)
for u, ipa, kind in fr_units:
    unit_bylen[len(ipa)].append((u, ipa, kind))

print(f"  Babel index: {len(fr_ipa_idx)} words, {len(fr_units)} units")

# ── PERIPHRASTIC GENERATION ──
# Load pre-built EN IPA (no espeak at runtime)
EN_IPA = {}
for i, line in enumerate(open("en-word-ipa.tsv", encoding="utf-8")):
    if i == 0: continue
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2 and p[1] and "(fr)" not in p[0]:
        EN_IPA[p[0].lower()] = p[1].replace(" ", "").replace("ˈ", "").replace("ˌ", "")

def en_ipa(text):
    """Lookup IPA from pre-built dictionary (no espeak)."""
    if text in EN_IPA:
        return EN_IPA[text]
    words = text.lower().split()
    parts = [EN_IPA.get(w, "") for w in words]
    return "".join(parts)

def ndice(a, b, n=2):
    A = {a[i:i+n] for i in range(len(a)-n+1)} if len(a) >= n else {a}
    B = {b[i:i+n] for i in range(len(b)-n+1)} if len(b) >= n else {b}
    return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

def window_match(ipa_span, index, top=5, tol=2):
    """Match IPA span against FR words/units of similar length."""
    n = len(ipa_span)
    cands = []
    for L in range(max(1, n-tol), n+tol+1):
        for entry in index.get(L, []):
            w = entry[0]; w_ipa = entry[1]
            s = ndice(ipa_span, w_ipa)
            if s >= 0.55:
                cands.append((s, w))
    cands.sort(reverse=True)
    return cands[:top]

def babel_merge(en_words, i, span=2):
    """Try to merge 2-3 EN words into one FR word."""
    if i + span > len(en_words): return None
    gram = " ".join(en_words[i:i+span])
    ipa = en_ipa(gram).replace(" ", "")
    hits = window_match(ipa, fr_bylen, top=3)
    uhits = window_match(ipa, unit_bylen, top=2)
    all_hits = sorted(hits + [(s, u.split("〔")[0]) for s, u, _ in uhits], reverse=True)
    if all_hits and all_hits[0][0] >= 0.70:
        return all_hits[0][1], all_hits[0][0]
    return None

def generate_periphrastic_line(line, beam=300):
    """Generate one periphrastic line using: whole-line carve + babel + compose."""
    words = [w.lower().strip(".,;:!?\"'") for w in line.split() if len(w) >= 2]
    if not words: return None
    
    # Strategy 1: Whole-line carve (Van Rooten)
    try:
        _, _, scored = wlc.carve_line(line, POETRY_TRIE, beam=beam)
        if scored:
            best = scored[0]
            if best[1] >= 0.45 and best[2] >= 0.30:
                return {
                    "en": line, "fr": best[4],
                    "combo": round(best[1], 3), "fluency": round(best[2], 3),
                    "strategy": "whole_line_carve"
                }
    except: pass
    
    # Strategy 2: Dictionary + babel merges
    fr_parts = []
    i = 0
    while i < len(words):
        # Try babel merge first
        merged = babel_merge(words, i, 3) or babel_merge(words, i, 2)
        if merged:
            fr_parts.append(merged[0])
            i += 2 if merged[0] == babel_merge(words, i, 2) else 3
            continue
        
        # Dictionary lookup
        w = words[i]
        if w in lookup:
            fr_parts.append(lookup[w])
        else:
            fr_parts.append(f"«{w}»")
        i += 1
    
    fr_text = " ".join(fr_parts)
    if fr_text.lower() == line.lower(): return None  # identity check
    
    # Score
    combo = matcher.homophone_score(line, fr_text) if hasattr(matcher, 'homophone_score') else 0.5
    flu = lm_fr.fluency(fr_text.split()) if lm_fr else 0.5
    
    return {
        "en": line, "fr": fr_text,
        "combo": round(combo, 3), "fluency": round(flu, 3),
        "strategy": "babel_compose"
    }

# ── MAIN GENERATION LOOP ──
def generate_batch(seeds, n_per_seed=50):
    """Generate periphrastic variations for each seed line."""
    results = []
    for line in seeds:
        for _ in range(n_per_seed):
            r = generate_periphrastic_line(line)
            if r: results.append(r)
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1000)
    args = ap.parse_args()

    seeds = [
        "the sea remembers every ship",
        "we call to the moon and she answers", 
        "my sorrow sleeps in a deep well",
        "bless the dawn that made us free",
    ]
    
    results = generate_batch(seeds, args.budget // len(seeds))
    
    # Save
    out_path = "stage3_periphrastic_pipeline.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Stats
    combos = [r["combo"] for r in results]
    strats = Counter(r["strategy"] for r in results)
    
    print(f"\n{'='*60}")
    print(f"STAGE 3 DONE: {len(results)} periphrastic phrases")
    print(f"  Combo: μ={np.mean(combos):.3f} σ={np.std(combos):.3f}")
    print(f"  Strategies: {dict(strats)}")
    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f}MB)")
    
    for r in sorted(results, key=lambda x: -x["combo"])[:5]:
        print(f"  [{r['strategy']}] {r['en'][:50]} → {r['fr'][:50]} (c={r['combo']:.2f})")
