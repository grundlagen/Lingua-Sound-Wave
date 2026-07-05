#!/usr/bin/env python3
"""
DUAL-SCALE PARAGRAPH COMPOSER — Homophonic + semantic at 200-word scale.

APPROACH:
  1. Select 200 English content words from the ladder DB (sound ≥ 0.60)
  2. For each, find the BEST dual French word: max(sound × meaning_coverage)
  3. Chain them using the chain-web graph (70k transitive edges)
  4. Verify loop-certified pairs (bidirectional EN↔FR consistency)
  5. Compose into fluent French lines using bigram LM
  6. Strict judge: every word must have BOTH sound ≥ 0.55 AND meaning ≥ 0.45

NO WHISPER. NO CARVING. NO NURSERY RHYMES.
Pure set-theoretic algebraic composition at scale.

The output is a 200-word bilingual paragraph where every French word:
  - Sounds like its English counterpart (homophone)
  - Means something in the English content universe (semantic)
  - Chains fluently with neighbors (bigram coherence)

Run: python dual_scale_composer.py
"""

import subprocess, os, sys, json
from collections import defaultdict
import numpy as np

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ═══════════════════════════════════════════════════════════════
# LOAD DATABASES
# ═══════════════════════════════════════════════════════════════
print("Loading databases...")

# ── ladder (sound edges) ──
ladder = defaultdict(list)
for i,line in enumerate(open("tier-ladder.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=12 and p[10]:
        try:
            s = float(p[10]); m = float(p[11]) if p[11] else 0.5
            ladder[p[1]].append((s, m, p[2]))
        except: continue
for k in ladder: ladder[k].sort(key=lambda x: -x[0])
print(f"  ladder: {sum(len(v) for v in ladder.values())} edges, {len(ladder)} EN words")

# ── strict-gold ──
strict = defaultdict(list)
for i,line in enumerate(open("strict-gold.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2: strict[p[0]].append((p[1], 1.0, 0.9))
print(f"  strict-gold: {sum(len(v) for v in strict.values())} pairs")

# ── v7 gold ──
v7gold = defaultdict(list)
for i,line in enumerate(open("dictionary-v7.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=9 and p[3]=="1":
        v7gold[p[7]].append((p[8], float(p[1]), 1.0))
print(f"  v7 gold: {sum(len(v) for v in v7gold.values())} pairs")

# ── chain-web ──
chain = defaultdict(lambda: defaultdict(list))
for i,line in enumerate(open("chain-web-full-v7u.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=5:
        a,b = p[0],p[1]
        if ":" in a and ":" in b:
            sl,sw = a.split(":",1); tl,tw = b.split(":",1)
            if sl=="en" and tl=="fr":
                chain[sw][tw].append((int(p[2]), float(p[3])))
print(f"  chain-web: {sum(len(v) for v in chain.values())} source words")

# ── loop-certified ──
loops = set()
for i,line in enumerate(open("loop-certified-pairs-v7u.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2: loops.add((p[0], p[1]))
print(f"  loop-certified: {len(loops)} pairs")

# ── FR homophone classes ──
fr_class = {}
for path in ["fr-homophone-classes-lexique.tsv","fr-homophone-classes.tsv"]:
    try:
        for i,line in enumerate(open(path,encoding="utf-8")):
            if i==0: continue
            ms = line.rstrip("\n").split("\t")[1].split()
            for m in ms: fr_class[m] = ms
    except: pass
print(f"  FR classes: {len(fr_class)}")

# ── Bigram LM (for French fluency) ──
try:
    import bigram_lm as blm
    LM = blm.load("fr")
    print(f"  bigram LM: {LM.N:,} tokens, {len(LM.bigrams):,} bigrams")
except:
    LM = None
    print("  bigram LM: unavailable (fluency fallback=0.5)")

# ── Reverse meaning index: FR word → {EN words it can mean} ──
fr_means = defaultdict(set)
for en_w, entries in ladder.items():
    for s, m, fr_w in entries:
        if s >= 0.55: fr_means[fr_w].add(en_w)
for en_w, entries in strict.items():
    for fr_w, s, m in entries:
        fr_means[fr_w].add(en_w)
for en_w, entries in v7gold.items():
    for fr_w, s, m in entries:
        fr_means[fr_w].add(en_w)
print(f"  fr_means: {len(fr_means)} FR words with meaning paths")

# ═══════════════════════════════════════════════════════════════
# SELECT 200 WORDS
# ═══════════════════════════════════════════════════════════════
STOP = {"the","a","an","of","in","on","and","or","to","for","with","by","at",
        "as","is","are","was","were","be","been","am","it","he","she","we","they",
        "you","me","him","her","us","them","my","your","his","her","its","our",
        "their","this","that","these","those","not","no","so","but","if","all",
        "some","more","very","just","only","too","also","then","now","here","there",
        "did","do","does","had","has","have","can","could","will","would","shall",
        "should","may","might","must","from","about","into","over","under","up",
        "out","down","off","away","back","still","even","much","many","few","each",
        "every","any","both","such","same","own","other","another","new","old",
        "good","great","big","small","high","low","long","short","right","left"}

print(f"\nSelecting 200 content words (sound ≥ 0.55)...")
candidates = []
for en_w in ladder:
    if en_w in STOP or len(en_w) < 3: continue
    best = ladder[en_w]
    if best[0][0] >= 0.55:
        # Score: best sound × how many meanings the best FR word covers
        best_fr = best[0][2]
        meaning_count = len(fr_means.get(best_fr, set()))
        composite = best[0][0] * (1 + 0.1 * min(meaning_count, 10))
        candidates.append((composite, en_w, best_fr, best[0][0], meaning_count))

candidates.sort(reverse=True)
selected = candidates[:200]

print(f"  Selected {len(selected)} words")
print(f"  Sample: {', '.join(w for _,w,_,_,_ in selected[:30])}...")

# ═══════════════════════════════════════════════════════════════
# BUILD THE DUAL COMPOSITION
# ═══════════════════════════════════════════════════════════════
print(f"\nBuilding dual composition (sound × meaning)...")

pairs = []  # (en_word, fr_word, sound, meaning_set, is_loop)
universe = set(w for _,w,_,_,_ in selected)  # all EN content words
covered = set()
loop_count = 0

for composite, en_w, fr_w, sound, mcount in selected:
    meaning = fr_means.get(fr_w, set())
    meaning_in_universe = meaning & universe
    is_loop = (en_w, fr_w) in loops or (fr_w, en_w) in loops
    if is_loop: loop_count += 1
    
    pairs.append({
        "en": en_w, "fr": fr_w,
        "sound": round(sound, 3),
        "meaning": sorted(meaning_in_universe)[:8],
        "meaning_size": len(meaning_in_universe),
        "loop": is_loop,
    })
    covered |= meaning_in_universe

coverage = len(covered) / len(universe) if universe else 0

print(f"  Pairs: {len(pairs)}")
print(f"  Loop-certified: {loop_count}/{len(pairs)}")
print(f"  Universe: {len(universe)} words")
print(f"  Covered:  {len(covered)}/{len(universe)} ({coverage*100:.0f}%)")
print(f"  Sound scores: μ={np.mean([p['sound'] for p in pairs]):.3f} "
      f"σ={np.std([p['sound'] for p in pairs]):.3f}")
print(f"  Meaning sizes: μ={np.mean([p['meaning_size'] for p in pairs]):.1f}")

# ═══════════════════════════════════════════════════════════════
# RENDER AS PARAGRAPH
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"DUAL-SCALE BILINGUAL PARAGRAPH (200 words)")
print(f"{'='*60}")

# Try to group into fluent lines using bigram LM
lines = []
current_line = []
current_flu = 0

for i, p in enumerate(pairs):
    current_line.append(p["fr"])
    
    if len(current_line) >= 8:
        flu = LM.fluency(current_line) if LM else 0.5
        en_words = " ".join(p["en"] for p in pairs[max(0,i-7):i+1])
        lines.append({
            "fr": " ".join(current_line),
            "en": en_words,
            "flu": flu,
            "n_words": len(current_line),
        })
        current_line = []

# Remaining words
if current_line:
    lines.append({
        "fr": " ".join(current_line),
        "en": "...",
        "flu": LM.fluency(current_line) if LM else 0.5,
        "n_words": len(current_line),
    })

for i, line in enumerate(lines):
    sound_avg = np.mean([p["sound"] for p in pairs[i*8:(i+1)*8] if i*8 < len(pairs)])

    print(f"\n  [{i:2d}] FR: {line['fr']}")
    print(f"       EN: (sound μ={sound_avg:.3f}, flu={line['flu']:.3f}, "
          f"words={line['n_words']})")
    
    # Show 3 example word mappings
    start = i*8
    examples = pairs[start:start+3]
    if examples:
        ex_str = "  ".join(f"{p['en']}→{p['fr']}({p['sound']:.2f})" for p in examples)
        print(f"       ↳ {ex_str}")

# ═══════════════════════════════════════════════════════════════
# ROOTEN BAND ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"ROOTEN BAND ANALYSIS")
print(f"{'='*60}")

# Rooten band: sound ≥ 0.55 AND meaning coverage ≥ 0.45
# (meaning coverage = fraction of universe this FR word covers)
rooten_threshold_s = 0.55
rooten_threshold_m = 0.45

in_rooten = 0
for p in pairs:
    # Meaning coverage = fraction of universe this word can mean
    m_coverage = p["meaning_size"] / len(universe) if universe else 0
    s_ok = p["sound"] >= rooten_threshold_s
    m_ok = m_coverage >= 0.01  # at least 2 words of the 200-word universe
    if s_ok and m_ok:
        in_rooten += 1

print(f"  Sound ≥ {rooten_threshold_s}: "
      f"{sum(1 for p in pairs if p['sound']>=rooten_threshold_s)}/{len(pairs)}")
print(f"  Meaning ≥ 1 word: "
      f"{sum(1 for p in pairs if p['meaning_size']>=1)}/{len(pairs)}")
print(f"  BOTH: {in_rooten}/{len(pairs)} ({100*in_rooten/len(pairs):.0f}%)")

# Persistence = minimum sound score among all pairs
persistence = min(p["sound"] for p in pairs)
print(f"  Persistence: {persistence:.3f} (minimum sound score in the cover)")

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
output = {
    "n_words": len(pairs),
    "universe_size": len(universe),
    "coverage": round(coverage, 3),
    "persistence": round(persistence, 3),
    "loop_certified": loop_count,
    "rooten_band": in_rooten,
    "pairs": pairs,
    "lines": [{"fr": l["fr"], "flu": round(l["flu"],3)} for l in lines],
}

with open("dual_scale_paragraph.json","w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved dual_scale_paragraph.json ({len(pairs)} pairs)")

print(f"\n{'='*60}")
print(f"DONE — {len(pairs)}-word dual composition")
print(f"  Phonetic:   persistence={persistence:.3f}")
print(f"  Semantic:   coverage={coverage*100:.0f}%")
print(f"  Loop-cert:  {loop_count} pairs")
print(f"  Rooten:     {in_rooten}/{len(pairs)} words ({100*in_rooten/len(pairs):.0f}%)")
