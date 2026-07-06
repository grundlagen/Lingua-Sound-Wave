#!/usr/bin/env python3
"""
STAGE 1: HIGH-QUALITY HOMOPHONE PAIRS (filtered).
Only pairs where French SOUNDS like English. No literal translations.

Quality filter: sound ≥ 0.55 or from gold sources.
Prefer: chain-web support, loop-certified, gold tier.

Output: stage1_homophones.jsonl (~50K filtered pairs)

WHY: Stage 1 = homophone engine. Stage 2 = meaning web. Stage 3 = generation. Stage 4 = paragraphs.
"""

import json, os
from collections import defaultdict

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

print("STAGE 1: HIGH-QUALITY HOMOPHONE PAIRS")
print("=" * 60)

MIN_SOUND = 0.55
pairs = {}  # (en,fr) → data
counts = {}

def add(en, fr, sound, meaning, source, tier="", chain=0, loop=False):
    if en == fr or not en or not fr: return
    key = (en.lower().strip(), fr.lower().strip())
    if key in pairs:
        if sound > pairs[key]["sound"]:
            pairs[key].update({"sound": sound, "meaning": meaning, "source": source, "tier": tier})
    else:
        pairs[key] = {"sound": sound, "meaning": meaning, "source": source, "tier": tier,
                       "chain_hops": chain, "loop": loop}

# ── strict-gold (verified judge, always included) ──
n = 0
for i,line in enumerate(open("strict-gold.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[0].lower() != p[1].lower():
        add(p[0], p[1], 1.0, 0.9, "strict-gold", "S"); n += 1
counts["strict-gold"] = n
print(f"  strict-gold: {n} (always included)")

# ── dictionary-v7 gold ──
n = 0
try:
    for i,line in enumerate(open("dictionary-v7.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=9 and p[3]=="1":
            add(p[7], p[8], float(p[1]) if p[1] else 1.0, 1.0, "v7", p[0]); n += 1
except: pass
counts["dictionary-v7"] = n
print(f"  dictionary-v7: {n} (gold tier)")

# ── tier-ladder (sound ≥ MIN_SOUND only) ──
n = 0
for i,line in enumerate(open("tier-ladder.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=12 and p[10]:
        try:
            snd = float(p[10])
            if snd >= MIN_SOUND:
                mng = float(p[11]) if p[11] else 0.5
                add(p[1], p[2], snd, mng, "ladder", p[0]); n += 1
        except: continue
counts["tier-ladder"] = n
print(f"  tier-ladder: {n} (sound ≥ {MIN_SOUND})")

# ── dual-pairs (identity-filtered, sound ≥ MIN_SOUND) ──
n = 0
for i,line in enumerate(open("dual-pairs.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=6 and p[0].lower() != p[1].lower():
        snd = float(p[2])
        if snd >= MIN_SOUND:
            tier = p[5] if len(p)>5 and p[5] in ("S","A","B") else ""
            add(p[0], p[1], snd, 0.7, "dual", tier); n += 1
counts["dual-pairs"] = n
print(f"  dual-pairs: {n} (identity-filtered, sound ≥ {MIN_SOUND})")

# ── fr-anchored-pairs ──
n = 0
try:
    for i,line in enumerate(open("fr-anchored-pairs.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=4:
            snd = float(p[2]) if len(p)>2 and p[2] else 0.8
            if snd >= MIN_SOUND:
                add(p[0], p[1], snd, 0.6, "fr-anchored", p[3] if len(p)>3 else ""); n += 1
except: pass
counts["fr-anchored"] = n
print(f"  fr-anchored: {n}")

# ── chain-web support ──
chain = defaultdict(int); loops = set()
try:
    for i,line in enumerate(open("chain-web/archive/chain-web-full-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=5 and ":" in p[0] and ":" in p[1]:
            sl,sw = p[0].split(":",1); tl,tw = p[1].split(":",1)
            if sl=="en" and tl=="fr":
                chain[(sw,tw)] = max(chain.get((sw,tw),0), int(p[2]))
except: pass
try:
    for i,line in enumerate(open("chain-web/archive/loop-certified-pairs-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=2: loops.add((p[0].lower(),p[1].lower()))
except: pass

for (en,fr) in list(pairs.keys()):
    if (en,fr) in chain: pairs[(en,fr)]["chain_hops"] = chain[(en,fr)]
    if (en,fr) in loops: pairs[(en,fr)]["loop"] = True

print(f"  chain-web: {sum(1 for d in pairs.values() if 'chain_hops' in d)} supported")
print(f"  loop-certified: {sum(1 for d in pairs.values() if d.get('loop'))} pairs")

# ── Output ──
total = len(pairs)
print(f"\n  TOTAL: {total} high-quality homophone pairs")

out = []
for (en,fr), d in sorted(pairs.items()):
    d["en"] = en; d["fr"] = fr
    out.append(d)

with open("stage1_homophones.jsonl","w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Stats
tiers = defaultdict(int)
for d in out:
    tiers[d.get("tier","?")] += 1

sounds = [d["sound"] for d in out]
import numpy as np
print(f"\n  Sound distribution: μ={np.mean(sounds):.3f} σ={np.std(sounds):.3f}")
print(f"  Tier distribution: {dict(tiers)}")
print(f"  Sources: {counts}")
print(f"  Size: {os.path.getsize('stage1_homophones.jsonl')/1e6:.1f}MB")
