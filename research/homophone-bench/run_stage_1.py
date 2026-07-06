#!/usr/bin/env python3
"""
STAGE 1: Unified Word Pair Database Builder
Loads EVERY word-pair source into one massive database.
Outputs: stage1_pairs.jsonl (every pair), stage1_stats.json (summary)

Sources:
  1. strict-gold.tsv (1,314 judge-verified pairs)
  2. dictionary-v7.tsv (2,070 gold pairs with tier labels)
  3. tier-ladder.tsv (98k pairs, sound ≥ 0.55)
  4. dual-pairs.tsv (102k pairs, identity-filtered)
  5. fr-anchored-pairs.tsv (1,497 FR-anchored pairs)
  6. MUSE translation dict (113k literal EN↔FR translations)
  7. GPU model predictions (if model available)
  8. Fragment-based generation (fr-units.tsv, 84k units)

Run: python run_stage_1.py
     python run_stage_1.py --include-muse --include-fragments
"""

import json, os, sys
from collections import defaultdict

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ═══════════════════════════════════════════════════════════════
print("STAGE 1: Unified Word Pair Database Builder")
print("=" * 60)

all_pairs = {}  # (en, fr) → {sound, meaning, source, tier, loop, chain, ...}
sources_stats = {}

# ── Source 1: strict-gold ──
count = 0
for i,line in enumerate(open("strict-gold.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[0] != p[1]:
        key = (p[0].lower().strip(), p[1].lower().strip())
        if key not in all_pairs or 1.0 > all_pairs[key].get("sound",0):
            all_pairs[key] = {"sound": 1.0, "meaning": 0.9, "source": "strict-gold", "tier": "S"}
        count += 1
sources_stats["strict-gold"] = count
print(f"  strict-gold: {count} pairs")

# ── Source 2: dictionary-v7 ──
count = 0
try:
    for i,line in enumerate(open("dictionary-v7.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=9 and p[3]=="1":  # gold column
            key = (p[7].lower().strip(), p[8].lower().strip())
            snd = float(p[1]) if p[1] else 1.0
            if key not in all_pairs or snd > all_pairs[key].get("sound",0):
                all_pairs[key] = {"sound": snd, "meaning": 1.0, "source": "v7", "tier": p[0]}
            count += 1
except FileNotFoundError: pass
sources_stats["dictionary-v7"] = count
print(f"  dictionary-v7: {count} pairs")

# ── Source 3: tier-ladder ──
count = 0
for i,line in enumerate(open("tier-ladder.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=12 and p[10]:
        try:
            snd = float(p[10])
            if snd >= 0.55:
                key = (p[1].lower().strip(), p[2].lower().strip())
                mng = float(p[11]) if p[11] else 0.5
                if key not in all_pairs or snd > all_pairs[key].get("sound",0):
                    all_pairs[key] = {"sound": snd, "meaning": mng, "source": "ladder", "tier": p[0]}
                count += 1
        except: continue
sources_stats["tier-ladder"] = count
print(f"  tier-ladder: {count} pairs (sound ≥ 0.55)")

# ── Source 4: dual-pairs (identity-filtered) ──
count = 0
for i,line in enumerate(open("dual-pairs.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=6 and p[0].lower() != p[1].lower():
        key = (p[0].lower().strip(), p[1].lower().strip())
        snd = float(p[2])
        if key not in all_pairs or snd > all_pairs[key].get("sound",0):
            all_pairs[key] = {"sound": snd, "meaning": 0.7, "source": "dual", "tier": p[5] if len(p)>5 else ""}
        count += 1
sources_stats["dual-pairs"] = count
print(f"  dual-pairs: {count} pairs (identity-filtered)")

# ── Source 5: fr-anchored-pairs ──
count = 0
try:
    for i,line in enumerate(open("fr-anchored-pairs.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=4:
            key = (p[0].lower().strip(), p[1].lower().strip())
            snd = float(p[2]) if len(p)>2 else 0.8
            if key not in all_pairs:
                all_pairs[key] = {"sound": snd, "meaning": 0.6, "source": "fr-anchored", "tier": p[3] if len(p)>3 else ""}
            count += 1
except FileNotFoundError: pass
sources_stats["fr-anchored"] = count
print(f"  fr-anchored: {count} pairs")

# ── Source 6: MUSE translation dict ──
count = 0
try:
    for line in open("/tmp/muse-en-fr.txt",encoding="utf-8"):
        p = line.split()
        if len(p)==2:
            key = (p[0].lower().strip(), p[1].lower().strip())
            if key not in all_pairs:
                all_pairs[key] = {"sound": 0.5, "meaning": 0.9, "source": "muse", "tier": "MUSE"}
            count += 1
except FileNotFoundError: pass
sources_stats["muse"] = count
print(f"  MUSE translation: {count} pairs")

# ── Source 7: fragment-based (fr-units → EN sound matches) ──
# Fragments are sub-word units — not directly word pairs.
# They're loaded for Stage 1 generation but not added as pairs here.
try:
    unit_count = sum(1 for _ in open("fr-units.tsv",encoding="utf-8"))
    sources_stats["fragments"] = unit_count
    print(f"  fr-units: {unit_count} fragments (loaded for generation)")
except: pass

# ── Deduplicate and sort ──
unique = {}
for (en, fr), data in all_pairs.items():
    if en == fr: continue  # skip identity pairs
    unique[(en,fr)] = data

print(f"\n  TOTAL UNIQUE: {len(unique)} EN↔FR pairs")

# ── Add chain-web support (from archive) ──
chain_support = defaultdict(int)
loop_support = set()
try:
    for i,line in enumerate(open("chain-web/archive/chain-web-full-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=5 and ":" in p[0] and ":" in p[1]:
            sl,sw = p[0].split(":",1); tl,tw = p[1].split(":",1)
            if sl=="en" and tl=="fr":
                chain_support[(sw,tw)] = max(chain_support.get((sw,tw),0), int(p[2]))
except: pass
try:
    for i,line in enumerate(open("chain-web/archive/loop-certified-pairs-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=2: loop_support.add((p[0].lower().strip(), p[1].lower().strip()))
except: pass

# Add chain/loop data
for (en,fr) in list(unique.keys()):
    if (en,fr) in chain_support:
        unique[(en,fr)]["chain_hops"] = chain_support[(en,fr)]
    if (en,fr) in loop_support:
        unique[(en,fr)]["loop_certified"] = True

chain_count = sum(1 for d in unique.values() if "chain_hops" in d)
loop_count = sum(1 for d in unique.values() if d.get("loop_certified"))
print(f"  Chain-web support: {chain_count} pairs")
print(f"  Loop-certified: {loop_count} pairs")

# ── Write output ──
with open("stage1_pairs.jsonl","w") as f:
    for (en, fr), data in sorted(unique.items()):
        record = {"en": en, "fr": fr, **data}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Stats
stats = {
    "total_pairs": len(unique),
    "sources": sources_stats,
    "chain_supported": chain_count,
    "loop_certified": loop_count,
    "tier_distribution": {},
    "sound_distribution": {"S": [0.75, 1.0], "A": [0.60, 0.75], "B": [0.45, 0.60]},
}

for d in unique.values():
    tier = d.get("tier","?")
    stats["tier_distribution"][tier] = stats["tier_distribution"].get(tier, 0) + 1

with open("stage1_stats.json","w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"STAGE 1 COMPLETE")
print(f"  Output: stage1_pairs.jsonl ({len(unique)} pairs)")
print(f"  Stats:  stage1_stats.json")
print(f"  Size:   {os.path.getsize('stage1_pairs.jsonl')/1e6:.1f}MB")
