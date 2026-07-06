#!/usr/bin/env python3
"""
STAGE 2: MEANING WEB — Full bipartite graph with meaning on both sides.
Builds on Stage 1's homophone pairs. Adds:
  - FR homophone classes (vert=verre=vers=ver, 33K classes)
  - Synonyms on both sides (muse-pivot-syn, 51K FR + 44K EN)
  - Chain-web transitive edges (70K)
  - Loop-certified bidirectional pairs (814)
  - Zipf frequencies (common words prioritized)
  - MUSE literal translations (113K, for meaning reference)
  - fr_means backlinks (what EN words each FR word maps to)

Output: stage2_graph.json — the complete bipartite meaning graph.
"""

import json, os, sys
from collections import defaultdict

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

print("STAGE 2: MEANING WEB")
print("=" * 60)

# Load Stage 1 output
pairs = {}
if os.path.exists("stage1_homophones.jsonl"):
    for line in open("stage1_homophones.jsonl",encoding="utf-8"):
        r = json.loads(line)
        pairs[(r["en"], r["fr"])] = r
    print(f"  Stage 1 pairs loaded: {len(pairs)}")
else:
    print("  Stage 1 not run yet — loading from raw sources...")
    # Fallback: load from raw files
    import run_stage_1
    pairs = run_stage_1.pairs

# Build forward/backward indices
en_to_fr = defaultdict(list)   # EN → [(FR, sound, ...)]
fr_to_en = defaultdict(set)    # FR → {EN words this FR means}
fr_class = {}                   # FR homophone classes
syn_en = defaultdict(set)      # EN synonyms
syn_fr = defaultdict(set)      # FR synonyms
zipf = {}                       # word → frequency rank
chain_hops = defaultdict(lambda: defaultdict(int))
loops = set()

# ── 1. Populate from Stage 1 pairs ──
for (en,fr), d in pairs.items():
    en_to_fr[en].append((fr, d.get("sound",1.0), d))
    fr_to_en[fr].add(en)

# ── 2. FR homophone classes ──
for path in ["fr-homophone-classes-lexique.tsv","fr-homophone-classes.tsv"]:
    try:
        for i,line in enumerate(open(path,encoding="utf-8")):
            if i==0: continue
            parts = line.rstrip("\n").split("\t")
            if len(parts)>=2:
                members = parts[1].split()
                for m in members: fr_class[m] = members
    except: pass
class_count = len(fr_class)
expanded = 0
for fr_word in list(fr_to_en.keys()):
    if fr_word in fr_class:
        for sib in fr_class[fr_word]:
            if sib != fr_word and sib not in fr_to_en:
                fr_to_en[sib] = fr_to_en[fr_word].copy(); expanded += 1
print(f"  FR homophone classes: {class_count} ({expanded} expanded to meaning map)")

# ── 3. Synonyms ──
for line in open("muse-pivot-syn.tsv",encoding="utf-8"):
    a,b,_ = line.rstrip("\n").split("\t")
    if a.startswith("en:") and b.startswith("en:"):
        syn_en[a[3:]].add(b[3:]); syn_en[b[3:]].add(a[3:])
    elif a.startswith("fr:") and b.startswith("fr:"):
        syn_fr[a[3:]].add(b[3:]); syn_fr[b[3:]].add(a[3:])
print(f"  EN synonyms: {sum(len(v) for v in syn_en.values())} edges")
print(f"  FR synonyms: {sum(len(v) for v in syn_fr.values())} edges")

# ── 4. MUSE literal translations ──
muse_count = 0
try:
    for line in open("/tmp/muse-en-fr.txt",encoding="utf-8"):
        p = line.split()
        if len(p)==2:
            en_m, fr_m = p[0].lower(), p[1].lower()
            if en_m not in en_to_fr: en_to_fr[en_m] = []  # mark as known
            if fr_m not in fr_to_en: fr_to_en[fr_m].add(en_m)
            muse_count += 1
except: pass
print(f"  MUSE: {muse_count} translations loaded")

# ── 5. Zipf frequencies ──
try:
    # Load from data/lexique.tsv (French word frequency)
    for i,line in enumerate(open("data/lexique.tsv",encoding="utf-8",errors="ignore")):
        w = line.split("\t")[0].strip().lower()
        if w and w not in zipf: zipf[w] = i + 1
        if len(zipf) > 100000: break
except: pass
# Also try fr-word-ipa.tsv as fallback
if len(zipf) < 1000:
    try:
        for i,line in enumerate(open("fr-word-ipa.tsv",encoding="utf-8")):
            p = line.rstrip("\n").split("\t")
            if len(p)>=2 and p[1] and "(en)" not in p[0]:
                w = p[0].lower()
                if w not in zipf: zipf[w] = 50000 + i
    except: pass
print(f"  Zipf: {len(zipf)} words ranked by frequency")

# ── 6. Chain-web ──
try:
    for i,line in enumerate(open("chain-web/archive/chain-web-full-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=5 and ":" in p[0] and ":" in p[1]:
            sl,sw = p[0].split(":",1); tl,tw = p[1].split(":",1)
            if sl=="en" and tl=="fr":
                h = int(p[2])
                if h < chain_hops[sw].get(tw, 999):
                    chain_hops[sw][tw] = h
            elif sl=="fr" and tl=="en":
                h = int(p[2])
                if h < chain_hops[tw].get(sw, 999):
                    chain_hops[tw][sw] = h
except: pass
print(f"  Chain-web: {sum(len(v) for v in chain_hops.values())} edges")

# ── 7. Loop-certified ──
try:
    for i,line in enumerate(open("chain-web/archive/loop-certified-pairs-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=2: loops.add((p[0].lower(), p[1].lower()))
except: pass
print(f"  Loop-certified: {len(loops)} bidirectional pairs")

# ── Build the Stage 2 graph ──
graph = {
    "nodes_en": len(en_to_fr),
    "nodes_fr": len(fr_to_en) + len(fr_class),
    "sound_edges": sum(len(v) for v in en_to_fr.values()),
    "meaning_edges": sum(len(v) for v in fr_to_en.values()),
    "homophone_classes": class_count,
    "synonym_edges": sum(len(v) for v in syn_en.values()) + sum(len(v) for v in syn_fr.values()),
    "chain_hops": sum(len(v) for v in chain_hops.values()),
    "loop_pairs": len(loops),
    "zipf_words": len(zipf),
    "muse_pairs": muse_count,
}

# Write the full graph as JSON (with subsets for size)
output = {
    "stats": graph,
    "en_to_fr": {k: [(f,s) for f,s,_ in v[:10]] for k,v in list(en_to_fr.items())[:5000]},
    "fr_to_en": {k: list(v)[:10] for k,v in list(fr_to_en.items())[:5000]},
    "fr_class_sample": {k: v for k,v in list(fr_class.items())[:1000]},
    "syn_en_sample": {k: list(v)[:5] for k,v in list(syn_en.items())[:1000]},
    "chain_hops_sample": {k: dict(list(v.items())[:5]) for k,v in list(chain_hops.items())[:1000]},
    "loops": list(loops)[:500],
    "zipf_top": {k: v for k,v in sorted(zipf.items(), key=lambda x: x[1])[:200]},
}

with open("stage2_graph.json","w") as f:
    json.dump(output, f, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"STAGE 2 COMPLETE")
print(f"  Output: stage2_graph.json ({os.path.getsize('stage2_graph.json')/1e6:.1f}MB)")
print(f"  Graph: {graph}")
