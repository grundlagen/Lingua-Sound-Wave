#!/usr/bin/env python3
"""
STAGE 2: MEANING WEB — Full graph, saved as pickle for speed.
ALL 96K EN + 193K FR nodes, ALL edges, NO sampling.
Used by Stage 3 for generation.
"""
import json, os, pickle
from collections import defaultdict

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")
print("STAGE 2: FULL MEANING WEB (rebuild)")

# Load Stage 1
pairs = {}
for line in open("stage1_homophones.jsonl",encoding="utf-8"):
    r = json.loads(line)
    pairs[(r["en"], r["fr"])] = r
print(f"  Stage 1 pairs: {len(pairs)}")

# Build ALL indices (NO sampling)
en_to_fr = defaultdict(list)     # EN → [(FR, sound)]
fr_to_en = defaultdict(set)      # FR → {EN}
fr_class = {}                     # FR homophone classes (33K)
syn_en = defaultdict(set)        # EN synonyms
syn_fr = defaultdict(set)        # FR synonyms
chain_hops = defaultdict(lambda: defaultdict(int))
loops = set()
zipf = {}

# 1. Populate from Stage 1
for (en,fr), d in pairs.items():
    en_to_fr[en].append((fr, d.get("sound",1.0)))
    fr_to_en[fr].add(en)

# 2. FR homophone classes
for path in ["fr-homophone-classes-lexique.tsv","fr-homophone-classes.tsv"]:
    try:
        for i,line in enumerate(open(path,encoding="utf-8")):
            if i==0: continue
            p = line.rstrip("\n").split("\t")
            if len(p)>=2:
                members = p[1].split()
                for m in members: fr_class[m] = members
    except: pass
print(f"  FR homophone classes: {len(fr_class)}")

# 3. Synonyms
for line in open("muse-pivot-syn.tsv",encoding="utf-8"):
    a,b,_ = line.rstrip("\n").split("\t")
    if a.startswith("en:") and b.startswith("en:"):
        syn_en[a[3:]].add(b[3:]); syn_en[b[3:]].add(a[3:])
    elif a.startswith("fr:") and b.startswith("fr:"):
        syn_fr[a[3:]].add(b[3:]); syn_fr[b[3:]].add(a[3:])

# 4. MUSE — kept SEPARATE as meaning reference, NOT sound edges
muse_pairs = {}
try:
    for line in open("/tmp/muse-en-fr.txt",encoding="utf-8"):
        p = line.split()
        if len(p)==2:
            muse_pairs[p[0].lower()] = p[1].lower()
            fr_to_en[p[1].lower()].add(p[0].lower())  # meaning only, no sound edge
except: pass

# 5. Zipf (from lexique)
try:
    for i,line in enumerate(open("data/lexique.tsv",encoding="utf-8",errors="ignore")):
        w = line.split("\t")[0].strip().lower()
        if w and w not in zipf: zipf[w] = i+1
        if i > 100000: break
except: pass

# 6. Chain-web + loops
try:
    for i,line in enumerate(open("chain-web/archive/chain-web-full-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=5 and ":" in p[0] and ":" in p[1]:
            sl,sw = p[0].split(":",1); tl,tw = p[1].split(":",1)
            if sl=="en" and tl=="fr":
                h = int(p[2]); chain_hops[sw][tw] = min(h, chain_hops[sw].get(tw, 999))
            elif sl=="fr" and tl=="en":
                h = int(p[2]); chain_hops[tw][sw] = min(h, chain_hops[tw].get(sw, 999))
except: pass
try:
    for i,line in enumerate(open("chain-web/archive/loop-certified-pairs-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=2: loops.add((p[0].lower(), p[1].lower()))
except: pass

# Convert defaultdicts to regular dicts for pickle
en_to_fr_d = {k: v for k,v in en_to_fr.items()}
fr_to_en_d = {k: list(v) for k,v in fr_to_en.items()}
chain_d = {k: dict(v) for k,v in chain_hops.items()}

stats = {
    "nodes_en": len(en_to_fr_d), "nodes_fr": len(fr_to_en_d),
    "sound_edges": sum(len(v) for v in en_to_fr_d.values()),
    "meaning_edges": sum(len(v) for v in fr_to_en_d.values()),
    "homophone_classes": len(fr_class),
    "synonym_edges": sum(len(v) for v in syn_en.values()) + sum(len(v) for v in syn_fr.values()),
    "chain_hops": sum(len(v) for v in chain_d.values()),
    "loops": len(loops), "zipf": len(zipf),
}

graph = {
    "en_to_fr": en_to_fr_d, "fr_to_en": fr_to_en_d,
    "fr_class": fr_class, "syn_en": {k:list(v) for k,v in syn_en.items()},
    "syn_fr": {k:list(v) for k,v in syn_fr.items()},
    "chain_hops": chain_d, "loops": loops, "zipf": zipf,
    "stats": stats,
}

# Save as pickle (full graph, fast load)
pickle.dump(graph, open("stage2_graph.pkl","wb"))
import gzip
with gzip.open("stage2_graph.pkl.gz","wb") as f:
    pickle.dump(graph, f)

size = os.path.getsize("stage2_graph.pkl")/1e6
gz_size = os.path.getsize("stage2_graph.pkl.gz")/1e6
print(f"  Saved: stage2_graph.pkl ({size:.1f}MB), stage2_graph.pkl.gz ({gz_size:.1f}MB)")
print(f"  Stats: {stats}")
