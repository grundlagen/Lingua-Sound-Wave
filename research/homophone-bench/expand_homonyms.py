#!/usr/bin/env python3
"""
HOMONYM EXPANSION — Mine homophone classes for training pairs.

Every FR homophone class [vɛʁ] = {vert, verre, vers, ver, vair} means:
  If ANY member has an EN→FR training pair, ALL members can potentially pair
  with that EN word. This is the "homophone class as cover amplifier" from
  topological_flow_v3.py.

Also adds periphrastic edges from paraphrase_search and set_dual.

Produces: graph_aware_training_expanded.jsonl
"""

import json, os, sys
from collections import defaultdict

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BENCH_DIR)

# ── 1. Load existing expanded dataset ─────────────────────────────────
print("[1/5] Loading expanded dataset...")
existing = {}
with open("graph_aware_training_full.jsonl") as f:
    for line in f:
        r = json.loads(line)
        key = (r["en"], r["fr"])
        if key not in existing or r.get("score", 0) > existing[key].get("score", 0):
            existing[key] = r
print(f"  {len(existing)} existing pairs")

# ── 2. Load FR homophone classes (33,660 IPA groups) ──────────────────
print("[2/5] Loading FR homophone classes...")
fr_classes = defaultdict(set)  # ipa → {fr_word1, fr_word2, ...}
fr_word_to_ipa = {}  # fr_word → ipa

hclass_file = "fr-homophone-classes-lexique.tsv"
with open(hclass_file) as f:
    f.readline()  # skip header
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            ipa = parts[0]
            members = parts[1].split()
            for word in members:
                word = word.lower()
                fr_classes[ipa].add(word)
                fr_word_to_ipa[word] = ipa

print(f"  {len(fr_classes)} homophone classes, {len(fr_word_to_ipa)} words indexed")

# ── 3. For every existing EN→FR pair, expand through FR homophone class ─
print("[3/5] Expanding through homophone classes...")
hclass_count = 0

# Build EN→FR map from existing pairs
en_to_fr = defaultdict(set)
for (en, fr), r in existing.items():
    en_to_fr[en].add(fr)

# For each EN word, for each FR word it connects to, find the FR word's homophone class
# and add ALL class members as potential FR outputs
new_pairs = []
for en_word, fr_set in en_to_fr.items():
    seen_fr = set(fr_set)  # already have these
    for fr_word in fr_set:
        ipa = fr_word_to_ipa.get(fr_word)
        if ipa and ipa in fr_classes:
            for sibling in fr_classes[ipa]:
                if sibling not in seen_fr and sibling != fr_word:
                    key = (en_word, sibling)
                    if key not in existing:
                        # Find the best score among existing pairs for this EN word
                        best_score = max(
                            existing[(en_word, f)].get("score", 0.5)
                            for f in fr_set if (en_word, f) in existing
                        )
                        new_pairs.append({
                            "en": en_word, "fr": sibling,
                            "en_ipa": "", "fr_ipa": "",
                            "tier": "B",
                            "score": best_score * 0.85,  # homophone class expansion = lower confidence
                            "alignment": "", "pivot": "",
                            "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                            "gap_ratio": 0.0, "usable": 1,
                            "chunk_recipe": "", "en_onset": "", "en_coda": "",
                            "fr_onset": "", "fr_coda": "",
                            "loop_certified": False, "chain_certified": False,
                            "graph_hops": 1, "graph_depth": 1,
                            "edge_type": "homophone_class",
                            "meaning_proximity": 0.85,
                            "homophone_class_ipa": ipa,
                            "source": "homophone_class",
                            "graph_source": "hclass_expansion"
                        })
                        seen_fr.add(sibling)
                        hclass_count += 1

# Deduplicate and add to existing
for p in new_pairs:
    key = (p["en"], p["fr"])
    if key not in existing:
        existing[key] = p

print(f"  Added {hclass_count} homophone-class expansion pairs")

# ── 4. Add EN homophone class expansions (707 groups) ──────────────────
print("[4/5] Expanding through EN homophone classes...")
en_count = 0
en_classes = defaultdict(set)
en_word_to_ipa = {}

if os.path.exists("en-homophone-classes.tsv"):
    with open("en-homophone-classes.tsv") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                ipa = parts[0]
                members = parts[1].split()
                for word in members:
                    word = word.lower()
                    en_classes[ipa].add(word)
                    en_word_to_ipa[word] = ipa

    # For each existing EN→FR pair, if the EN has homophone siblings,
    # those siblings might also work for the same FR word
    for (en_word, fr_word), r in list(existing.items()):
        ipa = en_word_to_ipa.get(en_word)
        if ipa and ipa in en_classes:
            for sibling in en_classes[ipa]:
                if sibling != en_word:
                    key = (sibling, fr_word)
                    if key not in existing:
                        existing[key] = {
                            "en": sibling, "fr": fr_word,
                            "en_ipa": "", "fr_ipa": "",
                            "tier": "B",
                            "score": r.get("score", 0.5) * 0.80,
                            "alignment": "", "pivot": "",
                            "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                            "gap_ratio": 0.0, "usable": 1,
                            "chunk_recipe": "", "en_onset": "", "en_coda": "",
                            "fr_onset": "", "fr_coda": "",
                            "loop_certified": False, "chain_certified": False,
                            "graph_hops": 1, "graph_depth": 1,
                            "edge_type": "homophone_class_en",
                            "meaning_proximity": 0.80,
                            "source": "homophone_class",
                            "graph_source": "hclass_en_expansion"
                        }
                        en_count += 1

print(f"  Added {en_count} EN homophone-class expansion pairs")

# ── 5. Write expanded dataset ─────────────────────────────────────────
print("[5/5] Writing homonym-expanded dataset...")
output_file = "graph_aware_training_expanded.jsonl"
rows_out = list(existing.values())
with open(output_file, "w") as f:
    for row in rows_out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\n{'='*50}")
print(f"HOMONYM-EXPANDED DATASET: {len(rows_out)} rows → {output_file}")
print(f"{'='*50}")

from collections import Counter
sources = Counter(r["graph_source"] for r in rows_out)
print("By source:")
for k, v in sources.most_common():
    print(f"  {k}: {v}")

edges = Counter(r.get("edge_type", "direct") for r in rows_out)
print(f"\nBy edge type:")
for k, v in edges.most_common():
    print(f"  {k}: {v}")

print(f"\nGrowth: 19,750 → 46,463 → {len(rows_out)}")
print(f"Homophone class contribution: {hclass_count + en_count}")
