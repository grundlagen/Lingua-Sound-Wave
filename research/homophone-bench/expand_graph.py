#!/usr/bin/env python3
"""
GRAPH EXPANSION — Mine the entire chain-web and mapping-web for training pairs.
Uncaps the 10k limit. Extracts every reachable word pair.

Produces: graph_aware_training_full.jsonl (~150-200K rows)
"""

import json, os, sys
from collections import defaultdict

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BENCH_DIR)

# ── 1. Load existing dataset ──────────────────────────────────────────
print("[1/6] Loading existing dataset...")
existing = {}
with open("graph_aware_training.jsonl") as f:
    for line in f:
        r = json.loads(line)
        key = (r["en"], r["fr"])
        if key not in existing or r.get("score", 0) > existing[key].get("score", 0):
            existing[key] = r
rows = list(existing.values())
print(f"  {len(rows)} unique pairs")

# ── 2. Mine ALL chain-web paths (uncapped) ────────────────────────────
print("[2/6] Mining ALL chain-web paths (uncapped)...")
EDGE_SYMBOLS = {"~": "sound", "≈": "meaning", "≡": "synonym"}
chain_count = 0

chain_file = "chain-web/archive/chain-web-full-v7u.tsv"
if not os.path.exists(chain_file):
    chain_file = "chain-web-full-v7u.tsv"

if os.path.exists(chain_file):
    with open(chain_file) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            a, b, total_hops, quality, subchain = parts[0], parts[1], int(parts[2]), float(parts[3]), parts[4]

            tokens = subchain.split()
            current_edge_type = None
            prev_node = None
            walk_step = 0

            for tok in tokens:
                if tok in EDGE_SYMBOLS:
                    current_edge_type = EDGE_SYMBOLS[tok]
                    continue
                if ":" in tok:
                    lang, word = tok.split(":", 1)
                    word = word.lower()
                    if prev_node and current_edge_type:
                        if prev_node[0] == "en" and lang == "fr" and current_edge_type == "sound":
                            en_word, fr_word = prev_node[1], word
                            walk_step += 1
                            if en_word and fr_word and en_word != fr_word and 2 <= len(en_word) <= 15 and 2 <= len(fr_word) <= 15:
                                key = (en_word, fr_word)
                                if key not in existing:
                                    existing[key] = {
                                        "en": en_word, "fr": fr_word,
                                        "en_ipa": "", "fr_ipa": "", "tier": "B",
                                        "score": quality * (0.93 ** (walk_step - 1)),
                                        "alignment": "", "pivot": "",
                                        "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                                        "gap_ratio": 0.0, "usable": 1,
                                        "chunk_recipe": "", "en_onset": "", "en_coda": "",
                                        "fr_onset": "", "fr_coda": "",
                                        "loop_certified": False, "chain_certified": True,
                                        "graph_hops": walk_step, "graph_depth": total_hops,
                                        "edge_type": "sound" if current_edge_type == "sound" else "meaning_reverse",
                                        "meaning_proximity": 0.9 ** walk_step,
                                        "source": "chain_web",
                                        "graph_source": f"walk_{current_edge_type}_step{walk_step}"
                                    }
                                    chain_count += 1
                        elif prev_node[0] == "fr" and lang == "en" and current_edge_type == "meaning":
                            fr_word, en_word = prev_node[1], word
                            if en_word and fr_word and 2 <= len(en_word) <= 15 and 2 <= len(fr_word) <= 15:
                                key = (en_word, fr_word)
                                if key not in existing:
                                    existing[key] = {
                                        "en": en_word, "fr": fr_word,
                                        "en_ipa": "", "fr_ipa": "", "tier": "B",
                                        "score": quality * (0.88 ** (walk_step - 1)),
                                        "alignment": "", "pivot": "",
                                        "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                                        "gap_ratio": 0.0, "usable": 1,
                                        "chunk_recipe": "", "en_onset": "", "en_coda": "",
                                        "fr_onset": "", "fr_coda": "",
                                        "loop_certified": False, "chain_certified": True,
                                        "graph_hops": walk_step, "graph_depth": total_hops,
                                        "edge_type": "meaning_reverse",
                                        "meaning_proximity": 0.85 ** walk_step,
                                        "source": "chain_web",
                                        "graph_source": f"walk_meaning_step{walk_step}"
                                    }
                                    chain_count += 1
                    prev_node = (lang, word)

            # Endpoint
            if ":" in a and ":" in b:
                sl, sw = a.split(":", 1)
                tl, tw = b.split(":", 1)
                if sl == "en" and tl == "fr":
                    en, fr = sw.lower(), tw.lower()
                    if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
                        key = (en, fr)
                        if key not in existing:
                            existing[key] = {
                                "en": en, "fr": fr,
                                "en_ipa": "", "fr_ipa": "", "tier": "B",
                                "score": quality, "alignment": "", "pivot": "",
                                "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                                "gap_ratio": 0.0, "usable": 1,
                                "chunk_recipe": "", "en_onset": "", "en_coda": "",
                                "fr_onset": "", "fr_coda": "",
                                "loop_certified": False, "chain_certified": True,
                                "graph_hops": total_hops, "graph_depth": total_hops,
                                "edge_type": "walk_endpoint",
                                "meaning_proximity": 0.9 ** total_hops,
                                "source": "chain_web",
                                "graph_source": f"walk_endpoint_h{total_hops}"
                            }
                            chain_count += 1

            if chain_count % 20000 == 0:
                print(f"  ... {chain_count} new pairs so far")

print(f"  Total new from chain-web: {chain_count}")

# ── 3. Mine ALL sound edges from mapping-web ──────────────────────────
print("[3/6] Mining mapping-web sound edges...")
web_count = 0
if os.path.exists("mapping-web.json"):
    web = json.load(open("mapping-web.json"))
    for edge in web.get("edges", []):
        if edge.get("type") != "sound_edge":
            continue
        src = edge["source"]
        tgt = edge["target"]
        for p in ["en:", "fr:", "sound:en:", "sound:fr:"]:
            src = src.replace(p, "")
            tgt = tgt.replace(p, "")
        en, fr = src, tgt
        if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
            score = float(edge.get("score", 0.5))
            key = (en, fr)
            if key not in existing:
                existing[key] = {
                    "en": en, "fr": fr,
                    "en_ipa": "", "fr_ipa": "", "tier": "B",
                    "score": score, "alignment": "", "pivot": "",
                    "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                    "gap_ratio": 0.0, "usable": 1,
                    "chunk_recipe": "", "en_onset": "", "en_coda": "",
                    "fr_onset": "", "fr_coda": "",
                    "loop_certified": False, "chain_certified": False,
                    "graph_hops": 0, "graph_depth": 0,
                    "edge_type": "sound",
                    "meaning_proximity": 1.0,
                    "source": "mapping_web",
                    "graph_source": "mapping_web_sound"
                }
                web_count += 1
print(f"  New from mapping-web: {web_count}")

# ── 4. Add meaning edges as reverse training pairs ────────────────────
print("[4/6] Adding meaning-edge reverse pairs...")
meaning_count = 0
if os.path.exists("mapping-web.json"):
    for edge in web.get("edges", []):
        if edge.get("type") != "meaning_edge":
            continue
        src = edge["source"]
        tgt = edge["target"]
        for p in ["en:", "fr:", "sound:en:", "sound:fr:"]:
            src = src.replace(p, "")
            tgt = tgt.replace(p, "")
        # If src is FR and tgt is EN, this is a meaning pair: en_word goes with fr_word
        # (the meaning edge says they mean the same thing)
        if src and tgt and src != tgt and 2 <= len(src) <= 15 and 2 <= len(tgt) <= 15:
            key = (tgt, src)  # en=tgt, fr=src
            if key not in existing:
                existing[key] = {
                    "en": tgt, "fr": src,
                    "en_ipa": "", "fr_ipa": "", "tier": "A",
                    "score": 0.95, "alignment": "", "pivot": "",
                    "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                    "gap_ratio": 0.0, "usable": 1,
                    "chunk_recipe": "", "en_onset": "", "en_coda": "",
                    "fr_onset": "", "fr_coda": "",
                    "loop_certified": False, "chain_certified": False,
                    "graph_hops": 0, "graph_depth": 0,
                    "edge_type": "meaning",
                    "meaning_proximity": 1.0,
                    "source": "mapping_web",
                    "graph_source": "mapping_web_meaning"
                }
                meaning_count += 1
print(f"  New meaning pairs: {meaning_count}")

# ── 5. Write expanded dataset ─────────────────────────────────────────
print("[5/6] Writing expanded dataset...")
output_file = "graph_aware_training_full.jsonl"
rows_out = list(existing.values())
with open(output_file, "w") as f:
    for row in rows_out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ── 6. Stats ──────────────────────────────────────────────────────────
print(f"\n[6/6] {'='*50}")
print(f"EXPANDED DATASET: {len(rows_out)} rows → {output_file}")
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

hops = Counter(r.get("graph_hops", 0) for r in rows_out)
print(f"\nHops: 0={hops.get(0,0)} 1-2={hops.get(1,0)+hops.get(2,0)} 3-6={sum(hops.get(i,0) for i in range(3,7))}")

print(f"\nGrowth: {len(existing)} pairs from {len(rows_out)} total")
print(f"Chain-web contribution: {chain_count}")
print(f"Mapping-web contribution: {web_count + meaning_count}")
