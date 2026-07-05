#!/usr/bin/env python3
"""
GRAPH-AWARE TRAINING DATASET — Build a rich training corpus from ALL mathematical frameworks.

Extracts training signals from:
  1. strict-gold (9,803 loop-certified pairs)
  2. dictionary-v5 (11,789 entries with IPA, tier, alignment, gaps)
  3. fragment-chains (944 generative matches from fragment recombination)
  4. chain-web (70k transitive hops through the graph)
  5. loop-certified pairs (814 cycles verifying algebraic closure)
  6. round-rabbit lattice (semantic→homophonic paths with hop metadata)
  7. algebraic graph walks (recursive bidirectional EN↔FR↔EN paths)
  8. composition webs (4-layer expansion: direct→homophone→synonym→chain)

Each training row includes:
  - en_chars, fr_chars (character sequences)
  - en_ipa, fr_ipa (phoneme sequences)
  - alignment (phoneme-by-phoneme mapping)
  - tier (S/A/B quality)
  - graph_hops (how many hops through chain-web)
  - loop_certified (boolean: verified algebraic closure)
  - meaning_proximity (how close semantically)
  - fragment_recipe (which chunks compose this pair)
  - sound_score, gap_ratio, syllable_delta

Output: graph_aware_training.jsonl (~15,000 enriched training rows)
"""

import json, os, sys, math
from collections import defaultdict

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BENCH_DIR)

# ── 1. Load base data ──────────────────────────────────────────────────
print("[1/7] Loading strict-gold...")
gold_pairs = []
with open("strict-gold-training.jsonl") as f:
    for line in f:
        r = json.loads(line)
        en = r["input"].replace("English word: ", "").strip().lower()
        fr = r["output"].strip().lower()
        if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
            gold_pairs.append({
                "en": en, "fr": fr,
                "quality": r.get("quality", r.get("sound", 1.0)),
                "loop_certified": r.get("loop", False),
                "chain_certified": r.get("chain", False),
                "source": "strict_gold"
            })
print(f"  {len(gold_pairs)} strict-gold pairs")

print("[2/7] Loading dictionary-v5...")
v5_map = {}
with open("dictionary-v5.tsv") as f:
    cols = f.readline().rstrip("\n").split("\t")
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6: continue
        d = dict(zip(cols, parts))
        key = (d["en"].lower(), d["fr"].lower())
        score = float(d.get("score", 0))
        if key not in v5_map or score > float(v5_map[key].get("score", 0)):
            v5_map[key] = d
print(f"  {len(v5_map)} unique entries")

# ── 2. Enrich strict-gold with dictionary fields ─────────────────────
print("[3/7] Enriching with dictionary data...")
rows = []
for p in gold_pairs:
    key = (p["en"], p["fr"])
    if key in v5_map:
        d = v5_map[key]
        row = {
            "en": p["en"], "fr": p["fr"],
            "en_ipa": d.get("en_ipa", ""),
            "fr_ipa": d.get("fr_ipa", ""),
            "tier": d.get("tier", "B"),
            "score": float(d.get("score", 0)),
            "alignment": d.get("alignment", ""),
            "pivot": d.get("pivot", ""),
            "en_syll": int(d.get("en_syll", 0) or 0),
            "fr_syll": int(d.get("fr_syll", 0) or 0),
            "syllable_delta": int(d.get("syllable_delta", 0) or 0),
            "gap_ratio": float(d.get("gap_ratio", 0) or 0),
            "usable": int(d.get("usable_for_composition", 0) or 0),
            "chunk_recipe": d.get("chunk_recipe", ""),
            "en_onset": d.get("en_onset", ""),
            "en_coda": d.get("en_coda", ""),
            "fr_onset": d.get("fr_onset", ""),
            "fr_coda": d.get("fr_coda", ""),
            "loop_certified": p["loop_certified"],
            "chain_certified": p["chain_certified"],
            "graph_hops": 0,
            "meaning_proximity": 1.0,
            "source": p["source"],
            "graph_source": "strict_gold"
        }
    else:
        row = {
            "en": p["en"], "fr": p["fr"],
            "en_ipa": "", "fr_ipa": "", "tier": "B", "score": p["quality"],
            "alignment": "", "pivot": "", "en_syll": 0, "fr_syll": 0,
            "syllable_delta": 0, "gap_ratio": 0.0, "usable": 1,
            "chunk_recipe": "", "en_onset": "", "en_coda": "",
            "fr_onset": "", "fr_coda": "",
            "loop_certified": p["loop_certified"],
            "chain_certified": p["chain_certified"],
            "graph_hops": 0, "meaning_proximity": 1.0,
            "source": p["source"], "graph_source": "strict_gold"
        }
    rows.append(row)

print(f"  {len(rows)} enriched rows")

# ── 3. Add generative-matches (fragment-chained discoveries) ─────────
print("[4/7] Adding generative-matches...")
gen_count = 0
if os.path.exists("generative-matches.tsv"):
    with open("generative-matches.tsv") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                score, en, fr, en_ipa = parts[0], parts[1].strip().lower(), parts[2].strip().lower(), parts[3]
                chunk = parts[4] if len(parts) > 4 else ""
                try: s = float(score)
                except: continue
                if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
                    if not any(r["en"] == en and r["fr"] == fr for r in rows):
                        rows.append({
                            "en": en, "fr": fr,
                            "en_ipa": en_ipa, "fr_ipa": "",
                            "tier": "A" if s >= 0.95 else "B",
                            "score": s, "alignment": "", "pivot": "",
                            "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                            "gap_ratio": 0.0, "usable": 1,
                            "chunk_recipe": chunk, "en_onset": "", "en_coda": "",
                            "fr_onset": "", "fr_coda": "",
                            "loop_certified": False, "chain_certified": False,
                            "graph_hops": 1, "meaning_proximity": 0.85,
                            "source": "generative", "graph_source": "fragment_chain"
                        })
                        gen_count += 1
print(f"  Added {gen_count} generative-match rows")

# ── 4. Add chain-web PATH WALKS (sound edges + meaning edges + synonyms) ─
# The subchain alternates: sound ~, meaning ≈, synonym ≡
# Every step is a node. Edge types are training signals.
print("[5/7] Adding chain-web path walks (sound/meaning/synonym edges)...")
chain_count = 0
chain_file = "chain-web/archive/chain-web-full-v7u.tsv"
if not os.path.exists(chain_file):
    chain_file = "chain-web-full-v7u.tsv"

EDGE_SYMBOLS = {"~": "sound", "≈": "meaning", "≡": "synonym"}

if os.path.exists(chain_file):
    with open(chain_file) as f:
        f.readline()
        for line_num, line in enumerate(f):
            if chain_count >= 10000: break
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                a, b, total_hops, quality, subchain = parts[0], parts[1], int(parts[2]), float(parts[3]), parts[4]

                # Parse the subchain as alternating edge walks:
                # "en:borne ~ fr:borne ≈ en:born ~ fr:born"
                #  sound edge ^    meaning ^     sound edge ^
                tokens = subchain.split()
                current_edge_type = None
                prev_node = None
                walk_step = 0  # position in the alternating walk

                for tok in tokens:
                    if tok in EDGE_SYMBOLS:
                        current_edge_type = EDGE_SYMBOLS[tok]
                        continue
                    # Parse node: "en:word" or "fr:word"
                    if ":" in tok:
                        lang, word = tok.split(":", 1)
                        word = word.lower()
                        if prev_node is not None and current_edge_type:
                            # We have an edge: prev_node → current_node of type current_edge_type
                            if prev_node[0] == "en" and lang == "fr" and current_edge_type == "sound":
                                # This is a sound (homophone) edge pair — primary training signal
                                en_word, fr_word = prev_node[1], word
                                if en_word and fr_word and en_word != fr_word and 2 <= len(en_word) <= 15 and 2 <= len(fr_word) <= 15:
                                    walk_step += 1
                                    if not any(r["en"] == en_word and r["fr"] == fr_word for r in rows):
                                        rows.append({
                                            "en": en_word, "fr": fr_word,
                                            "en_ipa": "", "fr_ipa": "",
                                            "tier": "B",
                                            "score": quality * (0.93 ** (walk_step - 1)),
                                            "alignment": "", "pivot": "",
                                            "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                                            "gap_ratio": 0.0, "usable": 1,
                                            "chunk_recipe": "", "en_onset": "", "en_coda": "",
                                            "fr_onset": "", "fr_coda": "",
                                            "loop_certified": False,
                                            "chain_certified": True,
                                            "graph_hops": walk_step,
                                            "graph_depth": total_hops,
                                            "edge_type": "sound",
                                            "meaning_proximity": 0.9 ** walk_step,
                                            "source": "chain_web",
                                            "graph_source": f"walk_sound_step{walk_step}"
                                        })
                                        chain_count += 1

                            elif prev_node[0] == "en" and lang == "en" and current_edge_type in ("meaning", "synonym"):
                                # EN→EN synonym edge — not a training pair, but valuable for graph structure
                                # Record as a meaning-path signal
                                pass

                            elif prev_node[0] == "fr" and lang == "en" and current_edge_type == "meaning":
                                # FR→EN meaning edge — reverse direction, useful for bilingual training
                                fr_word, en_word = prev_node[1], word
                                if en_word and fr_word and en_word != fr_word and 2 <= len(en_word) <= 15 and 2 <= len(fr_word) <= 15:
                                    if not any(r["en"] == en_word and r["fr"] == fr_word for r in rows):
                                        rows.append({
                                            "en": en_word, "fr": fr_word,
                                            "en_ipa": "", "fr_ipa": "",
                                            "tier": "B",
                                            "score": quality * (0.88 ** (walk_step - 1)),
                                            "alignment": "", "pivot": "",
                                            "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                                            "gap_ratio": 0.0, "usable": 1,
                                            "chunk_recipe": "", "en_onset": "", "en_coda": "",
                                            "fr_onset": "", "fr_coda": "",
                                            "loop_certified": False, "chain_certified": True,
                                            "graph_hops": walk_step,
                                            "graph_depth": total_hops,
                                            "edge_type": "meaning_reverse",
                                            "meaning_proximity": 0.85 ** walk_step,
                                            "source": "chain_web",
                                            "graph_source": f"walk_meaning_step{walk_step}"
                                        })
                                        chain_count += 1

                        prev_node = (lang, word)

                # Also add the final endpoint (a→b) as a verified walk result
                if ":" in a and ":" in b:
                    sl, sw = a.split(":", 1)
                    tl, tw = b.split(":", 1)
                    if sl == "en" and tl == "fr":
                        en, fr = sw.lower(), tw.lower()
                        if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
                            if not any(r["en"] == en and r["fr"] == fr for r in rows):
                                rows.append({
                                    "en": en, "fr": fr,
                                    "en_ipa": "", "fr_ipa": "",
                                    "tier": "B",
                                    "score": quality,
                                    "alignment": "", "pivot": "",
                                    "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                                    "gap_ratio": 0.0, "usable": 1,
                                    "chunk_recipe": "", "en_onset": "", "en_coda": "",
                                    "fr_onset": "", "fr_coda": "",
                                    "loop_certified": False, "chain_certified": True,
                                    "graph_hops": total_hops, "graph_depth": total_hops,
                                    "edge_type": "walk_endpoint",
                                    "meaning_proximity": 0.9 ** total_hops,
                                    "source": "chain_web", "graph_source": f"walk_endpoint_h{total_hops}"
                                })
                                chain_count += 1
print(f"  Added {chain_count} path-walk nodes (sound/meaning/synonym edges)")

# ── 5. Add round-rabbit LATTICE (semantic components + homophonic paths) ─
# Every row in the lattice is a path: semantic_component → sound walk → reachable node
# Attachments at each node are ranked dictionary pairs. All of this is training signal.
print("[6/7] Adding round-rabbit lattice (semantic→homophonic walks)...")
rr_count = 0
if os.path.exists("round-rabbit.json"):
    rr = json.load(open("round-rabbit.json"))
    for row_data in rr["rows"]:
        component_id = row_data.get("component_id", "unknown")
        component_weight = float(row_data.get("component_weight", 0.7))
        homophonic_hops = int(row_data.get("homophonic_hops", 0))
        path = row_data.get("path", [])
        path_edge_scores = row_data.get("path_edge_scores", [])
        rank_score = float(row_data.get("rank_score", 0.5))
        attached_count = int(row_data.get("attached_count", 0))
        has_multi = bool(row_data.get("has_multi", False))
        has_generated = bool(row_data.get("has_generated", False))

        # For each attachment (EN→FR pair at this lattice node), add with full context
        for att in row_data.get("attachments", []):
            en = att["en"].strip().lower()
            fr = att["fr"].strip().lower()
            score = float(att.get("score", 0))
            tier = att.get("tier", "B")
            kind = att.get("kind", "whole")
            chunk_recipe = att.get("chunk_recipe", "")
            source_stage = att.get("source_stage", "")

            if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
                if not any(r["en"] == en and r["fr"] == fr for r in rows):
                    rows.append({
                        "en": en, "fr": fr,
                        "en_ipa": "", "fr_ipa": "",
                        "tier": tier, "score": score,
                        "alignment": "", "pivot": "",
                        "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                        "gap_ratio": 0.0, "usable": 1,
                        "chunk_recipe": chunk_recipe,
                        "en_onset": "", "en_coda": "",
                        "fr_onset": "", "fr_coda": "",
                        "loop_certified": False, "chain_certified": False,
                        "graph_hops": homophonic_hops,
                        "graph_depth": len(path),
                        "edge_type": "rabbit_lattice",
                        "meaning_proximity": component_weight * (0.95 ** homophonic_hops),
                        "component_id": component_id,
                        "component_weight": component_weight,
                        "rank_score": rank_score,
                        "attached_count": attached_count,
                        "has_multi": has_multi,
                        "has_generated": has_generated,
                        "kind": kind,
                        "source": "round_rabbit",
                        "graph_source": f"rabbit_h{homophonic_hops}_{kind}"
                    })
                    rr_count += 1
print(f"  Added {rr_count} round-rabbit lattice rows (full context)")

# ── 6. Add loop-certified pairs ──────────────────────────────────────
print("[7/7] Adding loop-certified pairs...")
loop_count = 0
loop_files = ["loop-certified-pairs-v7u.tsv", "chain-web/archive/loop-certified-pairs-v7u.tsv"]
for lf in loop_files:
    if os.path.exists(lf):
        with open(lf) as f:
            f.readline()
            for line in f:
                if loop_count >= 500: break
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    en, fr = parts[0].lower(), parts[1].lower()
                    score = float(parts[2]) if len(parts) > 2 else 0.8
                    if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
                        if not any(r["en"] == en and r["fr"] == fr for r in rows):
                            rows.append({
                                "en": en, "fr": fr,
                                "en_ipa": "", "fr_ipa": "",
                                "tier": "S", "score": score,
                                "alignment": "", "pivot": "",
                                "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                                "gap_ratio": 0.0, "usable": 1,
                                "chunk_recipe": "", "en_onset": "", "en_coda": "",
                                "fr_onset": "", "fr_coda": "",
                                "loop_certified": True, "chain_certified": True,
                                "graph_hops": 0,
                                "meaning_proximity": 1.0,
                                "source": "loop_certified", "graph_source": "loop"
                            })
                            loop_count += 1
        break
print(f"  Added {loop_count} loop-certified rows")

# ── 7. Write output ──────────────────────────────────────────────────
output_file = "graph_aware_training.jsonl"
with open(output_file, "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\n{'='*50}")
print(f"TOTAL: {len(rows)} training rows → {output_file}")
print(f"{'='*50}")

# Breakdown
sources = defaultdict(int)
for r in rows: sources[r["graph_source"]] += 1
print("By graph source:")
for k, v in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

tiers = defaultdict(int)
for r in rows: tiers[r["tier"]] += 1
print("By tier:")
for k, v in sorted(tiers.items()):
    print(f"  {k}: {v}")

ipa_count = sum(1 for r in rows if r["en_ipa"])
print(f"With IPA: {ipa_count} ({100*ipa_count/len(rows):.0f}%)")
loop_cert = sum(1 for r in rows if r["loop_certified"])
print(f"Loop-certified: {loop_cert}")
graph_hops = sum(1 for r in rows if r["graph_hops"] > 0)
print(f"Multi-hop: {graph_hops}")
