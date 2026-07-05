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

# ── 4. Add chain-web transitive hops (EVERY NODE in the subchain) ─────
print("[5/7] Adding chain-web nodes (every step is a node)...")
chain_count = 0
chain_file = "chain-web/archive/chain-web-full-v7u.tsv"
if not os.path.exists(chain_file):
    chain_file = "chain-web-full-v7u.tsv"
if os.path.exists(chain_file):
    with open(chain_file) as f:
        f.readline()
        for line_num, line in enumerate(f):
            if chain_count >= 8000: break
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                a, b, hops, quality, subchain = parts[0], parts[1], int(parts[2]), float(parts[3]), parts[4]

                # Parse the subchain: "~ fr:lit en:lee ~ fr:lee"
                # Each "en:X ~ fr:Y" is a valid homophone node at some hop position
                tokens = subchain.split()
                hop_pos = 0
                prev_en = None
                for i, tok in enumerate(tokens):
                    if tok == "~":  # sound edge follows
                        continue
                    if tok.startswith("en:"):
                        prev_en = tok[3:].lower()
                    elif tok.startswith("fr:") and prev_en:
                        fr_word = tok[3:].lower()
                        en_word = prev_en
                        hop_pos += 1
                        if en_word and fr_word and en_word != fr_word and 2 <= len(en_word) <= 15 and 2 <= len(fr_word) <= 15:
                            if not any(r["en"] == en_word and r["fr"] == fr_word for r in rows):
                                rows.append({
                                    "en": en_word, "fr": fr_word,
                                    "en_ipa": "", "fr_ipa": "",
                                    "tier": "B",
                                    "score": quality * (0.95 ** (hop_pos - 1)),
                                    "alignment": "", "pivot": "",
                                    "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                                    "gap_ratio": 0.0, "usable": 1,
                                    "chunk_recipe": "", "en_onset": "", "en_coda": "",
                                    "fr_onset": "", "fr_coda": "",
                                    "loop_certified": False,
                                    "chain_certified": True,
                                    "graph_hops": hop_pos,
                                    "graph_depth": hops,
                                    "meaning_proximity": 0.9 ** hop_pos,
                                    "source": "chain_web",
                                    "graph_source": f"chain_step_{hop_pos}"
                                })
                                chain_count += 1
                        prev_en = None  # reset after making the pair

                # Also add the endpoint pair (a→b)
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
                                    "graph_hops": hops, "graph_depth": hops,
                                    "meaning_proximity": 0.9 ** hops,
                                    "source": "chain_web", "graph_source": f"chain_endpoint_h{hops}"
                                })
                                chain_count += 1
print(f"  Added {chain_count} chain-web nodes (every step is a training pair)")

# ── 5. Add round-rabbit attachments ──────────────────────────────────
print("[6/7] Adding round-rabbit attachments...")
rr_count = 0
if os.path.exists("round-rabbit.json"):
    rr = json.load(open("round-rabbit.json"))
    for row_data in rr["rows"]:
        hops = row_data.get("homophonic_hops", 0)
        for att in row_data.get("attachments", []):
            en = att["en"].strip().lower()
            fr = att["fr"].strip().lower()
            score = float(att.get("score", 0))
            tier = att.get("tier", "B")
            if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
                if not any(r["en"] == en and r["fr"] == fr for r in rows):
                    rows.append({
                        "en": en, "fr": fr,
                        "en_ipa": "", "fr_ipa": "",
                        "tier": tier, "score": score,
                        "alignment": "", "pivot": "",
                        "en_syll": 0, "fr_syll": 0, "syllable_delta": 0,
                        "gap_ratio": 0.0, "usable": 1,
                        "chunk_recipe": "", "en_onset": "", "en_coda": "",
                        "fr_onset": "", "fr_coda": "",
                        "loop_certified": False, "chain_certified": False,
                        "graph_hops": hops,
                        "meaning_proximity": 0.95 ** hops,
                        "source": "round_rabbit", "graph_source": f"rabbit_h{hops}"
                    })
                    rr_count += 1
print(f"  Added {rr_count} round-rabbit rows")

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
