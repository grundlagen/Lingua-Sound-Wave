"""Stage 2 emit: one ladder object per gold word (PIPELINE.md build order #1).

{word, lang, homonyms[], synonyms[], senses[], routes[], pairs[]}

Sources (all extant, no GPU):
  homonyms  en-homophone-classes.tsv + fr-homophone-classes(-lexique).tsv
  synonyms  muse-pivot-syn.tsv same-language edges (weight-ranked)
  routes    hops-all.tsv typed edges incident to the word
  pairs     tier-ladder.tsv cross-language partners with their tier
  senses    [] until node-vecs.npy is regenerated (cache_graph+cache_vecs on
            a box with the MUSE artifacts) — flagged senses_pending

Vocabulary = every EN and FR word in the tier ladder (the gold units).
Output: ladder-words.jsonl, one object per line, list caps keep it small.

Usage: python build_ladder_json.py [--out ladder-words.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SYN_CAP, ROUTE_CAP, HOMO_CAP, PAIR_CAP = 12, 20, 24, 12


def load_classes(path, lang, homo):
    for i, line in enumerate(open(path, encoding="utf-8")):
        if i == 0:
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        members = parts[1].split()
        for w in members:
            homo[(lang, w)].update(m for m in members if m != w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ladder-words.jsonl")
    ap.add_argument("--include-dual-b", action="store_true",
                    help="include the noisy DUAL-B identity bucket in the vocabulary")
    args = ap.parse_args()

    # vocabulary + cross-language pairs from the tier ladder
    vocab: set[tuple[str, str]] = set()
    pairs = defaultdict(list)        # (lang, word) -> [(other_word, tier, rank)]
    for i, line in enumerate(open("tier-ladder.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 4 or not p[1] or not p[2]:
            continue
        rank, en, fr, tier = p[0], p[1].lower(), p[2].lower(), p[3]
        pairs[("en", en)].append((fr, tier, rank))
        pairs[("fr", fr)].append((en, tier, rank))
        if tier == "DUAL-B" and not args.include_dual_b:
            continue      # MUSE identity noise: pairs recorded, vocab not grown
        vocab.add(("en", en))
        vocab.add(("fr", fr))

    # homonyms (same sound, same language)
    homo: dict[tuple[str, str], set] = defaultdict(set)
    load_classes("en-homophone-classes.tsv", "en", homo)
    load_classes("fr-homophone-classes.tsv", "fr", homo)
    load_classes("fr-homophone-classes-lexique.tsv", "fr", homo)

    # synonyms (same-language MUSE pivot edges)
    syn: dict[tuple[str, str], list] = defaultdict(list)
    for line in open("muse-pivot-syn.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        a, b, w = p
        la, wa = a.split(":", 1)
        lb, wb = b.split(":", 1)
        if la == lb:
            syn[(la, wa)].append((wb, float(w)))
            syn[(lb, wb)].append((wa, float(w)))

    # routes (typed hop edges)
    routes: dict[tuple[str, str], list] = defaultdict(list)
    for i, line in enumerate(open("hops-all.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        a, b, typ, w = p
        la, wa = a.split(":", 1)
        routes[(la, wa)].append({"to": b, "type": typ, "w": float(w)})

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for (lang, word) in sorted(vocab):
            key = (lang, word)
            syns = sorted(set(syn.get(key, [])), key=lambda t: -t[1])[:SYN_CAP]
            rts = sorted(routes.get(key, []), key=lambda r: -r["w"])[:ROUTE_CAP]
            prs = pairs.get(key, [])[:PAIR_CAP]
            obj = {
                "word": word, "lang": lang,
                "homonyms": sorted(homo.get(key, set()))[:HOMO_CAP],
                "synonyms": [{"w": s, "score": sc} for s, sc in syns],
                "senses": [], "senses_pending": True,
                "routes": rts,
                "pairs": [{"other": o, "tier": t, "rank": r} for o, t, r in prs],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    filled = "words with homonyms/synonyms/routes"
    print(f"{n} ladder objects -> {args.out}")
    for field in ("homonyms", "synonyms", "routes"):
        have = sum(1 for (lg, w) in vocab if
                   (field == "homonyms" and homo.get((lg, w))) or
                   (field == "synonyms" and syn.get((lg, w))) or
                   (field == "routes" and routes.get((lg, w))))
        print(f"  {field:9s}: {have}/{n} words covered")


if __name__ == "__main__":
    main()
