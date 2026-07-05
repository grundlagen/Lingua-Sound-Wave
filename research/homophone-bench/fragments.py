"""Fragment layer: turn v5's alignments into a generative chunk grammar.

Insight (user's): v5 entries are not just word matches — each alignment is a
sequence of small attested EN→FR sound blocks, often with French word
boundaries running through them. Extracting those blocks gives a fragment
index; chaining fragments proposes NEW candidate matches that no pairwise
or decoder pass ever looked at; the existing matcher/decoder re-scores as
the arbiter. Proposal grammar + arbiter, exactly the v5 pipeline shape.

Stages:
  1. extract maximal cleanly-matched runs (cost < 0.30, no gaps, len >= 2)
     from every usable entry's `align`;
  2. index them (fragments.tsv: en_chunk, fr_chunk, count, examples);
  3. generative probe: chain two frequent fragment EN-sides, look the result
     up in the English pronunciation lexicon — if it is a real English word
     with no usable v5 entry yet, decode it for a legitimate French side and
     keep it if the arbiter scores >= 0.88 (gates as in the decoder).

Run: python fragments.py
Outputs: fragments.tsv, generative-matches.tsv
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache

import matcher
from matcher import _canonical
import phonetic_decoder as pd
from lexicon_g2p import load_en

matcher._vecs = lru_cache(maxsize=None)(matcher._vecs.__wrapped__)
matcher._segs = lru_cache(maxsize=None)(matcher._segs.__wrapped__)

MAX_COST = 0.30
MIN_LEN = 2


def extract_runs(align):
    runs, cur = [], []
    for x, y, c in align:
        if x != "·" and y != "·" and c < MAX_COST:
            cur.append((x, y))
        else:
            if len(cur) >= MIN_LEN:
                runs.append(cur)
            cur = []
    if len(cur) >= MIN_LEN:
        runs.append(cur)
    return runs


def main():
    entries = json.load(open("dictionary-v5.json"))
    usable = [e for e in entries if e.get("usable_for_composition")]

    frags = Counter()
    examples = defaultdict(list)
    for e in usable:
        for run in extract_runs(e.get("align") or []):
            # all sub-chunks of length 2..4 — small parts of words, as asked
            for ln in range(MIN_LEN, min(4, len(run)) + 1):
                for i in range(len(run) - ln + 1):
                    chunk = tuple(run[i:i + ln])
                    en_seq = tuple(x for x, _ in chunk)
                    fr_seq = tuple(y for _, y in chunk)
                    frags[(en_seq, fr_seq)] += 1
                    if len(examples[(en_seq, fr_seq)]) < 3:
                        examples[(en_seq, fr_seq)].append(f"{e['en']}~{e['fr']}")

    with open("fragments.tsv", "w") as f:
        f.write("count\ten_chunk\tfr_chunk\texamples\n")
        for (en_seq, fr_seq), n in frags.most_common():
            if n < 2:
                continue
            f.write(f"{n}\t{''.join(en_seq)}\t{''.join(fr_seq)}"
                    f"\t{'; '.join(examples[(en_seq, fr_seq)])}\n")
    print(f"fragment index: {sum(1 for v in frags.values() if v >= 2)} chunks "
          f"(count >= 2) -> fragments.tsv", file=sys.stderr)

    # ---- generative probe: chain fragment EN-sides into real EN words ----
    lex_en = load_en()
    pron2words = defaultdict(list)
    for w, prons in lex_en.items():
        for p in prons:
            segs = matcher._segs(_canonical(p))
            if segs:
                pron2words[segs].append(w)

    have_usable = {e["en"] for e in usable}
    top = [(en_seq, fr_seq) for (en_seq, fr_seq), n in frags.most_common(400)]
    root = pd.build_trie(min_zipf=1.5)

    novel, seen = [], set()
    for a_en, _a_fr in top:
        for b_en, _b_fr in top:
            cand = a_en + b_en
            if not (3 <= len(cand) <= 7):
                continue
            for w in pron2words.get(cand, []):
                if w in have_usable or w in seen or len(w) < 3:
                    continue
                seen.add(w)
                ipa = lex_en[w][0]
                for c in pd.decode(ipa, root, top_n=2):
                    if (c["similarity"] >= 0.88 and c["coverage"] >= 0.85
                            and c["expensive_deletions"] == 0
                            and c["max_substitution"] <= 0.30):
                        novel.append((c["similarity"], w, c["fr"], ipa,
                                      "".join(a_en) + "+" + "".join(b_en)))
                        break

    novel.sort(reverse=True)
    with open("generative-matches.tsv", "w") as f:
        f.write("score\ten\tfr\ten_ipa\tchunk_recipe\n")
        for s, w, fr, ipa, recipe in novel:
            f.write(f"{s}\t{w}\t{fr}\t{ipa}\t{recipe}\n")
    print(f"generative probe: {len(novel)} novel matches from chunk chaining "
          f"-> generative-matches.tsv", file=sys.stderr)
    for s, w, fr, ipa, recipe in novel[:15]:
        print(f"  {s:.2f} {w:14s} ~ {fr:20s} [{ipa}]  via {recipe}")


if __name__ == "__main__":
    main()
