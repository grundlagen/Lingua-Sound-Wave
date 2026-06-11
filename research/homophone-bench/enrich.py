"""Enrich the dictionary into the composition-ready v5 representation.

The dual-language writing stage needs entries to be *chainable*: machine-
readable alignments (not display strings), junction features (what an entry
starts/ends with, for liaison/hiatus rules), syllable counts, and a pivot
string in equivalence-class space for indexing. This script computes all of
that for every entry, all tiers, both directions.

Schema added per entry:
  align       [[en_seg, fr_seg, cost], ...]   ('·' marks a gap)
  pivot       coarse class string of the matched skeleton, e.g. "CV.CVC"
  en_syll / fr_syll        vowel-nucleus counts
  en_onset / en_coda / fr_onset / fr_coda     first/last segment + V|C class
  direction   "en_fr" (word EN keyed) or "fr_en" (decoder reverse entries)
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache

import numpy as np

import matcher
from matcher import _canonical, _variants

matcher._vecs = lru_cache(maxsize=None)(matcher._vecs.__wrapped__)
matcher._segs = lru_cache(maxsize=None)(matcher._segs.__wrapped__)

VOWEL_FEATURE = 0  # panphon 'syl' is feature index 0


def _is_vowel(seg: str) -> bool:
    v = matcher._vecs(seg)
    return len(v) > 0 and v[0][VOWEL_FEATURE] == 1


def _cls(seg: str) -> str:
    return "V" if _is_vowel(seg) else "C"


def best_alignment(ipa_a: str, ipa_b: str):
    """NW traceback under the CURRENT cost model (equivalence floor + cheap
    gaps), across rule variants. Returns list of (seg_a, seg_b, cost)."""
    best_s, best_pair = -1.0, (_canonical(ipa_a), _canonical(ipa_b))
    for a in _variants(ipa_a):
        for b in _variants(ipa_b):
            s = matcher.nw_sim_ipa(a, b)
            if s > best_s:
                best_s, best_pair = s, (a, b)
    a, b = best_pair
    sa, sb = matcher._segs(a), matcher._segs(b)
    va, vb = matcher._vecs(a), matcher._vecs(b)
    n, m = len(va), len(vb)
    if n == 0 or m == 0:
        return []
    sub = matcher._sub_matrix(sa, va, sb, vb)
    ga = [matcher._gap_cost(s) for s in sa]
    gb = [matcher._gap_cost(s) for s in sb]
    cost = np.zeros((n + 1, m + 1))
    for j in range(1, m + 1):
        cost[0, j] = cost[0, j - 1] + gb[j - 1]
    for i in range(1, n + 1):
        cost[i, 0] = cost[i - 1, 0] + ga[i - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i - 1, j - 1] + sub[i - 1, j - 1],
                             cost[i - 1, j] + ga[i - 1],
                             cost[i, j - 1] + gb[j - 1])
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(cost[i, j] - (cost[i - 1, j - 1] + sub[i - 1, j - 1])) < 1e-9:
            out.append((sa[i - 1], sb[j - 1], round(float(sub[i - 1, j - 1]), 3)))
            i, j = i - 1, j - 1
        elif i > 0 and abs(cost[i, j] - (cost[i - 1, j] + ga[i - 1])) < 1e-9:
            out.append((sa[i - 1], "·", round(ga[i - 1], 3)))
            i -= 1
        else:
            out.append(("·", sb[j - 1], round(gb[j - 1], 3)))
            j -= 1
    out.reverse()
    return out


def enrich(entry: dict) -> dict:
    en_ipa = entry.get("en_ipa", "")
    fr_ipa = (entry.get("fr_ipa") or "").replace(" ", "")
    e = dict(entry)
    e.setdefault("direction", "fr_en" if entry.get("reverse") else "en_fr")
    if not en_ipa or not fr_ipa:
        return e
    sa = matcher._segs(_canonical(en_ipa))
    sb = matcher._segs(_canonical(fr_ipa))
    if not sa or not sb:
        return e
    align = best_alignment(en_ipa, fr_ipa)
    e["align"] = [list(t) for t in align]
    e["pivot"] = "".join(
        _cls(x if x != "·" else y) if c < 0.30 and x != "·" and y != "·" else "·"
        for x, y, c in align)
    e["en_syll"] = sum(1 for s in sa if _is_vowel(s))
    e["fr_syll"] = sum(1 for s in sb if _is_vowel(s))
    e["en_onset"], e["en_coda"] = f"{sa[0]}|{_cls(sa[0])}", f"{sa[-1]}|{_cls(sa[-1])}"
    e["fr_onset"], e["fr_coda"] = f"{sb[0]}|{_cls(sb[0])}", f"{sb[-1]}|{_cls(sb[-1])}"
    # display string for humans, kept for backward compat
    e["alignment"] = " ".join(f"{x}:{y}" for x, y, _ in align)
    return e


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "dictionary-v4.json"
    entries = json.load(open(src))
    try:
        entries += json.load(open("reverse-additions.json"))
    except FileNotFoundError:
        pass
    out = []
    for i, x in enumerate(entries):
        out.append(enrich(x))
        if (i + 1) % 2000 == 0:
            print(f"  enriched {i + 1}/{len(entries)}", file=sys.stderr)
    with open("dictionary-v5.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    with open("dictionary-v5.tsv", "w") as f:
        f.write("tier\tscore\tdirection\ten\tfr\tflags\ten_ipa\tfr_ipa\tpivot"
                "\ten_syll\tfr_syll\talignment\n")
        for x in out:
            flags = ",".join(k for k in
                             ["multiword", "cognate", "loanword", "pairbank", "decoder"]
                             if x.get(k))
            f.write(f"{x['tier']}\t{x['score']}\t{x.get('direction','en_fr')}\t{x['en']}"
                    f"\t{x['fr']}\t{flags}\t{x.get('en_ipa','')}\t{x.get('fr_ipa','')}"
                    f"\t{x.get('pivot','')}\t{x.get('en_syll','')}\t{x.get('fr_syll','')}"
                    f"\t{x.get('alignment','')}\n")
    n_al = sum(1 for x in out if x.get("align"))
    print(f"v5: {len(out)} entries, {n_al} with machine alignments "
          f"-> dictionary-v5.json/tsv")


if __name__ == "__main__":
    main()
