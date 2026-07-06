"""BABEL WINDOWS -- many-words-to-one and one-to-many, BOTH directions (B17/B18).

Permutations tried per sentence:
  EN n-gram (2-3 words) -> ONE French word        (at the ~ hâte)
  EN n-gram -> one FR UNIT (elision/liaison/place/interjection, fr-units.tsv)
  ONE EN word -> FR word PAIR (windowed over fr-word-ipa index)   (one->many)
  mirror: FR n-gram -> ONE English word (for the FR->EN composer)

Index = word->IPA tables from babel_mine + the units file. Matching = the
matcher's own similarity on joined IPA (no new judge).

Run: python babel_windows.py "we sat at the door of the inn"
     python babel_windows.py --fr "un petit d'un petit"     (mirror direction)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import matcher
from semantic_cosine import semantic_cosine


def combo_ipa(ia, ib):
    return 0.5 * matcher._ngram_channel(ia, ib) + 0.5 * matcher._feat_channel(ia, ib)


def load_index(path):
    idx = {}
    for i, line in enumerate(open(path, encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1]:
            idx[p[0]] = p[1]
    return idx


def load_units(path):
    rows = []
    try:
        for i, line in enumerate(open(path, encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                rows.append((p[0], p[1], p[2]))
    except FileNotFoundError:
        pass
    return rows


def by_len(idx):
    d = defaultdict(list)
    for w, p in idx.items():
        d[len(matcher._segs(p))].append((w, p))
    return d


def window_match(ipa, idx_bylen, top=5, tol=2):
    """ONE target for an IPA span: best matches within +-tol segments."""
    n = len(matcher._segs(ipa))
    cands = []
    for L in range(max(1, n - tol), n + tol + 1):
        cands.extend(idx_bylen.get(L, []))
    scored = []
    for w, p in cands:
        s = combo_ipa(ipa, p)
        if s >= 0.55:
            scored.append((s, w))
    scored.sort(reverse=True)
    return scored[:top]


def en_sentence(sent, fr_idx, units, top=4):
    ws = [w.lower().strip(".,;:!?'") for w in sent.split() if w.strip(".,;:!?'")]
    fr_bylen = by_len(fr_idx)
    unit_bylen = defaultdict(list)
    for u, p, k in units:
        unit_bylen[len(matcher._segs(p))].append((f"{u}〔{k}〕", p))
    print(f"EN: {sent}\n")
    # many EN words -> one FR word / unit
    for size in (2, 3):
        for i in range(len(ws) - size + 1):
            gram = " ".join(ws[i:i + size])
            ipa = matcher._canonical(matcher.g2p(gram, "en"))
            hits = window_match(ipa, fr_bylen, top=top)
            uhits = window_match(ipa, unit_bylen, top=2)
            allh = sorted(hits + uhits, reverse=True)[:top]
            if allh:
                print(f"  [{gram}] -> " + "  ".join(f"{w}({s:.2f})" for s, w in allh))
    # one EN word -> two FR words (one-to-many permutation)
    small = {w: p for w, p in fr_idx.items() if 1 <= len(matcher._segs(p)) <= 4}
    small_bylen = by_len(small)
    for w in ws:
        ipa = matcher._canonical(matcher.g2p(w, "en"))
        n = len(matcher._segs(ipa))
        if n < 4:
            continue
        best = []
        for cut in range(2, n - 1):
            head, tail = "".join(matcher._segs(ipa)[:cut]), "".join(matcher._segs(ipa)[cut:])
            h1 = window_match(head, small_bylen, top=1, tol=1)
            h2 = window_match(tail, small_bylen, top=1, tol=1)
            if h1 and h2:
                s = (h1[0][0] + h2[0][0]) / 2
                best.append((s, f"{h1[0][1]} {h2[0][1]}"))
        best.sort(reverse=True)
        if best and best[0][0] >= 0.62:
            s, pair = best[0]
            m = max(0.0, semantic_cosine(w, pair))
            print(f"  {w} -> «{pair}» ({s:.2f}, mng {m:.2f})   [one->many]")


def fr_sentence(sent, en_idx, top=4):
    ws = [w.lower().strip(".,;:!?'") for w in sent.split() if w.strip(".,;:!?'")]
    en_bylen = by_len(en_idx)
    print(f"FR: {sent}   (mirror direction)\n")
    for size in (2, 3):
        for i in range(len(ws) - size + 1):
            gram = " ".join(ws[i:i + size])
            ipa = matcher._canonical(matcher.g2p(gram, "fr"))
            hits = window_match(ipa, en_bylen, top=top)
            if hits:
                print(f"  [{gram}] -> " + "  ".join(f"{w}({s:.2f})" for s, w in hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--fr", default="")
    args = ap.parse_args()
    if args.fr:
        en_idx = load_index("en-word-ipa.tsv")
        fr_sentence(args.fr, en_idx)
        return
    fr_idx = load_index("fr-word-ipa.tsv")
    units = load_units("fr-units.tsv")
    for sent in (args.text or ["we sat at the door of the inn"]):
        en_sentence(sent, fr_idx, units)


if __name__ == "__main__":
    main()
