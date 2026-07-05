"""Show what the bigram LM does: generate ONE candidate pool, then rank it two
ways -- by mean wordfreq-zipf (the old fluency) and by the word-bigram LM --
and print them side by side so the reordering is visible.

Run from research/homophone-bench/ after building the LM:
  python bigram_lm.py /tmp/corpus
  python lm_compare.py
"""
from __future__ import annotations

import time

import phonetic_decoder as pd
import bigram_lm as B
import fragment_weave as fw


def zipf_joint(r):
    ew, frw = r["en"].split(), r["fr"].split()
    en_fl = sum(min(fw._zipf(w, "en"), 6.0) / 6.0 for w in ew) / len(ew)
    fr_fl = sum(min(fw._zipf(w, "fr"), 6.0) / 6.0 for w in frw) / len(frw)
    return r["sound"] * en_fl * fr_fl * (0.6 + 0.4 * r["novelty"]), en_fl, fr_fl


def main():
    en_lm, fr_lm = B.load("en"), B.load("fr")
    pd.BEAM = fw.DECODE_BEAM
    blocks = fw.load_blocks()
    known_en, known_fr = fw.known_sets()
    en_root = pd.build_trie(min_zipf=3.0, lang="en")
    fr_root = pd.build_trie(min_zipf=3.0, lang="fr")
    res = fw.grow(blocks, en_root, fr_root, known_en, known_fr,
                  max_len=fw.MAX_LEN, deadline=time.time() + 200)

    for r in res:
        zj, en_z, fr_z = zipf_joint(r)
        r["zipf_joint"] = zj
        en_l = en_lm.fluency(r["en"].split())
        fr_l = fr_lm.fluency(r["fr"].split())
        r["lm_joint"] = r["sound"] * en_l * fr_l * (0.6 + 0.4 * r["novelty"])
        r["en_l"], r["fr_l"] = en_l, fr_l

    by_zipf = sorted(res, key=lambda r: -r["zipf_joint"])
    by_lm = sorted(res, key=lambda r: -r["lm_joint"])
    zrank = {id(r): i for i, r in enumerate(by_zipf)}

    print(f"\n{len(res)} candidates. TOP 15 by each ranker.\n")
    print("== ranked by MEAN-ZIPF (old) ==")
    for r in by_zipf[:15]:
        print(f"  {r['en']:30s} | {r['fr']:28s}  zipf {r['zipf_joint']:.2f}")
    print("\n== ranked by BIGRAM-LM (new) ==   [Δ = rank change vs zipf]")
    for i, r in enumerate(by_lm[:15]):
        delta = zrank[id(r)] - i
        arrow = f"+{delta}" if delta > 0 else str(delta)
        print(f"  {r['en']:30s} | {r['fr']:28s}  lm {r['lm_joint']:.2f}  "
              f"(enL {r['en_l']:.2f} frL {r['fr_l']:.2f})  Δ{arrow}")

    print("\n== biggest LOSERS under the bigram LM (zipf liked, LM rejects) ==")
    drop = sorted(res, key=lambda r: r["zipf_joint"] - r["lm_joint"], reverse=True)
    for r in drop[:8]:
        print(f"  {r['en']:30s} | {r['fr']:28s}  zipf {r['zipf_joint']:.2f} "
              f"-> lm {r['lm_joint']:.2f}  (frL {r['fr_l']:.2f})")


if __name__ == "__main__":
    main()
