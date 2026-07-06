"""Mine the ZIPF GLUE: sound-true French stand-ins for the top English
function/conjugation words -- the words every sentence needs.

The 0/50 sentence-scope result came from literal glue ("the"->"le" sounds
nothing alike). But the beauty-of-language fix exists: th-stopping makes
the≈de, h-dropping makes he≈y, v~w makes was≈vase, nasal makes in≈un. So mine
it exhaustively: top-N English words by zipf x frequent French words, keep the
best sound matches. Output = the glue table dual composition needs.

Run: python zipf_glue.py [--n_en 120] [--n_fr 3000]
"""
from __future__ import annotations

import argparse
from wordfreq import top_n_list, zipf_frequency

import matcher


def combo(en, fr):
    try:
        qi, ci = matcher.g2p(en, "en"), matcher.g2p(fr, "fr")
        return 0.5 * matcher._ngram_channel(qi, ci) + 0.5 * matcher._feat_channel(qi, ci)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_en", type=int, default=120)
    ap.add_argument("--n_fr", type=int, default=3000)
    ap.add_argument("--keep", type=int, default=5)
    args = ap.parse_args()

    en_words = top_n_list("en", args.n_en)
    fr_words = top_n_list("fr", args.n_fr)
    print(f"mining glue: top {len(en_words)} EN x top {len(fr_words)} FR")

    rows = []
    for i, en in enumerate(en_words):
        scored = []
        for fr in fr_words:
            s = combo(en, fr)
            if s >= 0.50:
                scored.append((s, fr))
        scored.sort(reverse=True)
        for s, fr in scored[:args.keep]:
            rows.append((en, fr, s, zipf_frequency(fr, "fr")))
        if scored:
            top = "  ".join(f"{fr}({s:.2f})" for s, fr in scored[:3])
            print(f"  {en:10s} -> {top}")
        else:
            print(f"  {en:10s} -> (none >=0.50)")

    with open("zipf-glue.tsv", "w", encoding="utf-8") as f:
        f.write("en\tfr\tsound\tfr_zipf\n")
        for en, fr, s, z in rows:
            f.write(f"{en}\t{fr}\t{s:.3f}\t{z:.2f}\n")
    print(f"\nwrote zipf-glue.tsv ({len(rows)} glue mappings)")


if __name__ == "__main__":
    main()
