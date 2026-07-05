"""F47 TEASER: the tower has more floors -- same machinery, English<->SPANISH.

Scores MUSE EN-ES literal translations for sound with the same matcher
(espeak 'es' voice). If the dual density is comparable to French, the pipeline
is language-pair generic and the Babel tower generalizes.

Run: python babel_es.py [--limit 4000]
"""
from __future__ import annotations

import argparse

import matcher


def combo_es(en, es):
    try:
        qi = matcher.g2p(en, "en")
        ci = matcher.g2p(es, "es")
        return 0.5 * matcher._ngram_channel(qi, ci) + 0.5 * matcher._feat_channel(qi, ci)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4000)
    args = ap.parse_args()
    seen, pairs = set(), []
    for line in open("/tmp/muse-en-es.txt", encoding="utf-8"):
        p = line.split()
        if len(p) == 2 and p[0].isalpha() and p[1].isalpha() and (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            pairs.append((p[0], p[1]))
        if len(pairs) >= args.limit:
            break
    kept = {"DUAL-S": 0, "DUAL-A": 0, "DUAL-B": 0}
    rows = []
    for en, es in pairs:
        s = combo_es(en, es)
        t = "DUAL-S" if s >= 0.75 else ("DUAL-A" if s >= 0.60 else
                                        ("DUAL-B" if s >= 0.45 else ""))
        if t:
            kept[t] += 1
            rows.append((en, es, s, t))
    with open("dual-pairs-es-sample.tsv", "w", encoding="utf-8") as f:
        f.write("en\tes\tsound\ttier\n")
        for en, es, s, t in sorted(rows, key=lambda r: -r[2]):
            f.write(f"{en}\t{es}\t{s:.3f}\t{t}\n")
    print(f"EN-ES teaser on {len(pairs)} literal pairs: "
          f"S {kept['DUAL-S']}  A {kept['DUAL-A']}  B {kept['DUAL-B']}")
    for en, es, s, t in sorted(rows, key=lambda r: -r[2])[:10]:
        print(f"  {t} {en:14s} ~ {es:14s} {s:.2f}")
    print("(French, same slice, kept ~49%. Comparable density = the machinery "
          "is language-pair generic; the tower generalizes.)")


if __name__ == "__main__":
    main()
