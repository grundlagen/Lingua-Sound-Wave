"""STAGE 3 (multi-word pass) — inflect the words INSIDE multi-word gold units.

For each multi-word gold pair, vary one word position at a time with that
word's morphological variants (EN via LemmInflect, FR via the same rules +
Lexique reality check as stage3_expand). One-at-a-time keeps the candidate
count linear and the provenance obvious; espeak scores whole phrases, so
French liaison/elision at the seams is judged natively by stage 4.

Output (same schema as stage3_expand, phrases as the key):
  expansion-mw-{en,fr}.tsv    phrase <TAB> variant_phrase <TAB> zipf(changed word)

Run: python stage3_multiword.py --bench-dir <hb> --out-dir out-mw/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stage3_expand import FR_RULES, expand_en, expand_fr, load_fr_lexicon
from stage4_filter import load_gold


def variants_at_each_position(phrase: str, word_variants) -> set[tuple[str, float]]:
    """Swap one word at a time; returns (new_phrase, zipf_of_changed_word)."""
    words = phrase.split()
    out: set[tuple[str, float]] = set()
    for i, w in enumerate(words):
        for v, z in word_variants(w):
            if v != w:
                out.add((" ".join(words[:i] + [v] + words[i + 1:]), z))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--tiers", default="DUAL-S,S,STRICT-GOLD,LOOP2,LOOP1,GOLD")
    ap.add_argument("--out-dir", type=Path, default=Path("out-mw"))
    ap.add_argument("--max-variants", type=int, default=8,
                    help="per word position, top-zipf N")
    args = ap.parse_args()

    from wordfreq import zipf_frequency

    fr_lexicon = load_fr_lexicon(args.bench_dir)
    gold = load_gold(args.bench_dir, set(args.tiers.split(",")))
    mw = [(en, fr, t) for en, fr, t in gold if " " in en or " " in fr]
    print(f"multi-word gold: {len(mw)} pairs")

    en_cache: dict[str, list] = {}
    fr_cache: dict[str, list] = {}

    def en_vars(w):
        if w not in en_cache:
            vs = expand_en(w)
            ranked = sorted(vs, key=lambda v: -zipf_frequency(v, "en"))
            en_cache[w] = [(v, zipf_frequency(v, "en"))
                           for v in ranked[: args.max_variants]]
        return en_cache[w]

    def fr_vars(w):
        if w not in fr_cache:
            vs = expand_fr(w, fr_lexicon, zipf_frequency)
            ranked = sorted(vs, key=lambda v: -zipf_frequency(v, "fr"))
            fr_cache[w] = [(v, zipf_frequency(v, "fr"))
                           for v in ranked[: args.max_variants]]
        return fr_cache[w]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for lang, idx, word_variants in (("en", 0, en_vars), ("fr", 1, fr_vars)):
        out = args.out_dir / f"expansion-{lang}.tsv"
        n = 0
        seen: set[tuple[str, str]] = set()
        with open(out, "w", encoding="utf-8") as f:
            f.write("word\tvariant\tzipf\n")
            for row in mw:
                phrase = row[idx]
                if (phrase, phrase) in seen:
                    continue
                seen.add((phrase, phrase))
                for vp, z in variants_at_each_position(phrase, word_variants):
                    f.write(f"{phrase}\t{vp}\t{z:.2f}\n")
                    n += 1
        print(f"{lang}: {n} phrase variants -> {out}")


if __name__ == "__main__":
    main()
