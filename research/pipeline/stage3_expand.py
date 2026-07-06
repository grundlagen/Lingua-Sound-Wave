"""STAGE 3 (deterministic half) — expand the gold vocabulary morphologically.

No GPU, no model: conjugation is a lookup, not a skill. For every gold word:

  EN  every inflection from LemmInflect (all lemmas, all tags), e.g.
      river -> rivers; walk -> walks/walked/walking; deep -> deeper/deepest
  FR  rule-generated candidate forms (verb endings, plural, agreement),
      validated against the Lexique wordlist — only REAL French words
      survive (membership in fr-homophone-classes-lexique.tsv members or a
      wordfreq zipf floor)

Every variant carries its Zipf frequency so downstream ranking prefers
walks/walked over walkest. Output, per language:

  expansion-{en,fr}.tsv    word <TAB> variant <TAB> zipf

Run: python stage3_expand.py --bench-dir <dir> --ladder-dir out/ --out-dir out/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# French candidate suffix rules: (strip, add) applied to the gold form.
# Over-generates on purpose — the Lexique reality check filters.
FR_RULES: list[tuple[str, list[str]]] = [
    ("", ["s", "x", "e", "es", "ent", "ez", "ons", "ait", "aient", "ant",
          "er", "é", "ée", "és", "ées", "a", "ai", "as"]),
    ("e", ["es", "ent", "er", "ez", "ait", "ant", "é", "a"]),
    ("er", ["e", "es", "ent", "ez", "ons", "ait", "aient", "ant", "é", "ée",
            "és", "ées", "a", "ai"]),
    ("s", [""]),
    ("t", ["ts", "te", "tes"]),
]


def load_fr_lexicon(bench: Path) -> set[str]:
    """Every word Lexique's homophone classes know = real French words."""
    words: set[str] = set()
    for name in ("fr-homophone-classes-lexique.tsv", "fr-homophone-classes.tsv"):
        with open(bench / name, encoding="utf-8") as f:
            rd = csv.reader(f, delimiter="\t")
            next(rd, None)
            for row in rd:
                if len(row) >= 2:
                    words.update(row[1].split())
    return words


def expand_en(word: str) -> set[str]:
    from lemminflect import getAllInflections, getAllLemmas
    variants: set[str] = set()
    lemmas = {l for pos in getAllLemmas(word).values() for l in pos} or {word}
    for lemma in lemmas:
        variants.add(lemma)
        for forms in getAllInflections(lemma).values():
            variants.update(forms)
    variants.discard(word)
    return {v.lower() for v in variants if v.isalpha()}


def expand_fr(word: str, lexicon: set[str], zipf) -> set[str]:
    cands: set[str] = set()
    for strip, adds in FR_RULES:
        if strip and not word.endswith(strip):
            continue
        stem = word[: len(word) - len(strip)] if strip else word
        for add in adds:
            cands.add(stem + add)
    cands.discard(word)
    # reality check: known word list, or common enough that wordfreq knows it
    return {c for c in cands
            if c in lexicon or zipf(c, "fr") >= 2.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--ladder-dir", type=Path, default=Path("out"))
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    ap.add_argument("--max-variants", type=int, default=12,
                    help="keep the top-zipf N variants per word")
    args = ap.parse_args()

    from wordfreq import zipf_frequency

    fr_lexicon = load_fr_lexicon(args.bench_dir)
    print(f"FR reality-check lexicon: {len(fr_lexicon)} words")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for lang, expander in (("en", None), ("fr", None)):
        ladders = json.loads(
            (args.ladder_dir / f"ladders-{lang}.json").read_text())
        n_rows = 0
        out = args.out_dir / f"expansion-{lang}.tsv"
        with open(out, "w", encoding="utf-8") as f:
            f.write("word\tvariant\tzipf\n")
            for w in ladders:
                if not w or " " in w:      # single words only in this pass
                    continue
                if lang == "en":
                    vs = expand_en(w)
                else:
                    vs = expand_fr(w, fr_lexicon, zipf_frequency)
                ranked = sorted(vs, key=lambda v: -zipf_frequency(v, lang))
                for v in ranked[: args.max_variants]:
                    f.write(f"{w}\t{v}\t{zipf_frequency(v, lang):.2f}\n")
                    n_rows += 1
        print(f"{lang}: {n_rows} variants -> {out}")


if __name__ == "__main__":
    main()
