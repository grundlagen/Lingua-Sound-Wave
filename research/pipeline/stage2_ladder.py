"""STAGE 2 — the dual ladder: monolingual meaning scaffolding per word.

Wires the homonym classes that already exist (en/fr homophone-classes TSVs)
into one ladder JSON per language for the gold vocabulary, plus the
translation senses the v7 dictionary already knows. No cross-language sound
content — this is the context object stages 3 and 5 consume.

  {"word": "sea", "lang": "en",
   "homonyms": ["c", "ce", "se", "see", "si"],      # same sound, own language
   "senses":   ["mer"],                              # translations (sense proxies)
   "zipf": 4.5}

Inputs (from research/homophone-bench, via --bench-dir):
  tier-ladder.tsv                    the unified gold (words to ladder)
  en-homophone-classes.tsv           EN homonym classes (ipa -> members)
  fr-homophone-classes-lexique.tsv   FR homonym classes (Lexique, 33k)
  dictionary-v7-integrated.json      translation senses

Run: python stage2_ladder.py --bench-dir <dir> --out-dir out/
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

TOP_TIERS = {"DUAL-S", "S", "STRICT-GOLD", "LOOP2", "LOOP1", "GOLD"}


def load_gold(bench: Path, tiers: set[str]) -> list[tuple[str, str, str]]:
    """(en, fr, tier) rows from tier-ladder.tsv restricted to trusted tiers."""
    rows = []
    with open(bench / "tier-ladder.tsv", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            if r["ladder"] in tiers:
                rows.append((r["en"].strip(), r["fr"].strip(), r["ladder"]))
    return rows


def load_classes(path: Path) -> dict[str, list[str]]:
    """word -> other members of its same-sound class."""
    by_word: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd, None)  # header
        for row in rd:
            if len(row) < 2:
                continue
            members = row[1].split()
            for m in members:
                by_word[m] = [x for x in members if x != m]
    return by_word


def load_senses(bench: Path) -> tuple[dict, dict]:
    """en word -> fr translations seen in v7; and the reverse."""
    en_senses, fr_senses = defaultdict(set), defaultdict(set)
    data = json.loads((bench / "dictionary-v7-integrated.json").read_text())
    for e in data:
        en, fr = e.get("en", ""), e.get("fr", "")
        if en and fr:
            en_senses[en].add(fr)
            fr_senses[fr].add(en)
    return en_senses, fr_senses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--tiers", default=",".join(sorted(TOP_TIERS)))
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    args = ap.parse_args()
    bench = args.bench_dir
    tiers = set(args.tiers.split(","))

    try:
        from wordfreq import zipf_frequency
    except ImportError:
        zipf_frequency = lambda w, lang: 0.0

    gold = load_gold(bench, tiers)
    en_words = sorted({en for en, _, _ in gold})
    fr_words = sorted({fr for _, fr, _ in gold})
    print(f"gold: {len(gold)} pairs -> {len(en_words)} EN / {len(fr_words)} FR words")

    en_hom = load_classes(bench / "en-homophone-classes.tsv")
    fr_hom = load_classes(bench / "fr-homophone-classes-lexique.tsv")
    # the smaller curated FR file fills words Lexique lacks
    for w, ms in load_classes(bench / "fr-homophone-classes.tsv").items():
        fr_hom.setdefault(w, ms)
    en_senses, fr_senses = load_senses(bench)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for lang, words, hom, senses in (
        ("en", en_words, en_hom, en_senses),
        ("fr", fr_words, fr_hom, fr_senses),
    ):
        ladders = {}
        for w in words:
            ladders[w] = {
                "word": w,
                "lang": lang,
                "homonyms": hom.get(w, []),
                "senses": sorted(senses.get(w, [])),
                "zipf": round(zipf_frequency(w, lang), 2),
            }
        out = args.out_dir / f"ladders-{lang}.json"
        out.write_text(json.dumps(ladders, ensure_ascii=False, indent=0))
        with_hom = sum(1 for l in ladders.values() if l["homonyms"])
        stats[lang] = (len(ladders), with_hom)
        print(f"{lang}: {len(ladders)} ladders, {with_hom} with homonyms -> {out}")
    return stats


if __name__ == "__main__":
    main()
