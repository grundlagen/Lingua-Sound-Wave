"""STAGE 3 (paraphrase pass) — meaning-bridges from the corpus itself.

The move the user asked for: match ANY English word by meaning (paraphrase)
to a word that already HAS a proven French homophone — then the new word
inherits a sound cell via meaning. Same in the other direction for French.

Meaning source is the corpus's own DUAL tiers in tier-ladder.tsv (pairs that
are literal MUSE translation AND homophone): pivot synonymy —

    en1 = fr = en2   →   en1 ~ en2 are meaning-mates (shared FR translations)
    fr1 = en = fr2   →   fr1 ~ fr2 are meaning-mates (shared EN translations)

Outputs:
  paraphrase-{en,fr}.tsv          word <TAB> mate <TAB> shared_translations
  paraphrase-bridges-{en,fr}.tsv  word without a gold homophone <TAB> mate
                                  that has one <TAB> the mate's FR (or EN)
                                  homophone <TAB> shared — i.e. "say it with
                                  the mate, gain the sound"
  paraphrase_sft_{en,fr}.jsonl    ChatML SFT rows to train the GPU paraphraser
                                  (Qwen kit, research/qwen-finetune/) to
                                  generalize this beyond the corpus

Run: python stage3_paraphrase.py --bench-dir <hb> --out-dir out-para/
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DUAL_TIERS = {"DUAL-S", "DUAL-A", "DUAL-B"}
GOLD_TIERS = {"DUAL-S", "S", "STRICT-GOLD", "LOOP2", "LOOP1", "GOLD"}


def load_rows(bench: Path):
    with open(bench / "tier-ladder.tsv", encoding="utf-8") as f:
        yield from csv.DictReader(f, delimiter="\t")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("out-para"))
    ap.add_argument("--min-shared", type=int, default=2,
                    help="pivot translations required to call two words mates")
    ap.add_argument("--max-mates", type=int, default=10)
    args = ap.parse_args()

    # translation edges from DUAL rows; gold sound cells from trusted tiers
    en2fr, fr2en = defaultdict(set), defaultdict(set)
    gold_en, gold_fr = defaultdict(set), defaultdict(set)  # word -> sound partners
    for r in load_rows(args.bench_dir):
        en, fr, tier = r["en"].strip(), r["fr"].strip(), r["ladder"]
        if not en or not fr:
            continue
        if tier in DUAL_TIERS:
            en2fr[en].add(fr)
            fr2en[fr].add(en)
        if tier in GOLD_TIERS:
            gold_en[en].add(fr)
            gold_fr[fr].add(en)
    print(f"translation edges: {sum(len(v) for v in en2fr.values())} "
          f"| gold sound cells: {len(gold_en)} EN / {len(gold_fr)} FR")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for lang, fwd, rev, gold in (("en", en2fr, fr2en, gold_en),
                                 ("fr", fr2en, en2fr, gold_fr)):
        # pivot: mates share >= min_shared translations
        mates: dict[str, list[tuple[str, int]]] = {}
        for w, trans in fwd.items():
            c: Counter = Counter()
            for t in trans:
                for w2 in rev[t]:
                    if w2 != w:
                        c[w2] += 1
            top = [(m, n) for m, n in c.most_common(args.max_mates)
                   if n >= args.min_shared]
            if top:
                mates[w] = top

        n_pairs = sum(len(v) for v in mates.values())
        with open(args.out_dir / f"paraphrase-{lang}.tsv", "w", encoding="utf-8") as f:
            f.write("word\tmate\tshared\n")
            for w, ms in sorted(mates.items()):
                for m, n in ms:
                    f.write(f"{w}\t{m}\t{n}\n")
        print(f"{lang}: {len(mates)} words with mates, {n_pairs} mate pairs")

        # bridges: word has NO gold sound cell, but a mate does
        n_bridges = 0
        with open(args.out_dir / f"paraphrase-bridges-{lang}.tsv", "w",
                  encoding="utf-8") as f:
            f.write("word\tmate\tmate_homophone\tshared\n")
            for w, ms in sorted(mates.items()):
                if w in gold:
                    continue
                for m, n in ms:
                    for hp in sorted(gold.get(m, [])):
                        f.write(f"{w}\t{m}\t{hp}\t{n}\n")
                        n_bridges += 1
        print(f"{lang}: {n_bridges} meaning-bridges (word without a sound "
              f"cell -> mate that has one)")

        # SFT data for the GPU paraphraser
        sys_prompt = (
            f"Give {'English' if lang=='en' else 'French'} words or short "
            "phrases that mean the same as the given word. One per line, "
            "closest meaning first.")
        with open(args.out_dir / f"paraphrase_sft_{lang}.jsonl", "w",
                  encoding="utf-8") as f:
            for w, ms in sorted(mates.items()):
                if len(ms) < 2:      # single-mate rows teach too little
                    continue
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": w},
                    {"role": "assistant",
                     "content": "\n".join(m for m, _ in ms)},
                ]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
