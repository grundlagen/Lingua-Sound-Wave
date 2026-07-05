"""Build ChatML SFT JSONL for Agent A (EN→FR carver) and Agent B (FR→EN hearer).

Sources (all already produced by the pipeline):
  - dictionary-v7-integrated.json   word/phrase pairs, combo-filtered
  - phrase-bank-balanced.tsv        verified phrase carves
  - corpus-carves.tsv               PD nursery-rhyme carves

Agent B additionally gets the "honest hearer" mix (~30%): ordinary French
lines whose target is a faithful phonetic rendering, NOT an intended English
original — so B reports what it hears rather than solving the puzzle.
Provide those via --honest-tsv (fr<TAB>heard_en); build them locally by
running French through FR G2P and the reverse beam decoder (EN trie).

Usage:
  python build_sft_data.py --bench-dir ../homophone-bench \
      --min-combo 0.45 --out-dir data/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from common import (
    SYSTEM_A,
    SYSTEM_B,
    chatml_sft,
    load_g2p,
    load_matcher,
    user_prompt_a,
    user_prompt_b,
    write_jsonl,
)


def read_dictionary(path: Path):
    """Yield (en, fr, combo|None) from dictionary-v7-integrated.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.values() if isinstance(data, dict) else data
    for e in entries:
        if isinstance(e, dict):
            en = e.get("en") or e.get("english") or e.get("source")
            fr = e.get("fr") or e.get("french") or e.get("target")
            combo = e.get("combo") or e.get("score")
            if en and fr:
                yield str(en), str(fr), combo
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            yield str(e[0]), str(e[1]), (e[2] if len(e) > 2 else None)


def read_tsv_pairs(path: Path, en_col=0, fr_col=1, combo_col=None):
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) <= max(en_col, fr_col):
                continue
            combo = None
            if combo_col is not None and len(row) > combo_col:
                try:
                    combo = float(row[combo_col])
                except ValueError:
                    combo = None
            yield row[en_col].strip(), row[fr_col].strip(), combo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, default=None)
    ap.add_argument("--dictionary", type=Path, default=None,
                    help="dictionary-v7-integrated.json (default: bench dir)")
    ap.add_argument("--phrase-bank", type=Path, default=None)
    ap.add_argument("--corpus-carves", type=Path, default=None)
    ap.add_argument("--honest-tsv", type=Path, default=None,
                    help="fr<TAB>heard_en lines for the honest-hearer mix")
    ap.add_argument("--min-combo", type=float, default=0.45)
    ap.add_argument("--honest-frac", type=float, default=0.30)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    bench = args.bench_dir
    g2p = load_g2p(bench)
    combo_score = load_matcher(bench)
    rng = random.Random(args.seed)

    from common import BENCH_DIR
    root = bench or BENCH_DIR
    dictionary = args.dictionary or root / "dictionary-v7-integrated.json"
    phrase_bank = args.phrase_bank or root / "phrase-bank-balanced.tsv"
    corpus = args.corpus_carves or root / "corpus-carves.tsv"

    pairs = []
    if dictionary.exists():
        pairs += list(read_dictionary(dictionary))
    if phrase_bank.exists():
        pairs += list(read_tsv_pairs(phrase_bank))
    if corpus.exists():
        pairs += list(read_tsv_pairs(corpus))
    if not pairs:
        raise SystemExit("No source data found — check paths.")

    kept, dropped = [], 0
    seen = set()
    for en, fr, combo in pairs:
        key = (en.lower(), fr.lower())
        if key in seen:
            continue
        seen.add(key)
        if combo is None:
            try:
                combo = combo_score(en, fr)
            except Exception:
                combo = None
        if combo is not None and combo < args.min_combo:
            dropped += 1
            continue
        kept.append((en, fr))
    print(f"pairs: {len(kept)} kept, {dropped} below combo {args.min_combo}")

    a_rows, b_rows = [], []
    for en, fr in kept:
        try:
            ipa = g2p(en)
        except Exception:
            continue
        a_rows.append(chatml_sft(SYSTEM_A, user_prompt_a(en, ipa), fr))
        b_rows.append(chatml_sft(SYSTEM_B, user_prompt_b(fr), en))

    # Honest-hearer mix for B: faithful phonetic renderings of ordinary French.
    if args.honest_tsv and args.honest_tsv.exists():
        honest = [
            chatml_sft(SYSTEM_B, user_prompt_b(fr), heard)
            for fr, heard, _ in read_tsv_pairs(args.honest_tsv)
        ]
        target = int(len(b_rows) * args.honest_frac / (1 - args.honest_frac))
        rng.shuffle(honest)
        b_rows += honest[:target]
        print(f"honest-hearer rows added to B: {min(target, len(honest))}")
    else:
        print("WARNING: no --honest-tsv; Agent B will learn to be helpful, "
              "not honest — build the honest mix before training B for real.")

    rng.shuffle(a_rows)
    rng.shuffle(b_rows)
    out = args.out_dir
    for name, rows in (("agent_a", a_rows), ("agent_b", b_rows)):
        n_val = max(1, int(len(rows) * args.val_frac))
        n_train = write_jsonl(out / f"{name}_sft_train.jsonl", rows[n_val:])
        n = write_jsonl(out / f"{name}_sft_val.jsonl", rows[:n_val])
        print(f"{name}: {n_train} train / {n} val -> {out}/")


if __name__ == "__main__":
    main()
