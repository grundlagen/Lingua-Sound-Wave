"""Synthesize multi-turn REPAIR examples for Agent A.

Takes verified carves, corrupts one span with a low-combo alternative, and
trains the model to fix exactly that span on a REVISE turn — the distribution
the A↔B↔C loop actually produces. Corruptions come from the dictionary's
other FR entries for nearby-sounding English words (cheap hard negatives).

Usage:
  python build_repair_data.py --bench-dir ../homophone-bench --out data/agent_a_repair_train.jsonl
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from build_sft_data import read_dictionary, read_tsv_pairs
from common import SYSTEM_A, load_g2p, load_matcher, user_prompt_a, write_jsonl
from common import BENCH_DIR


def revise_prompt(keep: list[str], en_span: str, heard: str, combo: float,
                  candidates: list[str]) -> str:
    keep_s = ", ".join(f'"{w}"' for w in keep) if keep else "(nothing)"
    cands = " | ".join(candidates) if candidates else "(search the carve pool)"
    return (f'REVISE. Keep: {keep_s}. Fix span "{en_span}" '
            f'(heard as "{heard}", combo {combo:.2f}). Candidates: {cands}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/agent_a_repair_train.jsonl"))
    ap.add_argument("--max-examples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    root = args.bench_dir or BENCH_DIR
    g2p = load_g2p(args.bench_dir)
    combo_score = load_matcher(args.bench_dir)
    rng = random.Random(args.seed)

    # Phrase-level carves are the repair substrate (need >1 word).
    sources = []
    pb = root / "phrase-bank-balanced.tsv"
    cc = root / "corpus-carves.tsv"
    for p in (pb, cc):
        if p.exists():
            sources += [(en, fr) for en, fr, _ in read_tsv_pairs(p)
                        if len(en.split()) > 1 and len(fr.split()) > 1]
    if not sources:
        raise SystemExit("No phrase sources found (phrase bank / corpus carves).")

    # FR corruption pool from the dictionary.
    fr_pool = []
    d = root / "dictionary-v7-integrated.json"
    if d.exists():
        fr_pool = [fr for _, fr, _ in read_dictionary(d)]
    if not fr_pool:
        raise SystemExit("Dictionary not found for the corruption pool.")

    rows = []
    rng.shuffle(sources)
    for en, fr in sources:
        if len(rows) >= args.max_examples:
            break
        en_words, fr_words = en.split(), fr.split()
        # Corrupt one FR position with a random low-combo FR word.
        i = rng.randrange(len(fr_words))
        en_target = en_words[min(i, len(en_words) - 1)]
        bad = rng.choice(fr_pool)
        try:
            bad_combo = combo_score(en_target, bad)
            if bad_combo > 0.35:      # not corrupt enough to be a repair case
                continue
            ipa = g2p(en)
        except Exception:
            continue
        corrupted = fr_words[:i] + [bad] + fr_words[i + 1:]
        keep = [w for j, w in enumerate(fr_words) if j != i]
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_A},
                {"role": "user", "content": user_prompt_a(en, ipa)},
                {"role": "assistant", "content": " ".join(corrupted)},
                {"role": "user", "content": revise_prompt(
                    keep, en_target, bad, bad_combo, [fr_words[i]])},
                {"role": "assistant", "content": fr},
            ]
        })

    n = write_jsonl(args.out, rows)
    print(f"repair examples: {n} -> {args.out}")


if __name__ == "__main__":
    main()
