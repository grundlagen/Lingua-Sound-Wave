"""Improved dataset REMINING with the new linguistics.

Re-scores the composition dataset (the phrase bank) with what we built since v7:
  - stress-weighted PROSODY (sound) instead of plain combo, so perceptually
    better homophones (stressed-aligned, unstressed-forgiven) rise;
  - SEMANTIC-COSINE (meaning) so we can flag the GOLD sound∧meaning units
    (sounds alike AND still means the source) -- the dual-translation atoms.

Preserves old data: writes phrase-bank-remined.tsv (never overwrites the source).
Reports promotions (prosody beats the old combo) and how many gold pairs surfaced.

Run: python remine.py [phrase-bank-balanced.tsv]
"""
from __future__ import annotations

import sys

import numpy as np

import prosody

SRC = sys.argv[1] if len(sys.argv) > 1 else "phrase-bank-balanced.tsv"
GOLD_SOUND, GOLD_MEAN = 0.70, 0.45


def main():
    rows = []
    for i, line in enumerate(open(SRC, encoding="utf-8")):
        if i == 0:
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            rows.append((parts[0], parts[1], float(parts[2])))   # en, fr, old combo
    print(f"remining {len(rows)} pairs from {SRC} ...", flush=True)

    # meaning: batch-encode all en + fr once (the multilingual encoder)
    from sentence_transformers import SentenceTransformer
    M = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    ens = [e for e, f, c in rows]
    frs = [f for e, f, c in rows]
    ve = M.encode(ens, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    vf = M.encode(frs, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    meaning = (np.asarray(ve) * np.asarray(vf)).sum(1)

    out = open("phrase-bank-remined.tsv", "w", encoding="utf-8")
    out.write("en\tfr\told_combo\tprosody\tmeaning\ttier\n")
    promoted = gold = 0
    examples = []
    for (en, fr, combo), mean in zip(rows, meaning):
        pros = prosody.prosodic_score(en, fr)
        is_gold = pros >= GOLD_SOUND and mean >= GOLD_MEAN
        tier = "GOLD" if is_gold else ("A" if pros >= 0.6 else "B")
        out.write(f"{en}\t{fr}\t{combo:.3f}\t{pros:.3f}\t{mean:.3f}\t{tier}\n")
        if pros > combo + 0.05:
            promoted += 1
        if is_gold:
            gold += 1
            if len(examples) < 12:
                examples.append((en, fr, pros, float(mean)))
    out.close()
    print(f"\nremined -> phrase-bank-remined.tsv")
    print(f"  prosody PROMOTED {promoted}/{len(rows)} pairs over the old combo "
          f"(better perceptual homophones)")
    print(f"  GOLD sound∧meaning pairs (prosody>={GOLD_SOUND}, meaning>={GOLD_MEAN}): "
          f"{gold}")
    print("  gold examples (sound AND meaning):")
    for en, fr, p, m in sorted(examples, key=lambda x: -(x[2] * x[3])):
        print(f"     {en:22s} -> {fr:22s} sound {p:.2f} meaning {m:.2f}")


if __name__ == "__main__":
    main()
