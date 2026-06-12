"""Sound + meaning: grade every sound-pair by cross-lingual semantic
similarity, so "sounds the same AND means something close" becomes a
queryable column instead of a binary cognate flag.

Why this and not an LLM doing it freehand: the meaning side needs a graded,
deterministic score over 5k+ pairs. Multilingual sentence embeddings
(paraphrase-multilingual-MiniLM) place EN and FR words in one vector space,
so cosine(en, fr) directly measures meaning closeness across languages —
including multiword French sides, which embed as phrases. The LLM's job
comes after (see LLM_RECIPE.md): judging/repairing the top slice and
proposing synonym substitutions, never generating phonetics or scores.

Bands:
  identical   sem >= 0.80   (cognate-grade: prove~prouve, breeze~brise)
  close       0.55 - 0.80   (the prize: same neighborhood, not same word)
  related     0.35 - 0.55   (domain echo: worth keeping for theming)
  unrelated   < 0.35        (pure homophones — still the pun dictionary)

Output: sound-meaning-v1.tsv ranked by combined = phonetic * (0.5 + sem/2).
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer


def band(sem: float) -> str:
    if sem >= 0.80:
        return "identical"
    if sem >= 0.55:
        return "close"
    if sem >= 0.35:
        return "related"
    return "unrelated"


def main():
    entries = json.load(open("dictionary-v5.json"))
    pairs = [e for e in entries if e.get("usable_for_composition")]
    print(f"scoring {len(pairs)} usable pairs", file=sys.stderr)

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    en_texts = sorted({e["en"] for e in pairs})
    fr_texts = sorted({e["fr"] for e in pairs})
    en_vec = dict(zip(en_texts, model.encode(en_texts, batch_size=256,
                                             normalize_embeddings=True,
                                             show_progress_bar=False)))
    fr_vec = dict(zip(fr_texts, model.encode(fr_texts, batch_size=256,
                                             normalize_embeddings=True,
                                             show_progress_bar=False)))

    rows = []
    for e in pairs:
        sem = float(np.dot(en_vec[e["en"]], fr_vec[e["fr"]]))
        combined = e["score"] * (0.5 + sem / 2)
        rows.append((combined, sem, e))

    rows.sort(key=lambda r: -r[0])
    from collections import Counter
    bands = Counter(band(s) for _c, s, _e in rows)
    print(f"bands: {dict(bands)}", file=sys.stderr)

    with open("sound-meaning-v1.tsv", "w") as f:
        f.write("combined\tsound\tsemantic\tband\ten\tfr\tflags\ten_ipa\tfr_ipa\n")
        for c, s, e in rows:
            flags = ",".join(k for k in
                             ["multiword", "cognate", "generative", "funcword"]
                             if e.get(k))
            f.write(f"{c:.3f}\t{e['score']}\t{s:.3f}\t{band(s)}\t{e['en']}"
                    f"\t{e['fr']}\t{flags}\t{e.get('en_ipa','')}\t{e.get('fr_ipa','')}\n")
    print("wrote sound-meaning-v1.tsv")

    print("\n=== the prize band: CLOSE (sound-same, meaning-near, NOT cognate) ===")
    shown = 0
    for c, s, e in rows:
        if band(s) == "close" and not e.get("cognate"):
            print(f"  snd {e['score']:.2f} sem {s:.2f}  {e['en']:14s} ~ {e['fr']}")
            shown += 1
            if shown >= 25:
                break


if __name__ == "__main__":
    main()
