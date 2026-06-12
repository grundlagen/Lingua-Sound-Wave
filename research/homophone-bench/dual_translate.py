"""Dual-track translation: literal AND homophonic, with a meaning-graded
blend — the goal the user's original homophone-agent-audio skeleton named.

For an English sentence:
  literal      word-by-word MUSE gloss (honest offline stand-in for MT)
  homophonic   sentence-level decoder rendering (sounds like the English)
  blend        decoder candidates re-ranked by embedding similarity to the
               LITERAL translation — homophonic lines that also lean toward
               the meaning. This is "homophonic and literal at once",
               graded rather than hoped-for.
  no-cognate   same, but candidates may not contain any French word that is
               a translation of a sentence word or shares its spelling —
               kills the trivial cognate path, as requested.

Usage: python dual_translate.py "the sea is cold" ...
Output: dual-translate-demo.txt
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict

import numpy as np

import phonetic_decoder as pd
from lexicon_g2p import clean_ipa

TOP_N = 60


def literal_gloss(words, muse):
    return [sorted(muse[w])[0] if muse.get(w) else f"[{w}]" for w in words]


def main():
    sentences = sys.argv[1:] or [
        "the sea is cold",
        "two men under the moon",
        "she said tell me more",
    ]
    muse = defaultdict(set)
    with open("/tmp/muse-en-fr.txt", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                muse[p[0]].add(p[1])

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    root = pd.build_trie(min_zipf=2.2)

    out = []
    for sent in sentences:
        words = [w.lower().strip(".,!?") for w in sent.split()]
        lit = literal_gloss(words, muse)
        lit_line = " ".join(w for w in lit if not w.startswith("["))

        r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", sent],
                           capture_output=True, text=True, check=True)
        ipa = clean_ipa(r.stdout.strip())
        cands = pd.decode(ipa, root, top_n=TOP_N, max_words=10)
        cands = [c for c in cands if c["coverage"] >= 0.8
                 and c["expensive_deletions"] == 0]
        if not cands:
            out.append(f"\nEN:        {sent}\n  (no homophonic candidates)")
            continue

        texts = [lit_line] + [c["fr"] for c in cands]
        vecs = np.asarray(model.encode(texts, normalize_embeddings=True,
                                       show_progress_bar=False))
        for c, v in zip(cands, vecs[1:]):
            c["sem"] = float(np.dot(vecs[0], v))
            c["blend"] = c["similarity"] * (0.5 + c["sem"] / 2)

        best_sound = max(cands, key=lambda c: c["similarity"])
        best_blend = max(cands, key=lambda c: c["blend"])

        banned = set()
        for w in words:
            banned |= muse.get(w, set())
            banned.add(w)
        nc = [c for c in cands
              if not any(fw in banned for fw in c["fr"].split())]
        best_nc = max(nc, key=lambda c: c["blend"]) if nc else None

        block = [f"\nEN:         {sent}   [{ipa}]",
                 f"literal:    {' '.join(lit)}",
                 f"homophonic: {best_sound['fr']}"
                 f"   (snd {best_sound['similarity']:.2f}, sem {best_sound['sem']:.2f})",
                 f"blend:      {best_blend['fr']}"
                 f"   (snd {best_blend['similarity']:.2f}, sem {best_blend['sem']:.2f})"]
        if best_nc:
            block.append(f"no-cognate: {best_nc['fr']}"
                         f"   (snd {best_nc['similarity']:.2f}, sem {best_nc['sem']:.2f})")
        out.append("\n".join(block))

    text = "\n".join(out)
    print(text)
    with open("dual-translate-demo.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
