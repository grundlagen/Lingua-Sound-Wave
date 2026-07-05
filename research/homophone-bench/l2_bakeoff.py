"""L2 bake-off: embeds or no? trigram or bigram? -- decided by AUC, not taste.

Three L2 candidates score French-line coherence:
  bigram      bigram_lm fluency (the old ceiling)
  trigram     trigram_lm fluency (OpenSubtitles, 18.7M tokens)
  embed       MiniLM topical cohesion: mean cos(word, sentence-centroid)
  tri*embed   product -- syntax x topic

Two honest tests on HELD-OUT subtitle sentences (never seen by the trigram):
  A. real vs SHUFFLED (same words, broken syntax)  -> measures GRAMMAR
  B. real vs RANDOM-VOCAB salad (same length)      -> measures SENSE
An L2 for verse selection must win BOTH: shuffled catches embeds' known
word-order blindness; salad catches n-grams' vocabulary blindness.

Run: python l2_bakeoff.py [--n 300]
"""
from __future__ import annotations

import argparse
import random
import re

import numpy as np

import bigram_lm
import trigram_lm

TOK = re.compile(r"[a-zà-ÿ']+")


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    rng = random.Random(0)

    real = []
    for line in open("/tmp/fr_heldout.txt", encoding="utf-8", errors="ignore"):
        ws = TOK.findall(line.lower())
        if 5 <= len(ws) <= 9:
            real.append(ws)
        if len(real) >= args.n:
            break
    vocab = sorted({w for ws in real for w in ws})
    shuffled = []
    for ws in real:
        s = ws[:]
        while True:
            rng.shuffle(s)
            if s != ws:
                break
        shuffled.append(s[:])
    salad = [[rng.choice(vocab) for _ in ws] for ws in real]
    print(f"{len(real)} real / shuffled / salad lines (held-out)\n")

    BI = bigram_lm.load("fr")
    TRI = trigram_lm.load("fr")

    from sentence_transformers import SentenceTransformer
    M = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def embed_cohesion(batches):
        """mean cos(word-embed, centroid) per sentence -- topical tightness."""
        out = []
        for ws in batches:
            V = M.encode(ws, show_progress_bar=False, normalize_embeddings=True)
            c = V.mean(0)
            c /= (np.linalg.norm(c) + 1e-9)
            out.append(float((V @ c).mean()))
        return out

    sets = {"real": real, "shuffled": shuffled, "salad": salad}
    scores = {}
    for name, data in sets.items():
        bi = [BI.fluency(ws) for ws in data]
        tri = [TRI.fluency(ws) for ws in data]
        em = embed_cohesion(data)
        scores[name] = {"bigram": bi, "trigram": tri, "embed": em,
                        "tri*embed": [t * e for t, e in zip(tri, em)]}

    print(f"{'L2':10s} {'A real|shuffled':>16s} {'B real|salad':>14s}")
    print("-" * 44)
    best, bestv = None, -1
    for m in ("bigram", "trigram", "embed", "tri*embed"):
        a = auc(scores["real"][m], scores["shuffled"][m])
        b = auc(scores["real"][m], scores["salad"][m])
        both = min(a, b)
        if both > bestv:
            best, bestv = m, both
        print(f"{m:10s} {a:16.3f} {b:14.3f}")
    print(f"\nWINNER (max of min(A,B)): {best}")
    print("Reading: embeds are word-order-blind (test A exposes them); n-grams "
          "are vocabulary-blind at fixed syntax (test B). The right L2 is the "
          "one that survives both -- use it as the verse selector.")


if __name__ == "__main__":
    main()
