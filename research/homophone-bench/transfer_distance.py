"""Q1: the OPPOSITE of a loop. Loops certify pairs whose meaning round-trips
home (safe, but biased to near-cognate morphology like feel~files). The
interesting art is the transfer chain that stays SOUND-bridged while traveling
FAR in meaning. We already have those in chain-web-v7u.tsv (seed EN -> endpoint
FR, >=1 sound + >=1 meaning hop). Rank them by semantic DISTANCE between seed
and endpoint, measured with the SAME multilingual encoder the weave used.

score_transfer = sound_quality * (1 - cos(seed, endpoint))
  high  = sounds-alike chain that lands on a DISTANT meaning  (frame-evasion gold)
  low   = endpoint means ~the same as the seed (loop-like / cognate)

No engine edits: reads the weave's own output + re-encodes the handful of words.
Run: python transfer_distance.py
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

WEB = "chain-web-v7u.tsv"


def load_web(path=WEB):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            src, dst, hops, quality, chain = line.rstrip("\n").split("\t")
            rows.append((src, dst, int(hops), float(quality), chain))
    return rows


def main():
    rows = load_web()
    words = sorted({r[0] for r in rows} | {r[1] for r in rows})
    print(f"{len(rows)} transfer chains, {len(words)} unique endpoints; encoding...",
          file=sys.stderr)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vecs = np.asarray(model.encode(words, batch_size=256,
                                   normalize_embeddings=True,
                                   show_progress_bar=False), dtype=np.float32)
    idx = {w: i for i, w in enumerate(words)}

    scored = []
    for src, dst, hops, q, chain in rows:
        cos = float(vecs[idx[src]] @ vecs[idx[dst]])
        dist = 1.0 - cos
        scored.append((q * dist, q, cos, hops, src, dst, chain))
    scored.sort(reverse=True)

    # FAR-MEANING transfers: high sound quality, low semantic similarity
    print("\n=== FAR-MEANING sound bridges (don't loop to the same meaning) ===")
    print("   score  snd   cos hops   seed -> endpoint")
    shown = 0
    for s, q, cos, h, src, dst, chain in scored:
        if cos > 0.45:                      # skip near-synonym/cognate landings
            continue
        print(f"  {s:5.2f} {q:4.2f} {cos:+.2f}  {h}   {src} -> {dst}")
        print(f"          {chain}")
        shown += 1
        if shown >= 18:
            break

    # contrast: the loop-like end (endpoint ~ seed in meaning)
    near = sorted(scored, key=lambda t: -t[2])[:6]
    print("\n=== NEAR-MEANING (loop-like: endpoint means ~the seed) — for contrast ===")
    for s, q, cos, h, src, dst, chain in near:
        print(f"  cos {cos:+.2f} snd {q:.2f}  {src} -> {dst}")

    # distribution
    coss = np.array([t[2] for t in scored])
    print(f"\ncos(seed,endpoint) over {len(coss)} transfers: "
          f"mean {coss.mean():+.2f}  <0.2: {(coss<0.2).mean()*100:.0f}%  "
          f">0.5: {(coss>0.5).mean()*100:.0f}%")
    print("Reading: most transfer chains ALREADY land far in meaning (low cos) — "
          "the weave is mostly non-looping transfer, not cognate echo. The top of "
          "this list is the writeable gold: sounds-alike across the seam, means "
          "something genuinely different.")

    with open("transfer-ranked-v7u.tsv", "w", encoding="utf-8") as f:
        f.write("score\tsound\tcos\thops\tseed\tendpoint\tchain\n")
        for s, q, cos, h, src, dst, chain in scored:
            f.write(f"{s:.4f}\t{q:.3f}\t{cos:.3f}\t{h}\t{src}\t{dst}\t{chain}\n")
    print("\nwrote transfer-ranked-v7u.tsv (all transfers, ranked by sound x distance)")


if __name__ == "__main__":
    main()
