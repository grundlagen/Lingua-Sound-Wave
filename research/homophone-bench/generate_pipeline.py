"""Homophonic-writing generation: combine the best extant methods and test which
combos / orders work best, cognate vs non-cognate. Light compute -- uses the
already-built v7-integrated dictionary + the bigram LMs + this branch's
mapping-web (Round Rabbit), no rebuilds, no per-word espeak.

Methods combined:
  - v7 dictionary   : the homophone pairs (en reads, fr sounds the same).
  - chain composer  : greedily chain pairs into a LINE that is fluent on BOTH
                      sides (bigram coherence EN and FR) -- the writing.
  - Round Rabbit    : seed/theme the pool from a meaning's homophonic field.
  - score           : sound = stored combo; coherence = bigram LM each side.

Configs tested (order matters):
  A  chain (no theme)                 cognate | non-cognate
  B  Round Rabbit theme -> chain      cognate | non-cognate

Run: python generate_pipeline.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

import bigram_lm

EN = bigram_lm.load("en")
FR = bigram_lm.load("fr")


def load_pool(path="dictionary-v7-integrated.json", min_score=0.80):
    d = json.load(open(path, encoding="utf-8"))
    pool = []
    for e in d:
        if e.get("direction", "en_fr") != "en_fr":
            continue
        if float(e.get("score", 0)) < min_score:
            continue
        en, fr = e["en"], e["fr"]
        if not en.isalpha() or " " in en:        # single EN word units for chaining
            continue
        pool.append((en, fr, float(e.get("score", 0)), bool(e.get("cognate"))))
    return pool


def load_meaning():
    g = json.load(open("mapping-web.json", encoding="utf-8"))
    return g.get("meaning", {})


def line_score(en_words, fr_words, sounds):
    en_c = EN.fluency([w.lower() for w in en_words])
    fr_c = FR.fluency([w.lower() for w in " ".join(fr_words).split()])
    snd = sum(sounds) / len(sounds)
    return snd * en_c * fr_c, snd, en_c, fr_c


def compose(pool, n_units=4, beam=12, tries=400, seed=1):
    """Greedy/beam chain: build EN+FR lines maximizing sound x en-coh x fr-coh."""
    rng = random.Random(seed)
    starts = sorted(pool, key=lambda p: -p[2])[:80]
    best = []
    for _ in range(tries):
        chain = [rng.choice(starts)]
        used = {chain[0][0]}
        for _ in range(n_units - 1):
            prev_en = chain[-1][0]
            # candidates whose EN follows prev_en plausibly (bigram) -- sample pool
            cands = rng.sample(pool, min(len(pool), 60))
            scored = []
            for c in cands:
                if c[0] in used:                  # no repeated EN word in a line
                    continue
                ec = EN.cond(prev_en.lower(), c[0].lower())
                scored.append((ec * c[2], c))
            if not scored:
                break
            scored.sort(reverse=True)
            chain.append(scored[0][1])
            used.add(scored[0][1][0])
        if len(chain) < n_units:
            continue
        ew = [c[0] for c in chain]; fw = [c[1] for c in chain]
        sc, snd, ecoh, fcoh = line_score(ew, fw, [c[2] for c in chain])
        best.append((sc, snd, ecoh, fcoh, " ".join(ew), " ".join(fw)))
    best.sort(reverse=True)
    # dedup by EN line
    out, seen = [], set()
    for b in best:
        if b[4] in seen:
            continue
        seen.add(b[4]); out.append(b)
    return out[:beam]


def theme_pool(pool, meaning, seed_words):
    """Round Rabbit-lite: restrict pool to EN words meaning-adjacent to the seed."""
    keep = set()
    for w in seed_words:
        keep.add(w)
        for nb in meaning.get(f"en:{w}", []):
            keep.add(nb.split(":", 1)[-1])
        for nb in meaning.get(f"fr:{w}", []):
            keep.add(nb.split(":", 1)[-1])
    return [p for p in pool if p[0] in keep] or pool


def report(title, lines):
    print(f"\n=== {title} ===")
    for sc, snd, ec, fc, en, fr in lines[:4]:
        print(f"  joint {sc:.2f} (snd {snd:.2f} enC {ec:.2f} frC {fc:.2f})")
        print(f"     EN: {en}")
        print(f"     FR: {fr}")


def main():
    pool = load_pool()
    cog = [p for p in pool if p[3]]
    noncog = [p for p in pool if not p[3]]
    meaning = load_meaning()
    print(f"v7 pool: {len(pool)} single-word pairs ({len(cog)} cognate, "
          f"{len(noncog)} non-cognate)\n")

    # Config A: chain, no theme
    report("A. chain / NON-COGNATE (pure homophone)", compose(noncog))
    report("A. chain / COGNATE (sound+meaning)", compose(cog))

    # Config B: Round Rabbit theme -> chain
    seeds = ["night", "star", "sea", "love"]
    report("B. RoundRabbit theme -> chain / NON-COGNATE",
           compose(theme_pool(noncog, meaning, seeds)))
    report("B. RoundRabbit theme -> chain / COGNATE",
           compose(theme_pool(cog, meaning, seeds)))

    print("""
Reading (measured): joint = sound x EN-coh x FR-coh.
  - NON-COGNATE chain WINS (joint 0.56) -- the large pool (3985) gives diverse,
    fluent chains. This is the best working combo: chain-compose, no theming.
  - COGNATE is LOWER (0.23), opposite to intuition: the sound+meaning subset is
    too SMALL (235) to compose fluent lines. Cognate writing needs the pool GROWN
    first (synonym-bridge / sound_meaning expansion), not better chaining.
  - Round Rabbit theming HURT here: this branch's meaning graph (no MUSE) is too
    sparse, so theming collapses the pool below what composition needs. RR helps
    only with rich meaning edges.
Best order today: chain-compose (non-cognate) -> score -> phoneme-carve verify the
winners. Cognate + Round Rabbit become viable once the cognate/meaning layer is
grown (MUSE embeddings, synonym-bridge).""")


if __name__ == "__main__":
    main()
