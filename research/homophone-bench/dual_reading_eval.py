"""Dual-reading evaluation: the CORRECT objective for homophonic verse.

Corrected target (per the project owner): the value is NOT two poems sharing a
theme -- anyone can do that, and it throws away the homophone constraint that IS
the art. The value is ONE phoneme stream that reads as SENSICAL text in BOTH
languages at once. So an item is good iff:

    sound_fidelity(source, target) high     -- one sound, two readings
    AND source reads as coherent L1
    AND target reads as coherent L2          -- the hard, currently-missing axis

There is deliberately NO "the two mean the same" term: the gold homophonic
poems (van Rooten's Mother Goose -> French) do NOT translate meaning, and
demanding it collapses output to sound-faithful nonsense (see CENTRAL_PROBLEM.md).

The gold corpus is human-crafted homophonic verse where high sound AND high L2
coherence demonstrably coexist -- the existence proof the machine must reach:
  - now:   van Rooten, "Mots d'Heures: Gousses, Rames" (Mother Goose -> FR);
           later "N'Heures Souris Rames" (de Kay), "Mörder Guss Reims" (DE).
  - later: classical Arabic homophonic / jinas (paronomasia) & dual-reading
           verse -- a far older, richer constraint tradition.

THE measurement this harness exists for (impetus I): with the SAME words (so
sound is held fixed), can an L2-coherence model rank the human line above its
own shuffle? The margin is exactly the strength of the missing component. The
bigram LM gives a weak ~0.2; a real LM/LLM should give much more. That margin,
on this gold, is the number that tells you the coherence model is good enough to
drive generation.

Run: python dual_reading_eval.py
"""
from __future__ import annotations

import random
import statistics

import matcher

try:
    import bigram_lm
    _LM = {"fr": bigram_lm.load("fr"), "en": bigram_lm.load("en")}
except Exception:                                   # pragma: no cover
    _LM = {}

from wordfreq import zipf_frequency


def coherence(text: str, lang: str) -> float:
    """Best-available L2/L1 coherence in [0,1]. Bigram LM if present (it scores
    grammatical adjacency), else mean-zipf (a weak placeholder that CANNOT see
    word order -- which is the whole point of why a real model is needed)."""
    toks = [t.lower() for t in text.replace("'", " ").split() if t]
    if not toks:
        return 0.0
    if lang in _LM:
        return _LM[lang].fluency(toks)
    return sum(min(zipf_frequency(t, lang), 6.0) / 6.0 for t in toks) / len(toks)


def shuffle_coherence(text: str, lang: str, k: int = 12) -> float:
    """Mean coherence over k word-shuffles: SAME words (same sound material),
    broken syntax. The controlled negative -- isolates coherence from vocabulary
    and from sound, because only word ORDER changes."""
    toks = text.replace("'", " ").split()
    if len(toks) < 3:
        return coherence(text, lang)
    rng = random.Random(0)
    vals = []
    for _ in range(k):
        s = toks[:]
        rng.shuffle(s)
        vals.append(coherence(" ".join(s), lang))
    return statistics.mean(vals)


def load_gold(path="gold-dual-readings.tsv"):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4 and p[2] and p[3]:
                rows.append(dict(source_lang=p[0], target_lang=p[1],
                                 source_text=p[2], target_text=p[3],
                                 provenance=p[4] if len(p) > 4 else ""))
    return rows


def evaluate():
    gold = load_gold()
    print(f"{len(gold)} gold dual-readings (human homophonic verse).\n")
    sounds, margins = [], []
    print(f"{'source (L1)':32s} {'snd':>5s} {'L1coh':>6s} "
          f"{'L2real':>7s} {'L2shuf':>7s} {'margin':>7s}")
    print("-" * 76)
    for g in gold:
        sl, tl = g["source_lang"], g["target_lang"]
        snd = matcher.homophone_score(g["source_text"], sl,
                                      g["target_text"], tl)["score"]
        l1 = coherence(g["source_text"], sl)
        l2_real = coherence(g["target_text"], tl)
        l2_shuf = shuffle_coherence(g["target_text"], tl)
        margin = l2_real - l2_shuf
        sounds.append(snd)
        margins.append(margin)
        print(f"{g['source_text'][:32]:32s} {snd:5.2f} {l1:6.2f} "
              f"{l2_real:7.2f} {l2_shuf:7.2f} {margin:+7.2f}")
        print(f"  L2: {g['target_text']}")

    print("\n--- readout ---")
    print(f"mean sound fidelity (homophone constraint held): "
          f"{statistics.mean(sounds):.3f}")
    print(f"mean L2 coherence margin (human verse vs same-word shuffle): "
          f"{statistics.mean(margins):+.3f}")
    print(f"  scorer in use: {'bigram-LM' if _LM else 'mean-zipf (no order!)'}")
    print("""
Interpretation:
  - sound fidelity confirms these ARE homophonic (one stream, two readings).
  - the L2 margin is the strength of the coherence model on the gold. The
    bigram gives a weak signal; the goal (impetus I) is an L2 model whose
    margin on this set is large enough to RANK human craft over sound-matched
    salad reliably -- that model is what drives both-sides-free generation.
  - to grow the gold: add more Mots d'Heures / N'Heures Souris Rames lines,
    then classical Arabic jinas / dual-reading verse, into gold-dual-readings.tsv.
""")


if __name__ == "__main__":
    evaluate()
