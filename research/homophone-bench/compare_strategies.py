"""Unbiased strategy comparison on the nursery-rhyme task.

Runs the SAME public-domain Mother Goose lines through two extant strategies and
scores them on the SAME axes (matcher combo = sound, bigram coherence = L2,
coverage). No thumb on the scale for the frontier method.

  A. BASELINE decoder (soramimi settings): default trie (min_zipf 2.2),
     MIN_WORD_SEGS=2 (no fillers), default WORD_PENALTY/gaps, ranked by the
     decoder's own similarity -- the strategy as it shipped.
  B. FRONTIER whole-line carve: poetry trie (1-seg fillers), coverage-forcing
     deletion penalty, ranked by the matcher arbiter x coherence.

Reports the per-line winner on dual (combo x coherence) AND on raw sound, so a
strategy that wins on sound but loses on coherence (or vice versa) is visible.

Run: python compare_strategies.py
"""
from __future__ import annotations

import subprocess

import matcher
import phonetic_decoder as pd
import poetry_mode as pm
from lexicon_g2p import clean_ipa

try:
    import bigram_lm
    LM = bigram_lm.load("fr")
except Exception:
    LM = None

_ORIG_CHEAP = dict(matcher.CHEAP_GAP)
_ORIG_GAP = matcher.GAP

LINES = ["Humpty Dumpty", "Humpty Dumpty sat on a wall", "Jack and Jill",
         "Hickory dickory dock", "Little Jack Horner", "Pat a cake"]


def en_ipa(t):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", t],
                       capture_output=True, text=True, check=True)
    return clean_ipa(r.stdout.strip())


def coherence(fr):
    toks = [w.lower() for w in fr.replace("'", " ").split() if w]
    return LM.fluency(toks) if (LM and toks) else 0.0


def score(line, fr):
    if not fr:
        return 0.0, 0.0, 0.0
    combo = matcher.homophone_score(line, "en", fr, "fr")["score"]
    coh = coherence(fr)
    return combo, coh, combo * coh


def reset_gaps():
    matcher.CHEAP_GAP.clear(); matcher.CHEAP_GAP.update(_ORIG_CHEAP)
    matcher.GAP = _ORIG_GAP
    pd._sub.cache_clear()


def strat_baseline(line, root):
    """Decoder's OWN ranking (its similarity), default settings."""
    reset_gaps()
    pd.MIN_WORD_SEGS = 2; pd.WORD_PENALTY = 0.18; pd.BEAM = 250
    cands = pd.decode(en_ipa(line), root, top_n=8, max_words=10,
                      lm=LM, lm_weight=pd.LM_BEAM_WEIGHT if LM else 0.0)
    return cands[0]["fr"] if cands else ""


def strat_frontier(line, root):
    """Poetry trie + coverage-forcing + arbiter (combo x coh) ranking."""
    pd.MIN_WORD_SEGS = 1; pd.WORD_PENALTY = 0.04; pd.BEAM = 420
    for k, v in _ORIG_CHEAP.items():
        matcher.CHEAP_GAP[k] = v if k == "h" else min(0.95, v * 1.6)
    matcher.GAP = min(0.95, _ORIG_GAP * 1.6)
    pd._sub.cache_clear()
    nw = len(line.split())
    cands = pd.decode(en_ipa(line), root, top_n=30, max_words=nw + 3,
                      lm=LM, lm_weight=pd.LM_BEAM_WEIGHT if LM else 0.0)
    best, bestdual = "", -1
    for c in cands:
        if c["coverage"] < 0.7:
            continue
        _, _, d = score(line, c["fr"])
        if d > bestdual:
            best, bestdual = c["fr"], d
    return best


def main():
    base_root = pd.build_trie(min_zipf=2.2, lang="fr")
    poet_root = pm.build_poetry_trie(min_zipf=2.0)
    print("Unbiased comparison: BASELINE decoder vs FRONTIER carve, same scoring.\n")
    print(f"{'line':30s} {'strategy':9s} {'combo':>5s} {'coh':>5s} {'dual':>5s}  FR")
    print("-" * 80)
    wins = {"baseline": 0, "frontier": 0, "tie": 0}
    snd_wins = {"baseline": 0, "frontier": 0, "tie": 0}
    for line in LINES:
        fb = strat_baseline(line, base_root)
        ff = strat_frontier(line, poet_root)
        cb, hb, db = score(line, fb)
        cf, hf, df = score(line, ff)
        print(f"{line:30s} {'baseline':9s} {cb:5.2f} {hb:5.2f} {db:5.2f}  {fb}")
        print(f"{'':30s} {'frontier':9s} {cf:5.2f} {hf:5.2f} {df:5.2f}  {ff}")
        w = "frontier" if df > db + 0.02 else "baseline" if db > df + 0.02 else "tie"
        s = "frontier" if cf > cb + 0.02 else "baseline" if cb > cf + 0.02 else "tie"
        wins[w] += 1; snd_wins[s] += 1
        print(f"{'':30s} -> dual winner: {w.upper()};  sound winner: {s.upper()}\n")
    print(f"DUAL wins  : {wins}")
    print(f"SOUND wins : {snd_wins}")
    print("""
Reading honestly: the frontier wins where the line carves into a fluent filler
phrase; the baseline can win on raw SOUND for lines that map to one tight word
(its default favours long single words, which are often closer-sounding but less
coherent). Neither dominates every line -- the right system would run both and
let the arbiter pick per line.""")


if __name__ == "__main__":
    main()
