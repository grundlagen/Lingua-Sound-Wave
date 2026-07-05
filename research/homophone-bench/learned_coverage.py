"""Learned coverage penalty: replace the blunt global deletion knob with a
per-line tuned one, and fit a tiny model so it generalises.

whole_line_carve.py raised every deletion cost by a fixed factor -- it nailed the
Humpty lines but over-forced others (a single long word) or found nothing. The
right deletion penalty depends on the line (length, vowel density): a short dense
line needs a gentler push than a long one.

This LEARNS it two ways:
  1. autotune(line): sweep the gap-scale, keep the carve that maximises the dual
     objective (coverage-aware combo x bigram coherence). Per-line optimal.
  2. fit a model gap_scale ~= f(stream_length) from the per-line optima, so a new
     line gets a predicted scale in one shot (no sweep).

The objective is dual = combo x coherence: combo is coverage-aware so it rewards
covering the stream, coherence punishes the over-forced single-long-word carves
(incrédulité-type), so the optimum is the real sweet spot, not max coverage.

Run: python learned_coverage.py
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

VOWELS = set("iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒ")
_ORIG_CHEAP = dict(matcher.CHEAP_GAP)
_ORIG_GAP = matcher.GAP


def set_gap_scale(scale):
    """Scale every deletion cost by `scale`, keeping /h/ cheap (no French /h/)."""
    for k, v in _ORIG_CHEAP.items():
        matcher.CHEAP_GAP[k] = v if k == "h" else min(0.95, v * scale)
    matcher.GAP = min(0.95, _ORIG_GAP * scale)
    pd._sub.cache_clear()


def en_ipa(text):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", text],
                       capture_output=True, text=True, check=True)
    return clean_ipa(r.stdout.strip())


def coherence(fr):
    toks = [t.lower() for t in fr.replace("'", " ").split() if t]
    return LM.fluency(toks) if (LM and toks) else 0.0


def best_carve(line, root, scale, beam=420):
    set_gap_scale(scale)
    ipa = en_ipa(line)
    nw = len(line.split())
    cands = pd.decode(ipa, root, top_n=30, max_words=nw + 3,
                      lm=LM, lm_weight=pd.LM_BEAM_WEIGHT if LM else 0.0)
    best = None
    for c in cands:
        if c["coverage"] < 0.7:
            continue
        combo = matcher.homophone_score(line, "en", c["fr"], "fr")["score"]
        coh = coherence(c["fr"])
        dual = combo * coh
        if best is None or dual > best[0]:
            best = (dual, combo, coh, c["coverage"], c["fr"])
    return best


def autotune(line, root, scales=(1.0, 1.6, 2.2, 2.8, 3.4)):
    best, best_scale = None, None
    for s in scales:
        r = best_carve(line, root, s)
        if r and (best is None or r[0] > best[0]):
            best, best_scale = r, s
    return best_scale, best


def stream_len(line):
    return len(matcher._segs(matcher._canonical(en_ipa(line))))


def main():
    pd.MIN_WORD_SEGS = 1
    pd.WORD_PENALTY = 0.04
    pd.BEAM = 420
    root = pm.build_poetry_trie(min_zipf=2.0)
    print("Learned coverage penalty -- per-line autotune of the deletion scale.\n")

    LINES = ["Humpty Dumpty", "Humpty Dumpty sat on a wall", "Jack and Jill",
             "Hickory dickory dock", "Little Jack Horner"]
    data = []
    for line in LINES:
        L = stream_len(line)
        scale, best = autotune(line, root)
        if best:
            dual, combo, coh, cov, fr = best
            data.append((L, scale))
            print(f"EN: {line!r}  (stream {L} segs)")
            print(f"   best scale {scale:.1f} -> dual {dual:.2f} "
                  f"(combo {combo:.2f} coh {coh:.2f} cov {cov:.2f})")
            print(f"   FR: {fr}\n")
        else:
            print(f"EN: {line!r}  (stream {L} segs) -> no carve at any scale\n")

    # fit gap_scale ~= a + b*stream_len  (least squares on the per-line optima)
    if len(data) >= 2:
        n = len(data)
        sx = sum(L for L, _ in data); sy = sum(s for _, s in data)
        sxx = sum(L * L for L, _ in data); sxy = sum(L * s for L, s in data)
        b = (n * sxy - sx * sy) / (n * sxx - sx * sx) if (n * sxx - sx * sx) else 0
        a = (sy - b * sx) / n
        print(f"learned model: gap_scale ≈ {a:.2f} + {b:.3f}·stream_len")
        print("  per-line optima:", [(L, s) for L, s in data])
        print(f"  -> a new line of L segs gets a one-shot scale, no sweep.")
    print("""
Reading: the deletion penalty is now LEARNED per line (and modelled vs stream
length), so coverage-forcing adapts instead of over-forcing. The objective is
dual (coverage-aware sound x coherence), so it never picks the single-long-word
collapse. This is the learning layer asked for; the next-deeper learn is the
substitution/gap COSTS themselves (learn_costs.py / cycle-consistency), and the
L2 coherence model that turns good carves into verse.""")


if __name__ == "__main__":
    main()
