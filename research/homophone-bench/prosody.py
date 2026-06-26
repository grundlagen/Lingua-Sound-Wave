"""Prosody-aware sound scoring -- stress-weighted, and DIVERGED per language.

Homophony lives in the stressed syllables; the ear forgives a fudged unstressed
2nd/3rd syllable. And the two languages carry prominence differently, so "sounds
right in English" and "sounds right in French" are scored separately:

  English  stress-timed: lexical stress early/mid, unstressed vowels reduce to
           schwa, declarative pitch DECLINES to a low final (falling).
  French   syllable-timed: even syllables, prominence is PHRASE-FINAL, the group
           rises toward its last syllable.

So a good homophonic line (a) preserves the English stressed segments under the
cross-lingual alignment, AND (b) reads with natural FRENCH prosody (final-syllable
prominence, even rhythm). Extra tricks folded in: onset salience (onsets carry
more identity than codas), rhythm/syllable-count match, French final-consonant
leniency.

espeak supplies the stress marks both ways (EN hˈʌmpti, FR pətˈi).
Test: python prosody.py
"""
from __future__ import annotations

import subprocess
import unicodedata

import numpy as np

import matcher

W_PRIMARY, W_SECONDARY, W_UNSTRESSED = 1.0, 0.6, 0.3
VOWELS = set("aeiouyɑɐɒæɛɜəɪɔœøʊʌyɨʉɘɵɤoɯ")


def _is_vowel(seg: str) -> bool:
    base = "".join(c for c in unicodedata.normalize("NFD", seg)
                   if not unicodedata.combining(c))
    return bool(base) and base[0] in VOWELS


def _espeak(text, lang):
    voice = "en-us" if lang == "en" else "fr"
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, text],
                       capture_output=True, text=True, check=True)
    return unicodedata.normalize("NFD", r.stdout.strip())


def stressed_segments(text, lang):
    """-> (segments, weights) with per-segment prominence; onsets weighted up."""
    segs_all, w_all = [], []
    for word in _espeak(text, lang).split():
        demark, stops = "", {}
        for ch in word:
            if ch == "ˈ":
                stops[len(demark)] = W_PRIMARY; continue
            if ch == "ˌ":
                stops[len(demark)] = W_SECONDARY; continue
            if ch in ".‿|‖ˑː":
                continue
            demark += ch
        segs = matcher._segs(demark)
        off, seg_w, vidx = 0, [], []
        for i, s in enumerate(segs):
            w = W_UNSTRESSED
            for o, sw in stops.items():
                if off <= o <= off + len(s):
                    w = max(w, sw)
            seg_w.append(w)
            if _is_vowel(s):
                vidx.append(i)
            off += len(s)
        for i, s in enumerate(segs):           # consonants inherit nearest vowel
            if i in vidx:
                continue
            near = [seg_w[j] for j in vidx if abs(j - i) <= 1]
            if near:
                # onset (consonant just BEFORE a vowel) gets full inheritance;
                # coda a touch less (onsets carry more perceptual identity)
                onset = any(j > i for j in vidx if abs(j - i) == 1)
                seg_w[i] = max(seg_w[i], max(near) * (1.0 if onset else 0.85))
        segs_all += list(segs); w_all += seg_w
    return segs_all, w_all


def _syllable_weights(segs, weights):
    """prominence per syllable = weight of its nucleus vowel, in order."""
    return [weights[i] for i, s in enumerate(segs) if _is_vowel(s)]


def _segdist(a, b):
    if a == b:
        return 0.0
    f = matcher._equiv_floor(a, b)
    va, vb = matcher._vecs(a), matcher._vecs(b)
    if len(va) == 0 or len(vb) == 0:
        return min(f, 0.6)
    d = min(1.0, float(np.abs(va[0] - vb[0]).sum()) / (2.0 * matcher.N_FEATURES) / matcher.SHARPEN)
    return min(f, d)


def _aligned_cost(sa, wa, sb, wb):
    n, m = len(sa), len(sb)
    GAP = 0.42
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + GAP * wa[i - 1]
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + GAP * wb[j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            w = (wa[i - 1] + wb[j - 1]) / 2.0
            D[i][j] = min(D[i - 1][j - 1] + _segdist(sa[i - 1], sb[j - 1]) * w,
                          D[i - 1][j] + GAP * wa[i - 1],
                          D[i][j - 1] + GAP * wb[j - 1])
    tot = sum(wa) + sum(wb)
    return 2.0 * D[n][m] / tot if tot else 1.0


def french_naturalness(syl_w) -> float:
    """French is syllable-timed with PHRASE-FINAL prominence. Reward the last
    syllable being (near-)max prominent and an even rhythm; penalise a strong
    early stress with a weak ending (that is an English contour, not French)."""
    if not syl_w:
        return 0.0
    last, mx = syl_w[-1], max(syl_w)
    final_prom = last / mx if mx else 0.0
    evenness = 1.0 - (np.std(syl_w) / (np.mean(syl_w) + 1e-9))   # syllable-timed
    return float(max(0.0, min(1.0, 0.6 * final_prom + 0.4 * max(0.0, evenness))))


def english_naturalness(syl_w) -> float:
    """English declines: prominence early/mid, NOT on the final syllable; tail
    reduces. Reward an early/mid peak and a lower final."""
    if not syl_w:
        return 0.0
    peak = int(np.argmax(syl_w))
    early_peak = 1.0 - peak / max(1, len(syl_w) - 1)             # peak near front
    falling = 1.0 if syl_w[-1] <= np.mean(syl_w) else 0.4        # low final
    return float(max(0.0, min(1.0, 0.6 * early_peak + 0.4 * falling)))


def prosodic_score(en, fr, diverged=True):
    """Cross-lingual stress-weighted match, blended with FRENCH-side naturalness
    (the FR rendering must read as French) and rhythm/syllable-count agreement."""
    sa, wa = stressed_segments(en, "en")
    sb, wb = stressed_segments(fr, "fr")
    if not sa or not sb:
        return 0.0
    match = max(0.0, 1.0 - _aligned_cost(sa, wa, sb, wb))
    sy_a, sy_b = _syllable_weights(sa, wa), _syllable_weights(sb, wb)
    rhythm = 1.0 - abs(len(sy_a) - len(sy_b)) / max(1, len(sy_a) + len(sy_b))
    fr_nat = french_naturalness(sy_b)
    if not diverged:
        return 0.85 * match + 0.15 * rhythm
    return 0.6 * match + 0.2 * fr_nat + 0.2 * rhythm


def both_sides(en, fr):
    """diverged report: how it sounds as English vs as French."""
    sa, wa = stressed_segments(en, "en")
    sb, wb = stressed_segments(fr, "fr")
    return {
        "match": round(max(0.0, 1.0 - _aligned_cost(sa, wa, sb, wb)), 2),
        "english_contour(en)": round(english_naturalness(_syllable_weights(sa, wa)), 2),
        "french_contour(fr)": round(french_naturalness(_syllable_weights(sb, wb)), 2),
        "score": round(prosodic_score(en, fr), 2),
    }


if __name__ == "__main__":
    pairs = [
        ("Humpty Dumpty", "un petit un petit"),
        ("it for", "est fort"),
        ("to tell", "t elle"),
        ("happy", "happo"),               # unstressed fudge -> cheap
        ("happy", "hoppy"),               # stressed fudge
        ("the door", "un voile d or"),    # natural French final prominence
    ]
    for en, fr in pairs:
        print(f"  {en:14s} -> {fr:18s} {both_sides(en, fr)}")
