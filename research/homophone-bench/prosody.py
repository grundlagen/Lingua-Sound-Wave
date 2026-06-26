"""Prosody-aware sound scoring: weight STRESSED syllables, cheap out unstressed.

The plain matcher strips espeak's stress marks (ˈ ˌ) and treats every segment
equally. But homophony lives in the STRESSED syllables -- the ear forgives a
fudged 2nd/3rd (unstressed) syllable. And the languages differ: English is
stress-timed (early stress, reduced tails, falling end); French is syllable-timed
with phrase-final prominence (rises to the end). espeak marks both
(EN hˈʌmpti, FR pətˈi), so we can use it.

prosodic_score(en, fr): a stress-WEIGHTED featural alignment -- a mismatch on a
stressed segment costs full; on an unstressed segment it is cheap. Plus a small
METER term rewarding a similar strong-beat count/contour.

Test on CPU: python prosody.py
"""
from __future__ import annotations

import subprocess
import unicodedata

import numpy as np

import matcher

W_PRIMARY, W_SECONDARY, W_UNSTRESSED = 1.0, 0.6, 0.3


def _espeak(text, lang):
    voice = "en-us" if lang == "en" else "fr"
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, text],
                       capture_output=True, text=True, check=True)
    return unicodedata.normalize("NFD", r.stdout.strip())


def stressed_segments(text, lang):
    """-> (segments, weights): each segment's prominence (stress) weight."""
    segs_all, w_all = [], []
    for word in _espeak(text, lang).split():
        demark, stops = "", {}            # char-offset in demark -> stress weight
        for ch in word:
            if ch == "ˈ":
                stops[len(demark)] = W_PRIMARY; continue
            if ch == "ˌ":
                stops[len(demark)] = W_SECONDARY; continue
            if ch in ".‿|‖ˑː":            # drop syllable/tie/length marks
                continue
            demark += ch
        segs = matcher._segs(demark)
        # map char offsets -> segment index
        off, seg_w, vowel_idx = 0, [], []
        for i, s in enumerate(segs):
            base = "".join(c for c in s if not unicodedata.combining(c))
            w = W_UNSTRESSED
            for o, sw in stops.items():
                if off <= o <= off + len(s):
                    w = max(w, sw)
            seg_w.append(w)
            if base and matcher._is_vowel(base) if hasattr(matcher, "_is_vowel") \
                    else (base and base[0] in "aeiouyɑɐɒæɛɜəɪɔœøʊʌyɨʉ"):
                vowel_idx.append(i)
            off += len(s)
        # propagate: a consonant inherits the max weight of adjacent vowels
        for i, s in enumerate(segs):
            if i in vowel_idx:
                continue
            near = [seg_w[j] for j in vowel_idx if abs(j - i) <= 1]
            if near:
                seg_w[i] = max(seg_w[i], max(near))
        segs_all += list(segs); w_all += seg_w
    return segs_all, w_all


def _segdist(a, b):
    if a == b:
        return 0.0
    f = matcher._equiv_floor(a, b)
    va, vb = matcher._vecs(a), matcher._vecs(b)
    if len(va) == 0 or len(vb) == 0:
        return min(f, 0.6)
    d = min(1.0, float(np.abs(va[0] - vb[0]).sum()) / (2.0 * matcher.N_FEATURES) / matcher.SHARPEN)
    return min(f, d)


def prosodic_score(en, fr):
    sa, wa = stressed_segments(en, "en")
    sb, wb = stressed_segments(fr, "fr")
    if not sa or not sb:
        return 0.0
    n, m = len(sa), len(sb)
    GAP = 0.42
    # weighted Needleman-Wunsch: cost scaled by the prominence of the segments
    INF = 1e9
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + GAP * wa[i - 1]
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + GAP * wb[j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            w = (wa[i - 1] + wb[j - 1]) / 2.0
            sub = D[i - 1][j - 1] + _segdist(sa[i - 1], sb[j - 1]) * w
            dele = D[i - 1][j] + GAP * wa[i - 1]
            ins = D[i][j - 1] + GAP * wb[j - 1]
            D[i][j] = min(sub, dele, ins)
    total_w = sum(wa) + sum(wb)
    cost = 2.0 * D[n][m] / total_w if total_w else 1.0
    sound = max(0.0, 1.0 - cost)
    # meter term: similar count of strong beats (primary-stressed vowels)
    beats_a = sum(1 for w in wa if w >= W_PRIMARY)
    beats_b = sum(1 for w in wb if w >= W_PRIMARY)
    meter = 1.0 - abs(beats_a - beats_b) / max(1, beats_a + beats_b)
    return 0.85 * sound + 0.15 * meter


if __name__ == "__main__":
    pairs = [
        ("Humpty Dumpty", "un petit un petit"),
        ("it for", "est fort"),
        ("to tell", "t elle"),
        # same onset, fudged TAILS (unstressed) -- should stay high:
        ("happy", "happi"), ("happy", "happo"),
        # fudged STRESSED vowel -- should drop more:
        ("happy", "hoppy"),
    ]
    print(f"{'EN':16s} {'FR':16s} {'prosodic':>9s} {'plain combo':>12s}")
    for en, fr in pairs:
        ps = prosodic_score(en, fr)
        cb = matcher.homophone_score(en, "en", fr, "fr")["score"]
        print(f"{en:16s} {fr:16s} {ps:9.2f} {cb:12.2f}")
