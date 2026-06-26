"""Reward for self-learning homophonic carving.

reward(en, fr) = sound_match (combo, our matcher) * french_validity (real French
words) -- both LOCAL and free, so it can score thousands of samples per training
step. The LLM judge (fr_coherence) is kept for periodic *evaluation*, not the
per-sample reward (too slow/costly inside an RL loop).

Importable on a GPU box (needs the repo's matcher.py + espeak-ng + panphon +
wordfreq). Test on CPU: python reward.py
"""
from __future__ import annotations

import sys
import os

# allow importing the parent repo's matcher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matcher                                    # noqa: E402
from wordfreq import zipf_frequency               # noqa: E402
try:
    import prosody                                # stress-weighted sound judge
except Exception:
    prosody = None


def french_validity(fr: str) -> float:
    """fraction of output tokens that are real French words (zipf>=2)."""
    toks = [t for t in fr.replace("'", " ").split() if t.isalpha()]
    if not toks:
        return 0.0
    return sum(zipf_frequency(t, "fr") >= 2.0 for t in toks) / len(toks)


def sound_match(en: str, fr: str) -> float:
    """stress-weighted prosodic score if available (stressed syllables matter,
    unstressed cheaped out), else the plain combo."""
    if prosody is not None:
        try:
            return prosody.prosodic_score(en, fr)
        except Exception:
            pass
    try:
        return float(matcher.homophone_score(en, "en", fr, "fr")["score"])
    except Exception:
        return 0.0


def reward(en: str, fr: str, alpha: float = 0.6) -> float:
    """alpha weights sound vs french-validity. Geometric-ish blend so BOTH must
    be decent (a sound-true line of non-French words still scores low)."""
    s = sound_match(en, fr)
    v = french_validity(fr)
    return (s ** alpha) * (v ** (1 - alpha))


if __name__ == "__main__":
    cases = [
        ("Humpty Dumpty", "un petit un petit"),     # good homophone, real French
        ("Humpty Dumpty", "zzz qqq"),               # nonsense
        ("it for", "est fort"),                     # tight homophone, real French
        ("it for", "it for"),                       # English (not French)
        ("to tell", "t elle"),
    ]
    for en, fr in cases:
        print(f"  reward({en!r}, {fr!r}) = {reward(en, fr):.3f}"
              f"   (sound {sound_match(en, fr):.2f}, frVal {french_validity(fr):.2f})")
