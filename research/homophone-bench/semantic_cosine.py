"""Semantic-cosine MEANING term -- a METHOD, deliberately NOT wired into the judge.

JOKER (combined phonetic-semantic embeddings) and the Nemotron council both point
at the same meaning signal: cosine similarity between the English SOURCE and the
French OUTPUT under the multilingual encoder -- i.e. does the carved French still
mean roughly what the English said? This module shows that signal as an available
method; `reward.py` / `prosody.py` / `fr_coherence.py` are UNCHANGED, so the
judging stays exactly as it is. Wire this in only by a deliberate, reviewed edit.

Run: python semantic_cosine.py
"""
from __future__ import annotations

import numpy as np

_M = None


def _model():
    global _M
    if _M is None:
        from sentence_transformers import SentenceTransformer
        _M = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _M


def semantic_cosine(en: str, fr: str) -> float:
    """cosine(embed(EN source), embed(FR output)) in [0,1]-ish; higher = closer
    in meaning. A METHOD for inspection -- not part of the live reward."""
    v = _model().encode([en, fr], normalize_embeddings=True)
    return float(v[0] @ v[1])


if __name__ == "__main__":
    cases = [
        ("the sea", "la mer"),                 # true translation -> high
        ("friend", "ami"),                     # true translation -> high
        ("the door", "un voile d or"),         # sounds alike, different meaning
        ("Humpty Dumpty", "un petit un petit"),
        ("the cat", "le chat"),
    ]
    print("semantic-cosine (METHOD only; judge unchanged):")
    for en, fr in cases:
        print(f"  {en:16s} ~ {fr:18s} {semantic_cosine(en, fr):+.2f}")
