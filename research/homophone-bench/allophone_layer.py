"""Allophone layer -- a SEPARATE enrichment that borrows the existing match-time
machinery (matcher._variants + EQUIV) and adds the English allophonic processes
that actually create cross-lingual homophony.

Rationale: REPRESENTATION.md says store BROAD phonemes, never allophones as data
-- allophony is accent-dependent and belongs at match time. This module honours
that: it expands a broad-IPA string into its allophonic REALIZATIONS on the fly
(borrowing _variants for the phonological rules already implemented) and adds the
classic English allophones that matter for homophony:

  - flapping:    /t/,/d/ -> [ɾ] between vowels  (butter ~ a French /d/ word)
  - l-darkening: coda /l/ -> [ɫ] (~[w]/[o])     (milk, full)
  - yod/glide and h-dropping are already in CHEAP_GAP.

A pair is scored as the MAX combo over the allophonic realizations of BOTH sides
-- the accent-independent way to let allophones help, as a separate layer on top
of the broad-phoneme matcher, not baked into the stored dictionary.

Run: python allophone_layer.py
"""
from __future__ import annotations

import re

import matcher
from matcher import _variants, _canonical, _ngram_channel, _feat_channel, g2p

VOWELS = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɑɒ"


def flap(ipa: str) -> str:
    # intervocalic t/d -> ɾ (English tap); also after vowel before syllabic r
    return re.sub(rf"(?<=[{VOWELS}])[td](?=[{VOWELS}ɚɹ])", "ɾ", ipa)


def darken_l(ipa: str) -> str:
    # coda l -> ɫ (then EQUIV/cheap-gap handle ɫ~w~o); apply at word end / before C
    out = re.sub(rf"l(?=[^{VOWELS}]|$)", "ɫ", ipa)
    return out


def allophones(ipa: str, lang: str = "en") -> list[str]:
    """Broad IPA -> set of allophonic realizations (borrowing _variants)."""
    base = set(_variants(ipa))           # existing phonological rule variants
    more = set()
    for v in base:
        more.add(v)
        if lang == "en":
            more.add(flap(v))
            more.add(darken_l(v))
            more.add(darken_l(flap(v)))
    return list(more)[:12]


def combo(qi: str, ci: str) -> float:
    return 0.5 * _ngram_channel(qi, ci) + 0.5 * _feat_channel(qi, ci)


def allophone_score(en_text: str, fr_text: str) -> tuple[float, float]:
    """(broad combo, best allophone-expanded combo)."""
    qi, ci = g2p(en_text, "en"), g2p(fr_text, "fr")
    broad = combo(qi, ci)
    best = broad
    for a in allophones(qi, "en"):
        for b in allophones(ci, "fr"):
            best = max(best, combo(_canonical(a), _canonical(b)))
    return broad, best


def main():
    print("Allophone layer (borrows _variants + flapping/l-darkening). "
          "broad combo vs allophone-expanded:\n")
    # pairs where English allophony plausibly helps (flapped t/d, dark l)
    tests = [("butter", "bader"), ("water", "wader"), ("little", "lidl"),
             ("city", "cidre"), ("matter", "madre"), ("metal", "medal"),
             ("bottle", "botte"), ("twenty", "tande"), ("shut up", "chat eau")]
    print(f"{'EN':10s} {'FR':10s} {'broad':>6s} {'+allo':>6s} {'gain':>6s}")
    print("-" * 42)
    gains = 0
    for en, fr in tests:
        try:
            b, a = allophone_score(en, fr)
        except Exception:
            continue
        g = a - b
        gains += 1 if g > 0.01 else 0
        flag = "  <-- allophone helps" if g > 0.01 else ""
        print(f"{en:10s} {fr:10s} {b:6.2f} {a:6.2f} {g:+6.2f}{flag}")
    print(f"\nallophone expansion improved {gains}/{len(tests)} (flapping/darkening "
          f"surfaced a closer realization).")
    print("""
Reading: this is a SEPARATE layer -- the dictionary stays broad-phoneme; allophony
is applied at score time by expanding both sides through the existing _variants
plus the English flap/darken rules. It catches homophony that only exists in the
realized speech (butter's [ɾ] sounding like /d/), without committing accent-
dependent allophones to storage. Plug allophone_score() in as an optional
re-ranker beside combo where flapping/darkening matter.""")


if __name__ == "__main__":
    main()
