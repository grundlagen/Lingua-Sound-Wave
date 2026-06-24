"""Whole-line coverage-aware carve: one French re-cut across an ENTIRE line,
not per-English-window fragments.

The fix to generation_engine's content_select: do NOT chop at English word
boundaries. Decode the line's WHOLE phoneme stream in one pass through the v6
poetry trie (1-segment fillers admitted), let the French boundaries fall wherever
French words land, and rank candidates by the coverage-aware matcher combo x
bigram coherence (the arbiter, not the decoder's coverage-blind similarity).

This is where the fillers + coherence of v6 do their real work: a single carve
that spans "Humpty Dumpty" as a unit (the way the art does), inserting little
words to carry the meter.

Public-domain source only (The Real Mother Goose, 1916). French is generated.

Run: python whole_line_carve.py
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


def en_ipa(text):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", text],
                       capture_output=True, text=True, check=True)
    return clean_ipa(r.stdout.strip())


def coherence(fr):
    toks = [t.lower() for t in fr.replace("'", " ").split() if t]
    return LM.fluency(toks) if (LM and toks) else 0.0


def force_coverage():
    """Penalise deletion so the beam MUST cover the stream instead of skipping
    half of it. The /h/ we don't have in French stays cheap; everything else is
    raised, which is what surfaces full filler carves over short partial matches."""
    for k in list(matcher.CHEAP_GAP):
        if k != "h":
            matcher.CHEAP_GAP[k] = min(0.9, matcher.CHEAP_GAP[k] * 2.2)
    matcher.GAP = 0.85
    pd._sub.cache_clear()


def carve_line(line, root, beam=420):
    """Decode the whole line's stream; rank by coverage-aware combo x coherence."""
    pd.BEAM = beam
    ipa = en_ipa(line)
    nwords = len(line.split())
    cands = pd.decode(ipa, root, top_n=40,
                      max_words=nwords + 3,             # room for filler insertion
                      lm=LM, lm_weight=pd.LM_BEAM_WEIGHT if LM else 0.0)
    scored = []
    for c in cands:
        if c["coverage"] < 0.75:                        # whole-line: demand coverage
            continue
        combo = matcher.homophone_score(line, "en", c["fr"], "fr")["score"]  # arbiter
        coh = coherence(c["fr"])
        nf = sum(1 for w in c["fr"].split() if w in pm.FILLER)
        scored.append((combo * coh, combo, coh, c["coverage"], c["fr"],
                       c["words"], nf))
    scored.sort(reverse=True)
    return ipa, nwords, scored


def main():
    pd.MIN_WORD_SEGS = 1
    pd.WORD_PENALTY = 0.04
    force_coverage()                       # the fix: make the beam cover, not skip
    root = pm.build_poetry_trie(min_zipf=2.0)
    print(f"whole-line carve (v6 poetry trie + arbiter rank + coverage-forcing). "
          f"coherence: {'bigram-LM' if LM else 'none'}\n")

    LINES = [
        "Hickory dickory dock",
        "Jack and Jill",
        "Humpty Dumpty",
        "Humpty Dumpty sat on a wall",
    ]
    for line in LINES:
        ipa, nw, scored = carve_line(line, root)
        print(f"EN: {line!r}  [{ipa}]  ({nw} EN words)")
        if not scored:
            print("   (no carve)\n")
            continue
        for dual, combo, coh, cov, fr, nfr, nf in scored[:4]:
            tag = f"  +{nf} filler" if nf else ""
            print(f"   dual {dual:.2f}  combo {combo:.2f} coh {coh:.2f} cov {cov:.2f}"
                  f"  [{nw}EN->{nfr}FR]{tag}")
            print(f"       FR: {fr}")
        print()

    print("""Reading: one carve now spans the WHOLE line (boundaries fall on French
words, not English ones), fillers carry the meter, and ranking is the coverage-
aware arbiter. Quality is still bounded by the bigram coherence -- it scores
fluency, not poetry -- so the top carve is sound-true and French-shaped but not
yet sensical verse. Swap in a real L2 model and the same search yields the art.""")


if __name__ == "__main__":
    main()
