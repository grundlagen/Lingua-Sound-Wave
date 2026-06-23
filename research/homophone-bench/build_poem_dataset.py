"""Test: how to ACHIEVE the homophonic-poem dataset, given the books are in
copyright.

Reality check first. Mots d'Heures: Gousses, Rames (van Rooten 1967) and
N'Heures Souris Rames (de Kay 1980) are IN COPYRIGHT -- their French renderings
cannot be bulk-collected or redistributed. What IS free:
  - the ENGLISH source rhymes (Mother Goose) are public domain;
  - a few attested lines quoted in scholarship (Lukes 2020, Wikipedia) -> the
    small fair-use GOLD anchor (gold-dual-readings.tsv);
  - anything the project GENERATES itself.

So the dataset strategy is NOT scraping. It is:
  (A) tiny fair-use gold of attested human lines  -> calibrates the scorer;
  (B) a large SELF-GENERATED set: render public-domain English rhymes into
      French homophonic candidates with our own decoder, score them on the
      dual-reading axes, and keep the good ones.  Copyright-clean, scalable.

This script tests (B) end to end, REUSING extant tools (phonetic_decoder +
soramimi direction + the LM-in-beam from this session + dual_reading_eval's
scoring) rather than reinventing them, and measures the human-vs-machine gap
against the one attested gold line.

Run: python build_poem_dataset.py
"""
from __future__ import annotations

import subprocess
import time

import matcher
import phonetic_decoder as pd
from lexicon_g2p import clean_ipa
import dual_reading_eval as dre

try:
    import bigram_lm
    LM = bigram_lm.load("fr")
except Exception:
    LM = None

# Public-domain Mother Goose SOURCE phrases (English originals, pre-1900),
# split into the short phrase chunks van Rooten actually worked at -- a whole
# line decodes as one over-long stream and the gate rejects it (tested).
# We generate the French ourselves; we do NOT reproduce any copyrighted rendering.
PD_SOURCE_CHUNKS = [
    "Humpty Dumpty", "sat on a wall", "Jack and Jill", "went up the hill",
    "twinkle twinkle", "little star", "black sheep", "any wool",
    "Miss Muffet", "curds and whey", "hickory dickory", "the mouse ran",
]
MINING_COVERAGE = 0.70   # looser than the 0.85 production gate: we mine, then rank


def render(text, root, lm, top_n=10):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", text],
                       capture_output=True, text=True, check=True)
    ipa = clean_ipa(r.stdout.strip())
    cands = pd.decode(ipa, root, top_n=top_n, max_words=8,
                      lm=lm, lm_weight=pd.LM_BEAM_WEIGHT if lm else 0.0)
    return [c for c in cands if c["coverage"] >= MINING_COVERAGE
            and c["expensive_deletions"] == 0]


def score_candidate(en, fr):
    snd = matcher.homophone_score(en, "en", fr, "fr")["score"]
    coh = dre.coherence(fr, "fr")
    coh_shuf = dre.shuffle_coherence(fr, "fr")
    return snd, coh, coh - coh_shuf


def main():
    pd.BEAM = 200
    root = pd.build_trie(min_zipf=2.2, lang="fr")
    print(f"LM-in-beam: {'on' if LM else 'OFF (zipf only)'}\n")

    rows = []
    attempts = 0
    t0 = time.time()
    for src in PD_SOURCE_CHUNKS:
        attempts += 1
        cands = render(src, root, LM, top_n=8)
        if not cands:
            print(f"EN: {src:18s} -> (no rendering cleared the mining gate)")
            continue
        scored = []
        for c in cands:
            snd, coh, margin = score_candidate(src, c["fr"])
            scored.append((snd * coh, snd, coh, margin, c["fr"]))
        scored.sort(reverse=True)
        best = scored[0]
        rows.append((src, best))
        dr, snd, coh, margin, fr = best
        print(f"EN: {src:18s} -> FR: {fr:26s} snd {snd:.2f} L2coh {coh:.2f} dual {dr:.2f}")

    # the attested human gold, scored the same way, as the ceiling reference
    g = dre.load_gold()[0]
    gsnd = matcher.homophone_score(g["source_text"], "en", g["target_text"], "fr")["score"]
    gcoh = dre.coherence(g["target_text"], "fr")
    print("=" * 70)
    print("human gold (van Rooten, fair-use single line) as the ceiling:")
    print(f"   EN: {g['source_text']}")
    print(f"   FR: {g['target_text']}")
    print(f"   snd {gsnd:.2f}  L2coh {gcoh:.2f}  dual {gsnd*gcoh:.2f}")

    print(f"\nyield: {len(rows)}/{attempts} PD source chunks rendered a kept "
          f"candidate ({100*len(rows)//max(1,attempts)}%) in {time.time()-t0:.0f}s")
    if rows:
        mach = sum(r[1][0] for r in rows) / len(rows)
        print(f"machine mean dual-reading score: {mach:.2f}   "
              f"human gold: {gsnd*gcoh:.2f}   gap {gsnd*gcoh - mach:+.2f}")
    print("""
Tested takeaway (honest): rendering a FIXED source has LOW yield -- a whole line
over-runs the gate, and even phrase chunks mostly fail, because fixing the exact
source phonemes AND demanding fluent French is the over-constraint wall at the
line level. So the dataset is NOT "auto-translate the rhymes". It is:
  (A) a tiny fair-use GOLD of attested human lines (calibration only);
  (B) CONTENT-SELECTED generation -- let the system mine where the languages
      DO align (fragment_weave finds new dual-readings; soramimi renders only
      the source fragments that clear the gate) rather than forcing every line.
That matches impetus III: choose what to say where the languages are in tune,
do not translate a fixed text. The copyrighted books are the target to MATCH,
never to ingest.""")


if __name__ == "__main__":
    main()
