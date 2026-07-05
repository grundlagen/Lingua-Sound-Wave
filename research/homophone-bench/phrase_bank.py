"""Build a big PHRASE BANK of sensical homophone matches to compose from.

For a large set of real English phrases (the most frequent English bigrams +
public-domain Mother Goose lines), carve each into French SOUND-FIRST (the lever
that nearly doubled combo: lm_weight 0.10, then rank by the matcher), and keep the
ones that are a strong homophone with good coverage. Each kept row is a reusable
unit: EN phrase that sounds like a real French phrase.

Output: phrase-bank.tsv  (en_phrase, fr_phrase, combo, coverage, fluency)
        written incrementally so partial runs are usable.

Run: python phrase_bank.py  [n_phrases]
"""
from __future__ import annotations

import sys

import bigram_lm
import matcher
import phonetic_decoder as pd
import poetry_mode as pm
import whole_line_carve as wlc

# public-domain English (The Real Mother Goose, 1916) -- the canonical targets
MOTHER_GOOSE = [
    "Humpty Dumpty", "Humpty Dumpty sat on a wall", "Hickory dickory dock",
    "Jack and Jill", "Jack and Jill went up the hill", "little Jack Horner",
    "twinkle twinkle little star", "baa baa black sheep", "hey diddle diddle",
    "the cat and the fiddle", "rain rain go away", "Jack be nimble",
    "old mother Hubbard", "little Miss Muffet", "Mary had a little lamb",
    "rock a bye baby", "this little piggy", "pat a cake pat a cake",
]

COMBO_MIN = 0.45
COV_MIN = 0.78


def main():
    n_phrases = int(sys.argv[1]) if len(sys.argv) > 1 else 1100
    # selection: "combo" = best homophone; "balanced" = best combo*(fluency+0.3)
    # so the bank prefers carves that are ALSO real French (lifts fluency).
    mode = sys.argv[2] if len(sys.argv) > 2 else "combo"
    out_path = "phrase-bank-balanced.tsv" if mode == "balanced" else "phrase-bank.tsv"
    EN = bigram_lm.load("en")
    # candidate phrases: most frequent English bigrams (real collocations)
    phrases = list(MOTHER_GOOSE)
    seen = set(p.lower() for p in phrases)
    for (a, b), c in EN.bi.most_common(20000):
        if a.isalpha() and b.isalpha() and len(a) > 1 and len(b) > 1:
            p = f"{a} {b}"
            if p not in seen:
                seen.add(p); phrases.append(p)
        if len(phrases) >= n_phrases:
            break

    wlc.force_coverage()
    root = pm.build_poetry_trie(min_zipf=2.0)
    pd.BEAM = 600
    pd.WORD_PENALTY = 0.04
    pd.MIN_WORD_SEGS = 1

    out = open(out_path, "w", encoding="utf-8")
    out.write("en_phrase\tfr_phrase\tcombo\tcoverage\tfluency\n")
    kept = 0
    for i, ph in enumerate(phrases):
        try:
            ipa = wlc.en_ipa(ph)
        except Exception:
            continue
        nw = len(ph.split())
        cands = pd.decode(ipa, root, top_n=80, max_words=nw + 3,
                          lm=wlc.LM, lm_weight=0.10)
        best, best_key = None, -1.0
        for c in cands:
            if c["coverage"] < COV_MIN:
                continue
            combo = matcher.homophone_score(ph, "en", c["fr"], "fr")["score"]
            if combo < COMBO_MIN:
                continue
            flu = wlc.coherence(c["fr"])
            key = combo * (flu + 0.3) if mode == "balanced" else combo
            if key > best_key:
                best_key = key
                best = (combo, c["coverage"], flu, c["fr"])
        if best and best[0] >= COMBO_MIN:
            out.write(f"{ph}\t{best[3]}\t{best[0]:.3f}\t{best[1]:.2f}\t{best[2]:.3f}\n")
            out.flush()
            kept += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(phrases)} phrases, {kept} kept "
                  f"({kept/(i+1)*100:.0f}%)", flush=True)
    out.close()
    print(f"\nphrase bank: {kept}/{len(phrases)} phrases kept "
          f"(combo>={COMBO_MIN}, coverage>={COV_MIN}) -> phrase-bank.tsv")


if __name__ == "__main__":
    main()
