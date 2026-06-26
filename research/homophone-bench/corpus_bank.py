"""Carve the PUBLIC-DOMAIN nursery-rhyme corpus into French -- our own legal
homophonic gallery (the tradition's English is centuries old / PD; we generate
the French, so nothing copyrighted is used or reproduced).

Each line is carved sound-first (the +0.24-combo lever) with balanced selection
(combo x fluency). Output is a gallery of whole-line homophonic translations
(content-rich, thematic) AND it grows the composition material with real verse.

Output: corpus-carves.tsv  (en_line, fr_carve, combo, coverage, fluency)
Run: python corpus_bank.py
"""
from __future__ import annotations

import matcher
import phonetic_decoder as pd
import poetry_mode as pm
import whole_line_carve as wlc

# Traditional nursery rhymes -- public domain (pre-20th-century folk verse).
PD_LINES = [
    "the cat and the fiddle", "the cow jumped over the moon",
    "the little dog laughed to see such sport", "Mary had a little lamb",
    "its fleece was white as snow", "twinkle twinkle little star",
    "how I wonder what you are", "up above the world so high",
    "like a diamond in the sky", "Jack and Jill went up the hill",
    "to fetch a pail of water", "Jack fell down and broke his crown",
    "Humpty Dumpty sat on a wall", "Humpty Dumpty had a great fall",
    "Hickory dickory dock", "the mouse ran up the clock",
    "baa baa black sheep have you any wool", "yes sir yes sir three bags full",
    "little Bo peep has lost her sheep", "little Miss Muffet sat on a tuffet",
    "along came a spider", "old Mother Hubbard went to the cupboard",
    "to fetch her poor dog a bone", "rain rain go away",
    "come again another day", "row row row your boat",
    "gently down the stream", "London bridge is falling down",
    "a pocket full of posies", "this little piggy went to market",
    "Peter Peter pumpkin eater", "sing a song of sixpence",
    "a pocket full of rye", "four and twenty blackbirds",
    "baked in a pie", "Georgie Porgie pudding and pie",
    "kissed the girls and made them cry", "Tom Tom the pipers son",
    "stole a pig and away he run", "little Jack Horner sat in a corner",
    "eating a Christmas pie", "hey diddle diddle",
    "wee Willie Winkie runs through the town", "Mary Mary quite contrary",
    "how does your garden grow", "there was an old woman",
    "who lived in a shoe", "Jack Sprat could eat no fat",
    "pat a cake pat a cake bakers man", "one two buckle my shoe",
    "three four knock at the door", "five six pick up sticks",
    "the north wind doth blow", "and we shall have snow",
    "ride a cock horse to Banbury cross", "to see a fine lady",
    "upon a white horse", "see saw Margery Daw",
    "Jack shall have a new master", "ladybird ladybird fly away home",
]

COMBO_MIN = 0.42
COV_MIN = 0.75


def main():
    wlc.force_coverage()
    root = pm.build_poetry_trie(min_zipf=2.0)
    pd.BEAM = 600
    pd.WORD_PENALTY = 0.04
    pd.MIN_WORD_SEGS = 1

    out = open("corpus-carves.tsv", "w", encoding="utf-8")
    out.write("en_line\tfr_carve\tcombo\tcoverage\tfluency\n")
    kept = 0
    for ln in PD_LINES:
        try:
            ipa = wlc.en_ipa(ln)
        except Exception:
            continue
        nw = len(ln.split())
        cands = pd.decode(ipa, root, top_n=80, max_words=nw + 3,
                          lm=wlc.LM, lm_weight=0.10)
        best, key = None, -1.0
        for c in cands:
            if c["coverage"] < COV_MIN:
                continue
            combo = matcher.homophone_score(ln, "en", c["fr"], "fr")["score"]
            if combo < COMBO_MIN:
                continue
            flu = wlc.coherence(c["fr"])
            k = combo * (flu + 0.3)
            if k > key:
                key, best = k, (combo, c["coverage"], flu, c["fr"])
        if best:
            out.write(f"{ln}\t{best[3]}\t{best[0]:.3f}\t{best[1]:.2f}\t{best[2]:.3f}\n")
            out.flush(); kept += 1
            print(f"  [{best[0]:.2f}/{best[2]:.2f}] {ln}  ->  {best[3]}", flush=True)
    out.close()
    print(f"\n{kept}/{len(PD_LINES)} PD rhyme lines carved -> corpus-carves.tsv")


if __name__ == "__main__":
    main()
