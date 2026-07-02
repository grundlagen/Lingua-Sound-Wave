"""E36: compose TO a rhyme -- helper that turns rhyme-index.tsv into couplet
ends. Given a theme word (or nothing), list rime families with BOTH French and
English members from the tier ladder, so a poet (human/LLM/composer) can end
two lines on the same sound in both languages.

Run: python rhyme_pick.py [theme]
"""
from __future__ import annotations

import sys
from collections import defaultdict


def main():
    theme = sys.argv[1] if len(sys.argv) > 1 else ""
    ladder_fr = defaultdict(list)      # fr word -> (rank, en)
    for i, line in enumerate(open("tier-ladder.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4 and int(p[0]) <= 6:
            ladder_fr[p[2]].append((int(p[0]), p[1]))

    fams = []
    for i, line in enumerate(open("rhyme-index.tsv", encoding="utf-8")):
        if i == 0:
            continue
        rime, fr_ws, en_ws = line.rstrip("\n").split("\t")
        fr_good = [(w, ladder_fr[w][0][1]) for w in fr_ws.split() if w in ladder_fr]
        if len(fr_good) >= 2:
            fams.append((rime, fr_good, en_ws.split()[:8]))

    fams.sort(key=lambda f: -len(f[1]))
    print(f"{len(fams)} rime families with >=2 ladder-grade French enders\n")
    shown = 0
    for rime, fr_good, en_ws in fams:
        if theme and theme not in en_ws and all(theme != en for _, en in fr_good):
            continue
        print(f"  -{rime}:  " + "  ".join(f"{fr}(≈{en})" for fr, en in fr_good[:6]))
        print(f"        EN enders: {' '.join(en_ws)}")
        shown += 1
        if shown >= (4 if theme else 8):
            break
    print("\nUse: end both lines of a couplet on the same family -- the rhyme "
          "then holds in BOTH languages at once.")


if __name__ == "__main__":
    main()
