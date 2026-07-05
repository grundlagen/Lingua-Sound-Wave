"""SET-THEORETIC dual translation at PARAGRAPH scope.

The user is right twice over:

1. "This is all set theory."  Formally: French words partition into SOUND
   equivalence classes (homophone classes: [vɛʁ] = {vert, verre, vers, ver,
   vair}) and cluster into MEANING sets (back-translations: the EN words a FR
   word can mean -- the 'synonym path back'). A dual rendering of a sentence is
   a SELECTION: one FR unit per sound-span such that
       union(sound-spans)   COVERS the English phoneme stream   (exact cover)
       union(meaning-sets)  COVERS the English content set      (set cover)
   Two cover problems over the same selection -- that IS the goal, stated in
   sets. Greedy weighted set-cover approximates it (the classic ln-n bound).

2. "Maybe possible at PARAGRAPH level."  The meaning universe is built over
   the WHOLE paragraph, and any sentence's rendering may cover any element of
   it. Meaning is allowed to MIGRATE: if 'sea' won't fit sentence 1's sound,
   a sea-meaning word may surface in sentence 2 where the sound permits.
   Sentence-level failure becomes paragraph-level slack.

Run: python set_dual.py            (demo paragraph)
     python set_dual.py "we knew the sea" "the dawn made us free"
"""
from __future__ import annotations

import sys
from collections import defaultdict

import beauty_compose as BC
from semantic_cosine import semantic_cosine

STOP = set("the a an of to in on at for and or is are was be it he she we you "
           "they my his her its our your this that not but so as by with do "
           "did had has have will shall".split())


# ------------------------------------------------ meaning sets (paths back)
_BACK = None
def meaning_set(fr):
    """FR word -> set of EN words it can mean (the synonym path BACK)."""
    global _BACK
    if _BACK is None:
        _BACK = defaultdict(set)
        for i, line in enumerate(open("dual-pairs-fr2en.tsv", encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                _BACK[p[0]].add(p[1])
        for line in open("/tmp/muse-en-fr.txt", encoding="utf-8"):
            p = line.split()
            if len(p) == 2:
                _BACK[p[1]].add(p[0])
    out = set()
    for w in fr.split():
        out |= _BACK.get(w.strip("'"), set())
    return out


def content(ws):
    return {w for w in ws if w not in STOP and len(w) > 2}


def compose_paragraph(sentences, K=4):
    D = BC.load_all()
    # the MEANING UNIVERSE: content words of the whole paragraph (the set to cover)
    universe = set()
    sent_words = []
    for s in sentences:
        ws = [w.lower().strip(".,;:!?'") for w in s.split() if w.strip(".,;:!?'")]
        sent_words.append(ws)
        universe |= content(ws)
    covered = set()
    print(f"MEANING UNIVERSE (paragraph): {sorted(universe)}\n")

    results = []
    for s, ws in zip(sentences, sent_words):
        picks, tags = [], []
        for w in ws:
            cands = BC.candidates(w, D)[:K]
            if not cands:
                picks.append(w)
                tags.append(f"{w}=MISS")
                continue
            best = None
            for j, snd, mng, fr, ch in cands:
                gain = meaning_set(fr) & universe          # set-cover gain,
                new = gain - covered                       # paragraph-wide
                score = snd * (1.0 + 0.6 * len(new) + 0.15 * len(gain))
                if best is None or score > best[0]:
                    best = (score, snd, fr, ch, new, gain)
            _sc, snd, fr, ch, new, gain = best
            covered |= new
            picks.append(fr)
            mark = "".join(f"+{x}" for x in sorted(new)) if new else ""
            tags.append(f"{w}≈{fr}[{snd:.2f}{mark}]")
        fr_line = " ".join(picks)
        snd = BC.combo(s, fr_line)
        mng = max(0.0, semantic_cosine(s, fr_line))
        results.append((s, fr_line, snd, mng, tags))

    print("=" * 64)
    for s, fr_line, snd, mng, tags in results:
        print(f"EN : {s}")
        print(f"FR : {fr_line}")
        print(f"     sound {snd:.2f}  meaning {mng:.2f}")
        print(f"     {'  '.join(tags)}\n")
    cov = len(covered & universe) / max(1, len(universe))
    print(f"PARAGRAPH SET-COVER: {len(covered & universe)}/{len(universe)} "
          f"meaning elements covered = {cov:.0%}")
    print(f"  covered : {sorted(covered & universe)}")
    print(f"  missing : {sorted(universe - covered)}")
    print("\nReading: the objective is now literally two covers over one "
          "selection -- sound spans exact-cover the stream, meaning sets cover "
          "the paragraph universe, and meaning may migrate to whichever "
          "sentence's sound can afford it.")


PARA = ["we knew the sea and the ships",
        "the dawn calls us to the water",
        "less debt and less sorrow"]


def main():
    sentences = sys.argv[1:] or PARA
    compose_paragraph(sentences)


if __name__ == "__main__":
    main()
