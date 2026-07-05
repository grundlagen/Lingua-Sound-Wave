"""fragment_route.py — chunk tunnelling: transfer a word with NO whole-word
sound row by routing its IPA chunks through fragment_edges and recomposing
the French side. This is the subword transducer the lattice was missing —
the route that fills the gaps chain_translate / whole-word matching leave.

  en:word  -> segment en IPA into chunks
           -> for each chunk: fragment_edge -> fr chunk(s)   (fragments.tsv)
           -> recompose fr chunks in order -> candidate fr phrase
           -> matcher arbiter scores the recomposition vs the original sound

Usage: python fragment_route.py cold moon bird wolf
"""
from __future__ import annotations

import csv
import subprocess
import sys
from collections import defaultdict

import matcher
from lexicon_g2p import clean_ipa


def load_fragments(min_count=2):
    en2fr = defaultdict(list)
    with open("fragments.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if int(r["count"]) >= min_count:
                en2fr[r["en_chunk"]].append((r["fr_chunk"], int(r["count"])))
    for k in en2fr:
        en2fr[k].sort(key=lambda x: -x[1])
    return en2fr


def g2p(word):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", word],
                       capture_output=True, text=True, check=True)
    return matcher._canonical(clean_ipa(r.stdout.strip()))


def tile(ipa, en2fr):
    """Greedy longest-chunk cover of the IPA string; recompose FR chunks."""
    i, en_tiles, fr_tiles = 0, [], []
    chunks = sorted(en2fr, key=len, reverse=True)
    while i < len(ipa):
        hit = None
        for ch in chunks:
            if len(ch) >= 2 and ipa.startswith(ch, i):
                hit = ch
                break
        if hit:
            en_tiles.append(hit)
            fr_tiles.append(en2fr[hit][0][0])
            i += len(hit)
        else:
            i += 1   # uncovered phoneme
    return en_tiles, fr_tiles


def main():
    words = sys.argv[1:] or ["cold", "moon", "bird", "wolf", "storm"]
    en2fr = load_fragments()
    print(f"fragment inventory: {len(en2fr)} EN chunks with FR routes\n")
    for w in words:
        ipa = g2p(w)
        en_tiles, fr_tiles = tile(ipa, en2fr)
        if not fr_tiles:
            print(f"{w:8s} [{ipa}] -> no chunk cover")
            continue
        fr_ipa = "".join(fr_tiles)
        sound = matcher.nw_sim_ipa(ipa, fr_ipa)
        recipe = " + ".join(f"{e}->{f}" for e, f in zip(en_tiles, fr_tiles))
        print(f"{w:8s} [{ipa}]  -> recomposed FR sound /{fr_ipa}/  (sound {sound:.2f})")
        print(f"          chunks: {recipe}")


if __name__ == "__main__":
    main()
