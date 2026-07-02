"""FINAL VERSE: end homophonic writing, sensible in both languages.

Strict recipe (fixes the tile-quality collapse the earlier composers hit):
  tiles   = phrase-bank units sound>=0.68  +  tier-ladder pairs sound>=0.85
            (DUAL/S/STRICT/GOLD/LOOP ranks only)
  compose = beam over 3-5 tiles; score = mean_sound x EN-bigram x FR-bigram,
            with a CONTENT gate (>=40% content words each side -- no glue soup)
  polish  = cross-word juncture lift on the French line (liaison/elision)

Run: python final_verse.py [-n 8] [--len 4]
"""
from __future__ import annotations

import argparse
import math
import random

import bigram_lm

def _load_fr():
    """Trigram L2 when built (trigram_lm.py build fr <corpus>), else bigram."""
    try:
        import trigram_lm
        return trigram_lm.load("fr")
    except Exception:
        return bigram_lm.load("fr")

STOP = set("the a an of to in on at for and or is are was be it he she we you "
           "they my his her its our your their this that not but so as by with "
           "de la le les un une des du et ou est sont en dans sur pour par que "
           "qui ne pas ce cette il elle nous vous ils elles mon ma mes son sa "
           "ses tout tous".split())


def content_ratio(words):
    w = [x for x in words if x]
    return sum(1 for x in w if x.lower() not in STOP) / max(1, len(w))


def good_en(phrase):
    from wordfreq import zipf_frequency
    ws = phrase.split()
    return all(len(w) >= 2 and zipf_frequency(w, "en") >= 3.0 for w in ws)


def load_tiles():
    tiles = []
    for i, line in enumerate(open("phrase-bank-balanced.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and float(p[2]) >= 0.68:
            tiles.append((p[0].lower(), p[1].lower(), float(p[2])))
    for i, line in enumerate(open("tier-ladder.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        # rank en fr ladder v5 v7 strict loops dual cognate sound meaning
        if len(p) >= 12 and p[10]:
            try:
                snd = float(p[10])
            except ValueError:
                continue
            if int(p[0]) <= 7 and snd >= 0.85 and p[1].isalpha():
                tiles.append((p[1].lower(), p[2].lower(), snd))
    seen, out = set(), []
    for en, fr, s in tiles:
        if en == fr:                     # identity tile: cognate cheat, drop
            continue
        if not good_en(en):              # EN side must be real, common words
            continue
        if any(len(w) < 2 for w in fr.split()):
            continue
        if (en, fr) not in seen:
            seen.add((en, fr))
            out.append((en, fr, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--len", dest="length", type=int, default=4)
    ap.add_argument("--beam", type=int, default=160)
    args = ap.parse_args()
    EN, FR = bigram_lm.load("en"), _load_fr()
    tiles = load_tiles()
    print(f"{len(tiles)} strict tiles (sound>=0.68 bank / >=0.85 ladder)\n")

    def en_last(t):
        return t[0].split()[-1]

    def fr_edge(t):
        return t[1].split()[0], t[1].split()[-1]

    # beam
    rng = random.Random(7)
    starts = sorted(tiles, key=lambda t: -t[2])[:220]
    beams = [([t], t[2]) for t in starts]
    for _ in range(args.length - 1):
        nxt = []
        for chain, sc in beams:
            last = chain[-1]
            used = {c[0] for c in chain}
            cands = rng.sample(tiles, min(500, len(tiles)))
            local = []
            for t in cands:
                if t[0] in used or t[1] == last[1]:
                    continue
                enc = EN.cond(en_last(last), t[0].split()[0])
                frc = FR.cond(fr_edge(last)[1], fr_edge(t)[0])
                step = t[2] * (enc + 0.03) * (frc + 0.03)
                local.append((sc + math.log(step + 1e-9), chain + [t]))
            local.sort(key=lambda x: -x[0])
            nxt.extend(local[:3])
        nxt.sort(key=lambda x: -x[0])
        beams = [(c, s) for s, c in nxt[:args.beam]]

    # final rank: sound x both fluencies x content gate
    ranked = []
    seen_en = set()
    for chain, _ in beams:
        enw = " ".join(t[0] for t in chain).split()
        frw = " ".join(t[1] for t in chain).split()
        if content_ratio(enw) < 0.4 or content_ratio(frw) < 0.34:
            continue
        key = tuple(enw)
        if key in seen_en:
            continue
        seen_en.add(key)
        snd = sum(t[2] for t in chain) / len(chain)
        score = snd * (EN.fluency(enw) ** 0.5) * (FR.fluency(frw) ** 0.5)
        ranked.append((score, snd, chain))
    ranked.sort(key=lambda x: -x[0])

    try:
        import juncture
        J = True
    except Exception:
        J = False
    for score, snd, chain in ranked[:args.n]:
        en_line = " ".join(t[0] for t in chain)
        fr_line = " ".join(t[1] for t in chain)
        jl = ""
        if J:
            try:
                js = juncture.best_juncture_score(en_line, fr_line)
                jl = f"  +juncture {js:.2f}"
            except Exception:
                pass
        print(f"EN reads : {en_line}")
        print(f"FR reads : {fr_line}")
        print(f"   sound {snd:.2f}{jl}   "
              + " | ".join(f"{t[0]}≈{t[1]}" for t in chain))
        print()


if __name__ == "__main__":
    main()
