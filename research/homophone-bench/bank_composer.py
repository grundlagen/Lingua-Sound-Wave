"""bank_composer -- chain phrase-bank entries into a whole Van Rooten line.

Each bank entry is an EN phrase that sounds like a FR phrase (a homophone unit).
We beam-search a CHAIN of entries so that, read across the chain:

  * the ENGLISH side is a fluent English sentence (EN bigram across the seam),
  * the FRENCH side is fluent French (FR bigram across the seam),
  * each unit is a strong homophone (its stored combo),

so the French, spoken, reconstructs the English -- and BOTH sides read. Because
the entries already carry their own homophone, we never have to force two fixed
sentences to align (the webbing wall); we only have to make the SEAMS fluent.

Run: python bank_composer.py [n_units] [seed_word ...]
"""
from __future__ import annotations

import sys
from heapq import nlargest

import bigram_lm

BANK = "phrase-bank-balanced.tsv"


def load_bank():
    rows = []
    for i, line in enumerate(open(BANK, encoding="utf-8")):
        if i == 0:
            continue
        en, fr, combo, cov, flu = line.rstrip("\n").split("\t")
        rows.append({"en": en, "fr": fr, "combo": float(combo),
                     "flu": float(flu), "ew": en.split(), "fw": fr.split()})
    return rows


def main():
    import math
    n_units = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "-n" else 4
    seeds = [w.lower() for w in sys.argv[1:] if not w.lstrip("-").isdigit()
             and w != "-n"]
    EN, FR = bigram_lm.load("en"), bigram_lm.load("fr")
    bank = load_bank()

    # starts: seeded by a word, else the most fluent strong homophones
    if seeds:
        starts = [e for e in bank if e["ew"][0] in seeds] or bank
    else:
        starts = nlargest(30, bank, key=lambda e: e["combo"] * (e["flu"] + 0.2))
    beams = [(0.0, [e]) for e in starts[:30]]

    for _ in range(n_units - 1):
        nxt = []
        for sc, chain in beams:
            used = {id(x) for x in chain}
            pe, pf = chain[-1]["ew"][-1], chain[-1]["fw"][-1]
            scored = []
            for e in bank:
                if id(e) in used:
                    continue
                eb = EN.cond(pe.lower(), e["ew"][0].lower())
                fb = FR.cond(pf.lower(), e["fw"][0].lower())
                step = eb * fb * e["combo"] * (e["flu"] + 0.15)
                scored.append((sc + math.log(step + 1e-12), chain + [e]))
            nxt.extend(nlargest(4, scored, key=lambda x: x[0]))
        beams = nlargest(40, nxt, key=lambda x: x[0])

    # final rank: both-side fluency x mean homophone
    out = []
    for _, chain in beams:
        ew = [w for e in chain for w in e["ew"]]
        fw = [w for e in chain for w in e["fw"]]
        ef, ff = EN.fluency([w.lower() for w in ew]), FR.fluency([w.lower() for w in fw])
        combo = sum(e["combo"] for e in chain) / len(chain)
        out.append((ef * ff * combo, ef, ff, combo, chain))
    out.sort(key=lambda x: -x[0])

    seen, shown = set(), 0
    print(f'composing from {len(bank)} bank units'
          + (f' seeded by {seeds}' if seeds else '') + "\n")
    for score, ef, ff, combo, chain in out:
        en_line = " ".join(e["en"] for e in chain)
        fr_line = " ".join(e["fr"] for e in chain)
        if en_line in seen:
            continue
        seen.add(en_line)
        print(f"  EN reads : {en_line}")
        print(f"  FR sounds: {fr_line}")
        print(f"     homophone {combo:.2f}  EN-fluency {ef:.2f}  FR-fluency {ff:.2f}"
              f"   [{' | '.join(e['en']+'≈'+e['fr'] for e in chain)}]\n")
        shown += 1
        if shown >= 6:
            break


if __name__ == "__main__":
    main()
