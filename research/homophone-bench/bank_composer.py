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


def load_theme(word, bank):
    """attach an embedding theme vector to each entry; return theme vec."""
    import json
    import numpy as np
    vecs = np.load("node-vecs.npy")
    ids = json.load(open("node-ids.json"))
    idx = {n: i for i, n in enumerate(ids)}
    for e in bank:
        vs = [vecs[idx["en:" + w]] for w in e["ew"] if "en:" + w in idx]
        e["vec"] = np.mean(vs, axis=0) if vs else None
    return vecs[idx["en:" + word]] if "en:" + word in idx else None


def main():
    import math
    argv = sys.argv[1:]
    n_units = 4
    if "-n" in argv:
        i = argv.index("-n"); n_units = int(argv[i + 1]); del argv[i:i + 2]
    theme_word = None
    if "--theme" in argv:
        i = argv.index("--theme"); theme_word = argv[i + 1]; del argv[i:i + 2]
    use_llm = "--llm" in argv
    if use_llm:
        argv.remove("--llm")
    seeds = [w.lower() for w in argv]
    EN, FR = bigram_lm.load("en"), bigram_lm.load("fr")
    bank = load_bank()

    theme = load_theme(theme_word, bank) if theme_word else None

    def thsim(e):
        if theme is None or e.get("vec") is None:
            return 0.0
        import numpy as np
        return float(e["vec"] @ theme)

    # starts: seeded by a word / theme, else the most fluent strong homophones
    if seeds:
        starts = [e for e in bank if e["ew"][0] in seeds] or bank
    elif theme is not None:
        starts = nlargest(30, bank, key=lambda e: e["combo"] * (e["flu"] + 0.2)
                          * (thsim(e) + 0.4))
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
                step = eb * fb * e["combo"] * (e["flu"] + 0.15) * (thsim(e) + 0.4)
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

    # optional: re-rank the top compositions by a REAL LLM French-coherence
    # judge (OpenRouter/Nemotron/Anthropic), the L2-model upgrade. Bigram
    # fallback if no key. Only the top ~12 are scored -> a single batch call.
    if use_llm:
        from fr_coherence import FRCoherence
        scorer = FRCoherence()
        top = out[:12]
        fr_lines = [" ".join(e["fr"] for e in tup[4]) for tup in top]
        llm = scorer.batch(fr_lines)
        rescored = [(tup[3] * (L + 0.1), tup[1], L, tup[3], tup[4])
                    for tup, L in zip(top, llm)]
        rescored.sort(key=lambda x: -x[0])
        out = rescored + out[12:]
        prov = scorer.provider[0] if scorer.available() else "bigram-fallback"
        print(f"[LLM French-coherence re-rank via {prov}]\n")

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
