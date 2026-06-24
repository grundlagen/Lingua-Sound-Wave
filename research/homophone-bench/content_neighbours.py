"""Content selection via Round Rabbit neighbours: the lever for lines that don't
carve.

The failing nursery lines (Jack and Jill, Hickory dickory dock) are dense in
non-lexical words with NO French carve. No grain or penalty fixes that -- you must
change WHAT is said. This scores each word's carve-richness and, for the
carve-poor ones, pulls homophone-neighbours (meaning- and sound-adjacent words
that DO carve) from the integrated v5+v6 dictionary + this branch's mapping-web.
The output is a "content palette": the carve-rich words to build the line from.

Run: python content_neighbours.py "moon star night fell crown water hill" ...
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import matcher


def load_carves(path="dictionary-v6-integrated.json"):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        d = json.load(open("dictionary-v5.json", encoding="utf-8"))
    best = {}
    for e in d:
        if e.get("direction", "en_fr") != "en_fr":
            continue
        en = e["en"]
        sc = float(e.get("score", e.get("combo", 0)))
        if best.get(en, (0,))[0] < sc:
            best[en] = (sc, e["fr"])
    return best


def load_web():
    g = json.load(open("mapping-web.json", encoding="utf-8"))
    return g.get("sound", {}), g.get("meaning", {})


def neighbours(word, sound, meaning, best, want=5):
    """carve-rich words meaning- or sound-adjacent to `word`."""
    cand = set()
    for nb in meaning.get(f"en:{word}", []):
        cand.add(nb.split(":", 1)[-1])
    for nb in meaning.get(f"fr:{word}", []):
        cand.add(nb.split(":", 1)[-1])
    # sound neighbours: words sharing a French homophone with `word`
    for edge in sound.get(f"en:{word}", []):
        fr = edge[0]
        for back in sound.get(fr, []):
            cand.add(back[0].split(":", 1)[-1])
    out = []
    for w in cand:
        if w == word or w not in best:
            continue
        out.append((best[w][0], w, best[w][1]))
    out.sort(reverse=True)
    return out[:want]


def main():
    themes = sys.argv[1:] or ["moon star night fell crown water hill children"]
    best = load_carves()
    sound, meaning = load_web()
    print(f"carve table: {len(best)} words.  Round Rabbit neighbour content selection.\n")

    for theme in themes:
        words = theme.split()
        scored = [(best.get(w, (0, ""))[0], w, best.get(w, (0, ""))[1]) for w in words]
        scored.sort(reverse=True)
        print(f"=== theme: {theme!r} ===")
        print("  carve-RICH content (write the line from these):")
        for sc, w, fr in scored:
            if sc >= 0.78:
                print(f"     {w:10s} -> {fr:18s} (combo {sc:.2f})")
        print("  carve-POOR words -> Round Rabbit neighbour substitutions:")
        for sc, w, fr in scored:
            if sc < 0.78:
                nbrs = neighbours(w, sound, meaning, best)
                sug = ", ".join(f"{n}->{f} ({s:.2f})" for s, n, f in nbrs[:3])
                print(f"     {w:10s} (best {sc:.2f}) :: {sug or 'no carve-rich neighbour'}")
        print()
    print("""Reading: this is the content-selection front end (impetus III). Lines built
from carve-RICH words render cleanly; carve-POOR words get neighbour
substitutions that preserve the theme while making the phoneme string carve --
the only lever for the lines no carve strategy can handle directly.""")


if __name__ == "__main__":
    main()
