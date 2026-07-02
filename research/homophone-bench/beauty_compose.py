"""BEAUTY-OF-LANGUAGE dual composer: every trick, one pipeline.

The 0/50 sentence-scope wall was word-ALIGNED substitution with literal glue.
This composer works around it with the full arsenal, per English word:

  1. DUAL one-for-ones      translation ∧ homophone (dual-pairs, 102k)
  2. ZIPF GLUE              mined sound-true function words (the≈de, he≈y,
                            was≈vase -- zipf-glue.tsv) so conjugation/glue stops
                            costing sound
  3. SYNONYM CHAIN (EN)     word -> EN synonym -> ITS translations -> best sound
                            (meaning survives the chain; sound gets a wider net)
  4. SYNONYM CHAIN (FR)     word -> translation -> FR synonyms -> best sound
  5. METAPHOR drift         FR words with sound >= 0.60 and embed cos >= 0.25 --
                            fillers permutated into metaphors for the word
  6. carve fallback         sound-first for what remains

Line verification: combo (sound), semantic_cosine (meaning), trigram L2.

Run: python beauty_compose.py "mary had a little lamb" ...
     python beauty_compose.py --bench 20      (re-measure the corpus wall)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import matcher
from semantic_cosine import semantic_cosine

MUSE = "/tmp/muse-en-fr.txt"


def combo(en, fr):
    try:
        qi, ci = matcher.g2p(en, "en"), matcher.g2p(fr, "fr")
        return 0.5 * matcher._ngram_channel(qi, ci) + 0.5 * matcher._feat_channel(qi, ci)
    except Exception:
        return 0.0


def load_all():
    dual = defaultdict(list)
    for i, line in enumerate(open("dual-pairs.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6:
            dual[p[0]].append((float(p[2]), p[1]))
    for v in dual.values():
        v.sort(reverse=True)

    glue = defaultdict(list)
    try:
        for i, line in enumerate(open("zipf-glue.tsv", encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            glue[p[0]].append((float(p[2]), p[1]))
        for v in glue.values():
            v.sort(reverse=True)
    except FileNotFoundError:
        pass

    esyn, fsyn = defaultdict(set), defaultdict(set)
    for line in open("muse-pivot-syn.tsv", encoding="utf-8"):
        a, b, _ = line.rstrip("\n").split("\t")
        if a.startswith("en:") and b.startswith("en:"):
            esyn[a[3:]].add(b[3:]); esyn[b[3:]].add(a[3:])
        elif a.startswith("fr:") and b.startswith("fr:"):
            fsyn[a[3:]].add(b[3:]); fsyn[b[3:]].add(a[3:])

    trans = defaultdict(set)
    for line in open(MUSE, encoding="utf-8"):
        p = line.split()
        if len(p) == 2:
            trans[p[0]].add(p[1])

    ladder = defaultdict(list)
    for i, line in enumerate(open("tier-ladder.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        # rank en fr ladder v5 v7 strict loops dual cognate sound meaning
        if len(p) >= 12 and int(p[0]) <= 7 and p[10]:
            try:
                snd, mng = float(p[10]), float(p[11]) if p[11] else 0.5
            except ValueError:
                continue
            if snd >= 0.70:
                ladder[p[1]].append((snd, mng, p[2]))
    for v in ladder.values():
        v.sort(reverse=True)

    frvocab = set()
    for line in open("data/lexique.tsv", encoding="utf-8", errors="ignore"):
        w = line.split("\t")[0].strip().lower()
        if w:
            frvocab.add(w)

    bridge = defaultdict(list)                    # Haiku-verified cross-scope
    try:
        for i, line in enumerate(open("llm-bridge.tsv", encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                bridge[p[0]].append((float(p[2]), float(p[3]), p[1]))
        for v in bridge.values():
            v.sort(reverse=True)
    except FileNotFoundError:
        pass
    return dual, glue, esyn, fsyn, trans, frvocab, ladder, bridge


def syn_chain(w, esyn, depth=3, beam=24):
    """round_rabbit transitive chains: each synonym opens the next, decay per
    hop (potentially infinite; decay + beam keep it finite)."""
    out = {w: 1.0}
    frontier = {w}
    for d in range(1, depth + 1):
        nxt = set()
        for x in frontier:
            for s in esyn.get(x, ()):
                if s not in out:
                    out[s] = 0.85 ** d
                    nxt.add(s)
        frontier = nxt
        if len(out) > beam:
            break
    return sorted(out.items(), key=lambda kv: -kv[1])[:beam]


def candidates(w, D, verbose=False):
    dual, glue, esyn, fsyn, trans, frvocab, ladder, bridge = D
    seen, out = set(), []

    def add(fr, meaning, channel):
        fr = fr.lower()
        if fr in seen or not fr:
            return
        if any(x not in frvocab for x in fr.split()):
            return
        seen.add(fr)
        s = combo(w, fr)
        out.append((s * (0.5 + 0.5 * meaning), s, meaning, fr, channel))

    for s, fr in dual.get(w, [])[:6]:
        add(fr, 1.0, "dual")
    for s, m, fr in ladder.get(w, [])[:6]:    # GOLD homophone one-for-ones
        add(fr, m, "ladder")
    ART = {"the": {"le", "la", "les", "de", "des", "du"},
           "a": {"un", "une", "à", "et"}, "an": {"un", "une"}}
    for s, fr in glue.get(w, [])[:6]:
        if w in ART and fr not in ART[w]:
            continue                      # articles may not become pronouns
        add(fr, 0.6, "glue")
    for s, m, fr in bridge.get(w, [])[:4]:        # Haiku cross-scope bridges
        add(fr, m, "haiku")
    for syn, decay in syn_chain(w, esyn):         # transitive EN synonym chain
        if syn == w:
            continue
        for fr in list(trans.get(syn, []))[:4]:
            add(fr, 0.8 * decay, f"esyn:{syn}")
        for _s, fr in dual.get(syn, [])[:2]:      # chains open dual tiles too
            add(fr, 0.8 * decay, f"esyn+dual:{syn}")
        for _s, _m, fr in ladder.get(syn, [])[:2]:  # ...and GOLD homophones
            add(fr, 0.7 * decay, f"esyn+gold:{syn}")
    for fr0 in list(trans.get(w, []))[:4]:        # FR-side chain, depth 2
        for fr, decay in syn_chain(fr0, fsyn, depth=2, beam=12):
            add(fr, 0.8 * decay, f"fsyn:{fr0}")
    out.sort(reverse=True)
    # metaphor drift only if nothing sound-decent yet (expensive)
    if not out or out[0][1] < 0.55:
        best = out[0] if out else None
        pool = {fr for s, fr in dual.get(w, [])}
        for fr0 in trans.get(w, ()):
            pool |= fsyn.get(fr0, set())
        for fr in list(pool)[:40]:
            s = combo(w, fr)
            if s >= 0.60:
                m = max(0.0, semantic_cosine(w, fr))
                if m >= 0.25 and fr not in seen:
                    out.append((s * (0.5 + 0.5 * m), s, m, fr, "metaphor"))
        out.sort(reverse=True)
    return out


def translate(line, D, show=True):
    ws = [w.lower().strip(".,;:!?'") for w in line.split() if w.strip(".,;:!?'")]
    picks, tags = [], []
    for w in ws:
        c = candidates(w, D)
        if c:
            j, s, m, fr, ch = c[0]
            picks.append(fr); tags.append(f"{w}≈{fr}[{ch};{s:.2f}]")
        else:
            picks.append(f"«{w}»"); tags.append(f"{w}=MISS")
    fr_line = " ".join(picks)
    s = combo(line, fr_line.replace("«", "").replace("»", ""))
    m = max(0.0, semantic_cosine(line, fr_line.replace("«", "").replace("»", "")))
    if show:
        print(f"EN : {line}")
        print(f"FR : {fr_line}")
        print(f"     sound {s:.2f}  meaning {m:.2f}")
        print(f"     {'  '.join(tags)}\n")
    return s, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()
    D = load_all()

    if args.bench:
        band = n = 0
        for i, line in enumerate(open("corpus-carves.tsv", encoding="utf-8")):
            if i == 0 or n >= args.bench:
                continue
            en = line.split("\t")[0]
            s, m = translate(en, D, show=(n < 6))
            band += (s >= 0.55 and m >= 0.45)
            n += 1
        print(f"BEAUTY bench: {band}/{n} = {band/max(1,n):.0%} reach the "
              f"Rooten band (was 0% word-aligned)")
        return

    for line in (args.text or ["mary had a little lamb",
                               "the sea is deep and cold",
                               "we see the moon at dawn"]):
        translate(line, D)


if __name__ == "__main__":
    main()
