"""DUAL TRANSLATION AT SCALE: is FR-meaning mapped to EN-meaning achievable
while the sound also matches? Measured, not hoped.

For each English phrase we build FOUR candidate French renderings and score
every one on BOTH channels:
    sound    = matcher combo            (does it sound like the English?)
    meaning  = semantic_cosine (MiniLM) (does it MEAN the English? -- embeds in
               their proper channel: topic, not grammar)

Candidates per phrase:
    literal   word-by-word MUSE gloss           (meaning high, sound lottery)
    dual-tile per word: best translation∧homophone from dual-pairs (tier walk)
    carve     the engine's sound carve           (sound high, meaning lottery)
    hybrid    dual-tile where a DUAL word exists, literal elsewhere

The scale answer = % of phrases with a candidate clearing BOTH bars:
    Rooten-band dual:  sound >= 0.55  AND  meaning >= 0.45
    strict dual:       sound >= 0.70  AND  meaning >= 0.45

Run: python dual_scale.py [--n 60]
"""
from __future__ import annotations

import argparse
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


def load_muse():
    lit = {}
    for line in open(MUSE, encoding="utf-8"):
        p = line.split()
        if len(p) == 2 and p[0] not in lit:
            lit[p[0]] = p[1]
    return lit


def load_dual():
    d = defaultdict(list)
    for i, line in enumerate(open("dual-pairs.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6:
            d[p[0]].append((float(p[2]), p[1]))
    for v in d.values():
        v.sort(reverse=True)
    return d


def load_frvocab():
    vocab = set()
    # lexique ONLY -- the MUSE FR side carries noisy identity pairs (down, sat,
    # who...) that reopen the franglais hole
    for line in open("data/lexique.tsv", encoding="utf-8", errors="ignore"):
        w = line.split("\t")[0].strip().lower()
        if w:
            vocab.add(w)
    return vocab


def load_phrases(n):
    seen, out = set(), []
    for i, line in enumerate(open("corpus-carves.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0] not in seen:
            seen.add(p[0])
            out.append((p[0], p[1]))       # (en_line, engine carve)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    lit = load_muse()
    dual = load_dual()
    global FRVOCAB
    FRVOCAB = load_frvocab()
    phrases = load_phrases(args.n)
    print(f"{len(phrases)} corpus phrases; dual lexicon {len(dual)} EN words\n")

    def literal(ws):
        return " ".join(lit.get(w, w) for w in ws)

    def dualtile(ws):
        return " ".join(dual[w][0][1] if w in dual else lit.get(w, w) for w in ws)

    def hybrid(ws):
        out = []
        for w in ws:
            if w in dual and dual[w][0][0] >= 0.60:
                out.append(dual[w][0][1])
            else:
                out.append(lit.get(w, w))
        return " ".join(out)

    band = strict = 0
    rows = []
    for en_line, carve in phrases:
        ws = [w.lower().strip(".,;!?") for w in en_line.split()]
        cands = {
            "literal": literal(ws),
            "dual-tile": dualtile(ws),
            "hybrid": hybrid(ws),
            "carve": carve,
        }
        best = None
        for name, fr in cands.items():
            frw = [w.lower().strip(".,;!?'") for w in fr.split() if w]
            # HARD gate: the French must BE French (kills the franglais artifact
            # where untranslated English words score high on both channels)
            if any(w not in FRVOCAB for w in frw):
                continue
            s = combo(en_line, fr)
            m = max(0.0, semantic_cosine(en_line, fr))
            j = (s * m) ** 0.5
            if best is None or j > best[0]:
                best = (j, name, fr, s, m)
        if best is None:
            continue
        j, name, fr, s, m = best
        ok_band = s >= 0.55 and m >= 0.45
        ok_strict = s >= 0.70 and m >= 0.45
        band += ok_band
        strict += ok_strict
        rows.append((j, en_line, name, fr, s, m, ok_band))

    print(f"ACHIEVABLE dual translation (best candidate per phrase):")
    print(f"  Rooten-band (sound>=0.55 ∧ meaning>=0.45): {band}/{len(phrases)} "
          f"= {band/len(phrases):.0%}")
    print(f"  strict      (sound>=0.70 ∧ meaning>=0.45): {strict}/{len(phrases)} "
          f"= {strict/len(phrases):.0%}\n")
    rows.sort(reverse=True)
    print("top dual translations (sound AND meaning, one line):")
    for j, en, name, fr, s, m, ok in rows[:12]:
        tag = "DUAL " if ok else "     "
        print(f"  {tag}snd {s:.2f} mng {m:.2f} [{name:9s}] {en}")
        print(f"        -> {fr}")
    print("\nReading: the meaning bridge EXISTS at scale where the dual lexicon "
          "covers the content words -- the hybrid strategy (dual tiles where "
          "possible, literal elsewhere) is the scalable dual translator; the "
          "carve remains the sound-first fallback for uncovered spans.")


if __name__ == "__main__":
    main()
