"""Paraphrase-search dual translation: unfreeze the English side.

The measured ceiling of sound-first decoding (~0.49 sentence semantics)
comes from fixing the English text: half the degrees of freedom are frozen.
This searches EN paraphrases too — each content word may be swapped for a
synonym — and decodes EVERY variant, keeping the (paraphrase, rendering)
pair with the best sound x meaning blend against the ORIGINAL sentence's
literal translation.

Offline synonym source: MUSE pivot (en -> fr -> en), zipf-filtered. This is
the deterministic stand-in for LLM Job 3 (an LLM proposes better, fluent
paraphrases; the architecture is identical — proposer swaps meaning-
preserving words, decoder owns all phonetics).

Usage: python paraphrase_translate.py "the sea is cold" ...
Output: paraphrase-demo.txt
"""
from __future__ import annotations

import subprocess
import sys
from collections import defaultdict

import numpy as np
from wordfreq import zipf_frequency

import phonetic_decoder as pd
from lexicon_g2p import clean_ipa

STOP = {"the", "a", "an", "is", "are", "was", "of", "and", "or", "in", "on",
        "at", "to", "it", "its", "me", "my"}
MAX_VARIANTS = 24


def synonyms(word, en2fr, fr2en):
    out = set()
    for fr in en2fr.get(word, ()):
        for en in fr2en.get(fr, ()):
            if (en != word and en.isalpha() and len(en) > 1
                    and zipf_frequency(en, "en") >= 3.3):
                out.add(en)
    return sorted(out, key=lambda w: -zipf_frequency(w, "en"))[:4]


def decode_best(sentence, root, model, lit_vec):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", sentence],
                       capture_output=True, text=True, check=True)
    ipa = clean_ipa(r.stdout.strip())
    cands = [c for c in pd.decode(ipa, root, top_n=25, max_words=10)
             if c["coverage"] >= 0.8 and c["expensive_deletions"] == 0]
    if not cands:
        return None
    vecs = np.asarray(model.encode([c["fr"] for c in cands],
                                   normalize_embeddings=True,
                                   show_progress_bar=False))
    best = None
    for c, v in zip(cands, vecs):
        sem = float(np.dot(lit_vec, v))
        blend = c["similarity"] * (0.5 + sem / 2)
        if best is None or blend > best["blend"]:
            best = {"fr": c["fr"], "snd": c["similarity"], "sem": sem,
                    "blend": blend}
    return best


def main():
    sentences = sys.argv[1:] or [
        "the sea is cold",
        "two men under the moon",
        "she said tell me more",
    ]
    en2fr, fr2en = defaultdict(set), defaultdict(set)
    with open("/tmp/muse-en-fr.txt", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                en2fr[p[0]].add(p[1])
                fr2en[p[1]].add(p[0])

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    root = pd.build_trie(min_zipf=2.2)

    out = []
    for sent in sentences:
        words = [w.lower().strip(".,!?") for w in sent.split()]
        lit = " ".join(sorted(en2fr[w])[0] for w in words if en2fr.get(w))
        lit_vec = model.encode([lit], normalize_embeddings=True,
                               show_progress_bar=False)[0]

        variants = [sent]
        for i, w in enumerate(words):
            if w in STOP:
                continue
            for s in synonyms(w, en2fr, fr2en):
                v = words.copy()
                v[i] = s
                variants.append(" ".join(v))
                if len(variants) >= MAX_VARIANTS:
                    break
            if len(variants) >= MAX_VARIANTS:
                break

        results = []
        for v in variants:
            b = decode_best(v, root, model, lit_vec)
            if b:
                results.append((b["blend"], v, b))
        if not results:
            continue
        results.sort(key=lambda r: -r[0])
        base = next((r for r in results if r[1] == sent), results[0])
        _bl, bv, bb = results[0]

        out.append(f"\nEN original:  {sent}")
        out.append(f"literal:      {lit}")
        out.append(f"baseline:     {base[2]['fr']}"
                   f"   (snd {base[2]['snd']:.2f}, sem {base[2]['sem']:.2f})")
        out.append(f"best variant: {bv!r}")
        out.append(f"rendering:    {bb['fr']}"
                   f"   (snd {bb['snd']:.2f}, sem {bb['sem']:.2f})")
        gain = bb["sem"] - base[2]["sem"]
        out.append(f"semantic gain from paraphrase search: {gain:+.2f}")

    text = "\n".join(out)
    print(text)
    with open("paraphrase-demo.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
