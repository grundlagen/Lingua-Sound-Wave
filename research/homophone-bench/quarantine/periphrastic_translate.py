"""Periphrastic chain translation: when a word won't transfer, DESCRIBE it.

The user's architecture, completed: greatest transfer comes from chain
complexity, and the deepest chains are not synonyms but PERIPHRASES — a
word swapped for a short poetic description that itself decodes
homophonically. "moonlight" has no clean French sound-twin, but "soft
light of the night" is a longer sound-stream the decoder can carve French
from, while the embedding anchor guarantees it still MEANS moonlight.

Per content word, candidate meanings (richest transfer surface):
  1. the word itself
  2. synonyms          (Datamuse rel_syn, free API, cached)
  3. definitions       (Datamuse md=d glosses, trimmed to content words)
  4. poetic templates  (gentle periphrases: "soft <x>", "<x> of the night")
Each candidate (word OR phrase) is decoded to French by the homophonic
decoder; the winner maximises sound x anchor-to-original-meaning blend.
Word order within a rendering follows the English sound stream (inherent
to homophony) — form is polished by FR elision + capitalization.

Offline-first: api-cache.json persists every API call, so reruns need no
network. Usage: python periphrastic_translate.py "moonlight on the sea"
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np
from wordfreq import zipf_frequency

import phonetic_decoder as pd
from lexicon_g2p import clean_ipa

CACHE_PATH = "api-cache.json"
STOP = {"the", "a", "an", "is", "are", "was", "of", "and", "or", "in", "on",
        "at", "to", "it", "its", "be", "by", "with", "as", "that", "this"}
CONTENT_STOP = STOP | {"sometimes", "often", "usually", "especially", "etc",
                       "something", "someone", "which", "who", "what", "for",
                       "from", "into", "onto", "used", "able", "being"}

try:
    _cache = json.load(open(CACHE_PATH))
except FileNotFoundError:
    _cache = {}


def _api(url: str):
    if url in _cache:
        return _cache[url]
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read().decode())
    except Exception:
        data = []
    _cache[url] = data
    return data


def synonyms(word: str) -> list[str]:
    url = f"https://api.datamuse.com/words?rel_syn={urllib.parse.quote(word)}&max=8"
    return [w["word"] for w in _api(url)
            if w["word"].replace(" ", "").isalpha() and len(w["word"]) > 1]


def definitions(word: str) -> list[str]:
    url = f"https://api.datamuse.com/words?sp={urllib.parse.quote(word)}&md=d&max=1"
    rows = _api(url)
    if not rows or "defs" not in rows[0]:
        return []
    out = []
    for d in rows[0]["defs"][:2]:
        gloss = d.split("\t", 1)[-1]
        content = [w for w in gloss.lower().replace(".", "").replace(",", "").split()
                   if w.isalpha() and w not in CONTENT_STOP
                   and zipf_frequency(w, "en") >= 2.5]
        if 1 <= len(content) <= 4:
            out.append(" ".join(content))
    return out


POETIC = ["soft {w}", "{w} of the night", "pale {w}", "gentle {w}", "the {w} light"]


def candidates(word: str) -> list[tuple[str, str]]:
    """(surface_text, kind) — meanings that might transfer better than the word."""
    cands = [(word, "word")]
    for s in synonyms(word):
        cands.append((s, "synonym"))
    for d in definitions(word):
        cands.append((d, "description"))
    # poetic periphrases only for evocative/imageable nouns
    if zipf_frequency(word, "en") >= 2.0 and word not in STOP:
        for t in POETIC[:2]:
            cands.append((t.format(w=word), "poetic"))
    # dedupe preserving order
    seen, out = set(), []
    for txt, k in cands:
        if txt not in seen:
            seen.add(txt)
            out.append((txt, k))
    return out[:14]


def decode_best(text, root, model, anchor_vec):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", text],
                       capture_output=True, text=True, check=True)
    ipa = clean_ipa(r.stdout.strip())
    cs = [c for c in pd.decode(ipa, root, top_n=18, max_words=8)
          if c["coverage"] >= 0.78 and c["expensive_deletions"] == 0]
    if not cs:
        return None
    vecs = np.asarray(model.encode([c["fr"] for c in cs],
                                   normalize_embeddings=True,
                                   show_progress_bar=False))
    best = None
    for c, v in zip(cs, vecs):
        sem = max(0.0, float(np.dot(anchor_vec, v)))
        blend = c["similarity"] * (0.45 + 0.55 * sem)
        if best is None or blend > best["blend"]:
            best = {"fr": c["fr"], "snd": c["similarity"], "sem": sem, "blend": blend}
    return best


def main():
    sents = sys.argv[1:] or ["moonlight on the sea", "the tiny bird sang"]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    root = pd.build_trie(min_zipf=2.2)

    out = []
    for sent in sents:
        words = [w.lower().strip(".,!?;:") for w in sent.split()]
        fr_line, details = [], []
        for w in words:
            if w in STOP:
                fr_line.append("·")
                continue
            anchor = model.encode([w], normalize_embeddings=True,
                                  show_progress_bar=False)[0]
            cands = candidates(w)
            non_poetic = [(t, k) for t, k in cands if k != "poetic"]
            poetic = [(t, k) for t, k in cands if k == "poetic"]
            best_overall = None
            for txt, kind in non_poetic:
                b = decode_best(txt, root, model, anchor)
                if b and (best_overall is None or b["blend"] > best_overall["blend"]):
                    best_overall = {**b, "via": txt, "kind": kind}
            # periphrasis only rescues words that did not transfer well
            if best_overall is None or best_overall["sem"] < 0.5:
                for txt, kind in poetic:
                    b = decode_best(txt, root, model, anchor)
                    if b and (best_overall is None or b["blend"] > best_overall["blend"]):
                        best_overall = {**b, "via": txt, "kind": kind}
            if best_overall is None:
                fr_line.append(f"[{w}]")
                details.append(f"    {w}: no transfer")
                continue
            fr_line.append(best_overall["fr"])
            via = "" if best_overall["kind"] == "word" else f"  via {best_overall['kind']} '{best_overall['via']}'"
            details.append(f"    {w} -> {best_overall['fr']}"
                           f"   (snd {best_overall['snd']:.2f}, sem {best_overall['sem']:.2f}){via}")
        out.append(f"\nEN: {sent}")
        out.append(f"FR: {' '.join(fr_line)}")
        out.extend(details)

    json.dump(_cache, open(CACHE_PATH, "w"))
    text = "\n".join(out)
    print(text)
    with open("periphrastic-demo.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
