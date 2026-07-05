"""chain_compose.py — translation built AROUND chain_translate, fed with real
synonyms (not context-drift), with poetic descriptions as the fallback.

The breakthrough is chain_translate's alternation-chain transfer (a word
reaches the target through interleaved sound/meaning hops). Its flaw: its
own ~sem edges are embedding-kNN context, which drifts (love -> aine). So
here chain_translate stays THE engine, but we control what it transfers:

  for each source word w:
    seeds = [w] + real synonyms(w)        (Datamuse rel_syn — trustworthy,
                                            not embedding context)
    for each seed: chain_translate.fr_endpoints(seed)   <- the breakthrough
    collect every (french_endpoint, chain, quality), then ANCHOR each to the
    MEANING OF w (multilingual embedding cos) and keep the best — the anchor
    rejects any chain that drifted off w's sense.
    if nothing clears the anchor bar, DESCRIBE w (poetic periphrasis) and
    transfer the description's head through chain_translate instead.

So: chain_translate does all the transferring; synonyms keep the meaning
moves honest; descriptions rescue the unmatchable; the anchor is the guard
against the context-drift you don't trust.

    python chain_compose.py --text "the sea is cold and the moon is bright"
    python chain_compose.py --llm --show-work --text "..."
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
from wordfreq import zipf_frequency

import chain_game
import chain_translate as ct

EN_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "and", "or",
           "in", "on", "at", "to", "it", "its", "be", "by", "with", "as",
           "that", "this", "but", "for", "from", "i", "you", "we", "they"}
CACHE = "api-cache.json"
try:
    _cache = json.load(open(CACHE))
except FileNotFoundError:
    _cache = {}


def _api(url):
    if url in _cache:
        return _cache[url]
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read().decode())
    except Exception:
        data = []
    _cache[url] = data
    return data


def synonyms(word):
    url = f"https://api.datamuse.com/words?rel_syn={urllib.parse.quote(word)}&max=6"
    return [w["word"] for w in _api(url) if w["word"].isalpha() and len(w["word"]) > 1]


def descriptions(word):
    rows = _api(f"https://api.datamuse.com/words?sp={urllib.parse.quote(word)}&md=d&max=1")
    out = []
    if rows and "defs" in rows[0]:
        for d in rows[0]["defs"][:2]:
            gloss = d.split("\t", 1)[-1]
            content = [x for x in gloss.lower().replace(".", "").replace(",", "").split()
                       if x.isalpha() and x not in EN_STOP and zipf_frequency(x, "en") >= 2.8]
            out += content[:2]   # head content words of the gloss, each a chain seed
    return out


def transfer(word, edges, model, anchor_bar=0.45):
    """chain_translate the word + its real synonyms; anchor to word's meaning."""
    wv = model.encode([word], normalize_embeddings=True, show_progress_bar=False)[0]
    seeds = [(word, "self")] + [(s, "synonym") for s in synonyms(word)]
    cands = []
    for seed, kind in seeds:
        for fw, (rank, q, hops, chain) in ct.fr_endpoints(edges, seed).items():
            if q >= 0.80:
                cands.append((fw, q, hops, chain, seed, kind))
    if not cands:
        # fallback: DESCRIBE the word, transfer the description heads
        for d, kind in [(x, "description") for x in descriptions(word)]:
            for fw, (rank, q, hops, chain) in ct.fr_endpoints(edges, d).items():
                if q >= 0.80:
                    cands.append((fw, q, hops, chain, d, kind))
    if not cands:
        return None
    # anchor: endpoint must still MEAN word (guards against context-drift)
    fr_words = list({c[0] for c in cands})
    fv = dict(zip(fr_words, model.encode(fr_words, normalize_embeddings=True,
                                         show_progress_bar=False)))
    best = None
    for fw, q, hops, chain, seed, kind in cands:
        anchor = max(0.0, float(np.dot(wv, fv[fw])))
        score = q * (0.4 + 0.6 * anchor)
        if anchor >= anchor_bar * 0.5 and (best is None or score > best["score"]):
            best = {"tgt": fw, "q": q, "anchor": anchor, "score": score,
                    "chain": chain, "seed": seed, "kind": kind}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--show-work", action="store_true")
    args = ap.parse_args()

    print("building chain graph (chain_translate's engine)...", file=sys.stderr)
    edges, _ = chain_game.build_graph()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    arranger = None
    if args.llm:
        import llm_layer
        arranger = llm_layer if llm_layer.available() else None
        if not arranger:
            print("  [--llm: no DEEPSEEK_API_KEY in env]")

    text = args.text or sys.stdin.read()
    lines = [ln for ln in text.splitlines() if ln.strip()] or [text]
    out = []
    for ln in lines:
        words = [w.lower().strip(".,!?;:\"'") for w in ln.split()]
        rendered, detail, llm_opts = [], [], []
        for w in words:
            if not w:
                continue
            if w in EN_STOP:
                rendered.append("·")
                continue
            b = transfer(w, edges, model)
            if not b:
                rendered.append(f"[{w}]")
                detail.append(f"      {w}: no chain transfer (even via synonyms/description)")
                continue
            rendered.append(b["tgt"])
            seed_note = "" if b["kind"] == "self" else f"  ⟵ {b['kind']} '{b['seed']}'"
            detail.append(f"      {w} → {b['tgt']}  (chain q{b['q']:.2f} · "
                          f"anchor {b['anchor']:.2f}){seed_note}")
            detail.append(f"          {b['chain']}")
            llm_opts.append({"src_word": w, "renderings": [{"tgt": b["tgt"],
                             "sound": min(1.0, b["q"])}]})
        line = " ".join(t for t in rendered if t != "·")
        out.append(f"ORIGINAL    {ln.strip()}")
        out.append(f"CHAIN       {line[:1].upper() + line[1:]}")
        if arranger and llm_opts:
            res = arranger.arrange_line(ln.strip(), "French", llm_opts)
            if res and res.get("line"):
                out.append(f"FLUENT      {res['line']}")
        if args.show_work:
            out.extend(detail)
        out.append("")
    json.dump(_cache, open(CACHE, "w"))
    txt = "\n".join(out)
    print(txt)
    open("chain-compose-demo.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
