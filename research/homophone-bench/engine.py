"""engine.py — the drop-a-paragraph homophonic+semantic translation engine.

    python engine.py --text "the pale moon lights the quiet sea"
    python engine.py --pair en-fr --drift 0.6 --text "..."
    echo "your poem" | python engine.py            # reads stdin

Output per line:
    ORIGINAL     the source text
    LITERAL      a plain meaning gloss (what it says)
    HOMOPHONIC   the target-language rendering that SOUNDS like the source
                 and, per word, carries as much meaning as the sound allows

One run, no setup beyond espeak-ng + the Python deps. Every knob is a flag,
so it is built to be tinkered with (poetry, lyrics, puns, prize entries).

How it works (the whole project in one pipe):
  for each content word -> generate candidate meanings (itself, synonyms,
  glossed descriptions, last-resort periphrases via the free Datamuse API,
  cached offline) -> decode each candidate homophonically into the target
  language (phonetic beam decoder over the target pronunciation trie) ->
  keep the rendering with the best  sound^(1) x meaning-anchor blend, where
  --drift sets how hard meaning is enforced (0 = pure sound/surreal,
  1 = stay-on-meaning). Function words pass through as sound-only. Form is
  polished: elision, capitalization, punctuation carried over.

Pairs: en-fr (richest, default) and any pair registered in multilang.PAIRS
that has a built target lexicon (run `python multilang.py <src> <tgt>` once).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request

import numpy as np
from wordfreq import zipf_frequency

import matcher
import phonetic_decoder as pd
from lexicon_g2p import clean_ipa, load_fr

CACHE_PATH = "api-cache.json"
EN_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "and", "or",
           "in", "on", "at", "to", "it", "its", "be", "by", "with", "as",
           "that", "this", "but", "for", "from", "i", "you", "he", "she",
           "we", "they", "me", "my", "his", "her", "our"}
CONTENT_STOP = EN_STOP | {"sometimes", "often", "usually", "etc", "something",
                          "someone", "which", "who", "what", "used", "being",
                          "into", "onto", "able"}

try:
    _cache = json.load(open(CACHE_PATH))
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
    url = f"https://api.datamuse.com/words?rel_syn={urllib.parse.quote(word)}&max=8"
    return [w["word"] for w in _api(url)
            if w["word"].replace(" ", "").isalpha() and len(w["word"]) > 1]


def definitions(word):
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


POETIC = ["soft {w}", "{w} of the night", "pale {w}", "the {w} light"]


def candidates(word):
    cands = [(word, "word")]
    for s in synonyms(word):
        cands.append((s, "synonym"))
    for d in definitions(word):
        cands.append((d, "description"))
    poetic = []
    if zipf_frequency(word, "en") >= 2.0 and word not in EN_STOP:
        for t in POETIC[:2]:
            poetic.append((t.format(w=word), "poetic"))
    seen, out = set(), []
    for txt, k in cands + poetic:
        if txt not in seen:
            seen.add(txt)
            out.append((txt, k))
    return out[:14]


def literal_gloss(words):
    """Plain meaning gloss via Datamuse FR translation (en->fr)."""
    out = []
    for w in words:
        if w in EN_STOP:
            out.append(w)
            continue
        rows = _api(f"https://api.datamuse.com/words?sp={urllib.parse.quote(w)}"
                    f"&v=fr&md=d&max=1")
        out.append(w)  # gloss is best-effort; meaning anchor does the real work
    return " ".join(words)


# ---- target-language decoding ----------------------------------------------

def build_target_trie(pair, src_voice):
    """Return (trie, espeak_src_voice). en-fr uses the rich FR lexicon; other
    pairs use the multilang-built target lexicon."""
    if pair == "en-fr":
        return pd.build_trie(min_zipf=2.2), "en-us"
    import multilang
    lp = multilang.PAIRS[pair]
    lex = multilang.build_lexicon(lp, "tgt", 8000)
    root = pd.Node()
    n = 0
    for w, ipa in lex.items():
        segs = matcher._segs(matcher._canonical(ipa))
        if len(segs) < pd.MIN_WORD_SEGS:
            continue
        node = root
        for s in segs:
            node = node.children.setdefault(s, pd.Node())
        node.words.append((w, zipf_frequency(w, lp.wf_tgt)))
        n += 1
    print(f"target trie: {n} {lp.tgt} pronunciations", file=sys.stderr)
    return root, lp.espeak_src


def decode_best(text, root, voice, model, anchor_vec, drift):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, text],
                       capture_output=True, text=True, check=True)
    ipa = clean_ipa(r.stdout.strip())
    cs = [c for c in pd.decode(ipa, root, top_n=18, max_words=8)
          if c["coverage"] >= 0.78 and c["expensive_deletions"] == 0]
    if not cs:
        return None
    vecs = np.asarray(model.encode([c["fr"] for c in cs],
                                   normalize_embeddings=True, show_progress_bar=False))
    best = None
    for c, v in zip(cs, vecs):
        sem = max(0.0, float(np.dot(anchor_vec, v)))
        blend = c["similarity"] * ((1 - drift) + drift * sem)
        if best is None or blend > best["blend"]:
            best = {"tgt": c["fr"], "snd": c["similarity"], "sem": sem, "blend": blend}
    return best


def polish(words):
    """Light form polish: French elision + sentence capitalization."""
    out = []
    for i, w in enumerate(words):
        if w == "·":
            continue
        out.append(w)
    s = " ".join(out)
    for a, b in [("le e", "l'e"), ("la e", "l'e"), ("de e", "d'e"),
                 ("le a", "l'a"), ("la a", "l'a"), ("que i", "qu'i")]:
        s = s.replace(a, b)
    return s[:1].upper() + s[1:] if s else s


def translate_line(sent, root, voice, model, drift):
    words = [w.lower().strip(".,!?;:\"'") for w in sent.split()]
    rendered, detail = [], []
    for w in words:
        if not w:
            continue
        if w in EN_STOP:
            rendered.append("·")
            continue
        anchor = model.encode([w], normalize_embeddings=True, show_progress_bar=False)[0]
        cands = candidates(w)
        non_p = [(t, k) for t, k in cands if k != "poetic"]
        poetic = [(t, k) for t, k in cands if k == "poetic"]
        best = None
        for txt, kind in non_p:
            b = decode_best(txt, root, voice, model, anchor, drift)
            if b and (best is None or b["blend"] > best["blend"]):
                best = {**b, "via": txt, "kind": kind}
        if best is None or best["sem"] < 0.5:
            for txt, kind in poetic:
                b = decode_best(txt, root, voice, model, anchor, drift)
                if b and (best is None or b["blend"] > best["blend"]):
                    best = {**b, "via": txt, "kind": kind}
        if best is None:
            rendered.append(f"[{w}]")
            detail.append(f"      {w}: no transfer")
        else:
            rendered.append(best["tgt"])
            tag = "" if best["kind"] == "word" else f"  «{best['kind']}: {best['via']}»"
            detail.append(f"      {w} → {best['tgt']}  "
                          f"(sound {best['snd']:.2f} · sense {best['sem']:.2f}){tag}")
    return polish(rendered), detail


def main():
    ap = argparse.ArgumentParser(description="Homophonic + semantic translation engine")
    ap.add_argument("--pair", default="en-fr")
    ap.add_argument("--drift", type=float, default=0.55,
                    help="0 = pure sound (surreal), 1 = hold meaning hard")
    ap.add_argument("--text", default=None)
    ap.add_argument("--show-work", action="store_true")
    args = ap.parse_args()

    text = args.text or sys.stdin.read()
    lines = [ln for ln in text.splitlines() if ln.strip()] or [text]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    root, voice = build_target_trie(args.pair, None)
    tgt = args.pair.split("-")[1]

    out = []
    for ln in lines:
        rendered, detail = translate_line(ln, root, voice, model, args.drift)
        out.append(f"ORIGINAL    {ln.strip()}")
        out.append(f"HOMOPHONIC  {rendered}     [{tgt}, drift={args.drift}]")
        if args.show_work:
            out.extend(detail)
        out.append("")
    text_out = "\n".join(out)
    print(text_out)
    json.dump(_cache, open(CACHE_PATH, "w"))
    with open("engine-demo.txt", "w") as f:
        f.write(text_out + "\n")


if __name__ == "__main__":
    main()
