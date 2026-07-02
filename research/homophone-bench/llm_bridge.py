"""HAIKU BRIDGE: a cheap LLM reads both databases and proposes cross-scope
matches -- French word-or-PHRASE for an English word (and reverse) that sounds
alike AND echoes the meaning. Every suggestion is then VERIFIED symbolically
(combo sound + MiniLM meaning + Lexique gate); the LLM proposes, the judge
disposes -- same law as everywhere in this repo.

This is the channel the databases can't compute alone: recombination
("berceau de l'eau", "mille-feuille") -- word COMBINATIONS in one language
matching word(s) in the other.

Run: python llm_bridge.py --n 40         (mines llm-bridge.tsv)
     python llm_bridge.py --words "thunder,freedom,little"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request

import matcher
from semantic_cosine import semantic_cosine

try:
    import _load_env
    _load_env.load_keys()
except Exception:
    pass

MODEL = "claude-haiku-4-5-20251001"


def combo(en, fr):
    try:
        qi, ci = matcher.g2p(en, "en"), matcher.g2p(fr, "fr")
        return 0.5 * matcher._ngram_channel(qi, ci) + 0.5 * matcher._feat_channel(qi, ci)
    except Exception:
        return 0.0


def haiku(prompt, max_tokens=1200):
    key = os.environ["ANTHROPIC_API_KEY"]
    body = json.dumps({
        "model": MODEL, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=120))
    return out["content"][0]["text"]


PROMPTS = {
 "antonym": """For EACH English word below, propose up to 2 French ANTONYM-based
renderings: a French word/phrase meaning the OPPOSITE, that SOUNDS like the
English word -- usable as "pas/sans + it" in verse. Ironic opposites allowed.
Real French only, non-cognates preferred.
Reply STRICT JSON: {"word": [["french", "opposite-link"], ...], ...}
English words: %s""",
 "kenning": """For EACH English word below, invent up to 2 French KENNINGS --
two-word poetic mini-definitions (like eau taire for water) whose SOUND, read
aloud in French, imitates the English word. Real French words only.
Reply STRICT JSON: {"word": [["french kenning", "meaning-link"], ...], ...}
English words: %s""",
 "metonym": """For EACH English word below, propose up to 2 French METONYMS or
PART/ASSOCIATE words (crown->roi, sea->sel/maree) that SOUND like the English
word when read aloud in French. Real French words only.
Reply STRICT JSON: {"word": [["french", "association"], ...], ...}
English words: %s""",
}

PROMPT = """You are helping build an English<->French homophone-translation lexicon.
For EACH English word below, propose up to 3 French renderings that:
  (a) SOUND like the English word when read aloud by a French speaker
      (use every phonetic trick: elision l'/d'/qu', liaison, silent endings,
      th->d/f, h-dropping, nasal vowels, multi-word combinations are welcome);
  (b) relate to or echo the MEANING (translation, near-synonym, or metaphor).
Real French words only. Prefer non-cognates. It is fine if one is a PHRASE.

Reply STRICT JSON: {"word": [["french rendering", "1-word meaning link"], ...], ...}

English words: %s"""


def load_frvocab():
    v = set()
    for line in open("data/lexique.tsv", encoding="utf-8", errors="ignore"):
        w = line.split("\t")[0].strip().lower()
        if w:
            v.add(w)
    return v


def mine(words, frvocab, prompt=None):
    P = prompt or PROMPT
    rows = []
    for i in range(0, len(words), 20):
        chunk = words[i:i + 20]
        try:
            txt = haiku(P % ", ".join(chunk))
            txt = txt[txt.index("{"): txt.rindex("}") + 1]
            data = json.loads(txt)
        except Exception as e:
            print(f"  (batch skipped: {e})", file=sys.stderr)
            continue
        for en, sugg in data.items():
            en = en.lower().strip()
            for item in sugg[:3]:
                fr = (item[0] if isinstance(item, list) else str(item)).lower().strip(" .!?")
                fr_words = [w.strip("'") for w in fr.replace("'", "' ").split()]
                if not fr or any(w and w not in frvocab for w in fr_words):
                    continue
                s = combo(en, fr.replace("'", " "))
                m = max(0.0, semantic_cosine(en, fr))
                if s >= 0.45 and m >= 0.25 and s * m >= 0.20:
                    rows.append((en, fr, s, m))
                    print(f"  KEEP {en:14s} -> {fr:22s} snd {s:.2f} mng {m:.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--words", default="")
    ap.add_argument("--mode", default="", choices=["", "antonym", "kenning", "metonym"])
    args = ap.parse_args()
    frvocab = load_frvocab()

    if args.words:
        words = [w.strip() for w in args.words.split(",")]
    else:
        # words the symbolic channels found hard: corpus content words w/o a
        # dual/ladder hit
        from wordfreq import zipf_frequency
        pool = set()
        for i, line in enumerate(open("corpus-carves.tsv", encoding="utf-8")):
            if i == 0:
                continue
            for w in line.split("\t")[0].lower().split():
                w = w.strip(".,;!?'")
                if w.isalpha() and len(w) > 2 and zipf_frequency(w, "en") > 2.5:
                    pool.add(w)
        words = sorted(pool)[:args.n]
    print(f"asking Haiku for {len(words)} words...", file=sys.stderr)
    rows = mine(words, frvocab, PROMPTS.get(args.mode) if args.mode else None)

    mode = "a" if os.path.exists("llm-bridge.tsv") else "w"
    with open("llm-bridge.tsv", mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("en\tfr\tsound\tmeaning\n")
        for en, fr, s, m in rows:
            f.write(f"{en}\t{fr}\t{s:.3f}\t{m:.3f}\n")
    print(f"\nkept {len(rows)} verified LLM bridges -> llm-bridge.tsv")


if __name__ == "__main__":
    main()
