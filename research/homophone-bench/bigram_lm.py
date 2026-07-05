"""Word-bigram language model with stupid-backoff, for scoring phrase fluency
on both the English and French sides of the homophone weave.

Why: mean wordfreq-zipf says every common word is equally good in any order, so
"set could" and "could set" score the same and weird-but-common sequences win.
A bigram model knows "in the" >> "the in" and rewards grammatical adjacency, so
generated carriers read like phrases, not word salad.

Model: P(w_i | w_{i-1}) via stupid backoff
  count(a,b)/count(a)            if the bigram was seen
  alpha * P_unigram(b)          otherwise
with P_unigram backed by the corpus, falling back to wordfreq for OOV words.
Trained on public-domain running text (Project Gutenberg): EN = Pride and
Prejudice (1342), Sherlock (1661), Moby Dick (2701), Frankenstein (84);
FR = Le Comte de Monte-Cristo (1184), Candide (4650). Deterministic, no API.

Usage (build + cache):  python bigram_lm.py /tmp/corpus
Then:  from bigram_lm import load; lm = load("en"); lm.fluency(["in","the","sea"])
"""
from __future__ import annotations

import math
import os
import pickle
import re
import sys
from collections import Counter

from wordfreq import zipf_frequency

WORD = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæ']+", re.I)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bigram-lm-{}.pkl")


def _clean_gutenberg(text: str) -> str:
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", text, re.S)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\* ?END OF", text)
    if m:
        text = text[:m.start()]
    return text


def _tokens(text: str) -> list[str]:
    return [w.lower().strip("'") for w in WORD.findall(text) if len(w) > 0]


class BigramLM:
    def __init__(self, lang: str, alpha: float = 0.4):
        self.lang = lang
        self.alpha = alpha
        self.uni: Counter = Counter()
        self.ctx: Counter = Counter()      # count of w as a left context
        self.bi: Counter = Counter()
        self.N = 0

    def fit_files(self, paths: list[str]) -> "BigramLM":
        for p in paths:
            txt = _clean_gutenberg(open(p, encoding="utf-8", errors="ignore").read())
            toks = _tokens(txt)
            self.uni.update(toks)
            self.N += len(toks)
            for a, b in zip(toks, toks[1:]):
                self.bi[(a, b)] += 1
                self.ctx[a] += 1
        return self

    def _p_uni(self, w: str) -> float:
        c = self.uni.get(w, 0)
        if c:
            return c / self.N
        z = zipf_frequency(w, self.lang)        # OOV: back off to wordfreq
        return (10 ** z) / 1e9 if z > 0 else 1e-8

    def cond(self, a: str, b: str) -> float:
        if self.ctx.get(a) and self.bi.get((a, b)):
            return self.bi[(a, b)] / self.ctx[a]
        return self.alpha * self._p_uni(b)

    def seq_logprob(self, words: list[str]) -> float:
        """Per-word average log-probability (higher = more fluent)."""
        if not words:
            return -20.0
        lp = math.log(self._p_uni(words[0]))
        for a, b in zip(words, words[1:]):
            lp += math.log(self.cond(a, b))
        return lp / len(words)

    def fluency(self, words: list[str]) -> float:
        """Map per-word logprob to ~[0,1] (calibrated: good phrase ~-6,
        word-salad ~-13)."""
        lp = self.seq_logprob(words)
        return max(0.0, min(1.0, (lp + 15.0) / 11.0))

    def save(self):
        with open(CACHE.format(self.lang), "wb") as f:
            pickle.dump(self, f)


def load(lang: str) -> BigramLM:
    with open(CACHE.format(lang), "rb") as f:
        return pickle.load(f)


def build(corpus_dir: str):
    groups = {"en": sorted(_g(corpus_dir, "en")), "fr": sorted(_g(corpus_dir, "fr"))}
    for lang, paths in groups.items():
        lm = BigramLM(lang).fit_files(paths)
        lm.save()
        print(f"{lang}: {lm.N:,} tokens, {len(lm.uni):,} types, "
              f"{len(lm.bi):,} bigrams -> {CACHE.format(lang)}")


def _g(d: str, prefix: str) -> list[str]:
    return [os.path.join(d, f) for f in os.listdir(d) if f.startswith(prefix) and f.endswith(".txt")]


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/corpus")
