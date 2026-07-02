"""Trigram stupid-backoff LM -- the L2 upgrade over bigram_lm.

Trained on OpenSubtitles French (colloquial, verse-adjacent register) and/or the
same Gutenberg English the bigram used. Same API as bigram_lm (cond, fluency,
seq_logprob) so every composer can swap it in; trigram context is what lets it
prefer a grammatical continuation over mere adjacency.

Build:  python trigram_lm.py build fr /tmp/fr_sub.txt
Use:    import trigram_lm; FR = trigram_lm.load("fr")
"""
from __future__ import annotations

import math
import pickle
import re
import sys
from collections import Counter

TOK = re.compile(r"[a-zà-ÿ']+")
ALPHA = 0.4


def toks(line):
    return TOK.findall(line.lower())


class TrigramLM:
    def __init__(self, lang):
        self.lang = lang
        self.uni = Counter()
        self.bi = Counter()
        self.tri = Counter()
        self.total = 0

    def fit_file(self, path, max_lines=0):
        for i, line in enumerate(open(path, encoding="utf-8", errors="ignore")):
            if max_lines and i >= max_lines:
                break
            ws = ["<s>", "<s>"] + toks(line) + ["</s>"]
            for j in range(2, len(ws)):
                self.uni[ws[j]] += 1
                self.bi[(ws[j - 1], ws[j])] += 1
                self.tri[(ws[j - 2], ws[j - 1], ws[j])] += 1
                self.total += 1
        # prune singletons to keep the pickle sane
        self.tri = Counter({k: v for k, v in self.tri.items() if v > 1})
        self.bi = Counter({k: v for k, v in self.bi.items() if v > 1})
        return self

    def _p_uni(self, w):
        return (self.uni.get(w, 0) + 0.5) / (self.total + 0.5 * (len(self.uni) + 1))

    def cond2(self, a, b):
        cab = self.bi.get((a, b), 0)
        if cab:
            return cab / max(1, self.uni.get(a, 0))
        return ALPHA * self._p_uni(b)

    def cond(self, a, b, c=None):
        """P(c | a b) stupid-backoff; called as cond(prev, w) it is bigram-compat."""
        if c is None:
            return self.cond2(a, b)
        cabc = self.tri.get((a, b, c), 0)
        if cabc:
            return cabc / max(1, self.bi.get((a, b), 1))
        return ALPHA * self.cond2(b, c)

    def seq_logprob(self, words):
        ws = ["<s>", "<s>"] + [w.lower() for w in words] + ["</s>"]
        lp = 0.0
        for j in range(2, len(ws)):
            lp += math.log(self.cond(ws[j - 2], ws[j - 1], ws[j]) + 1e-12)
        return lp

    def fluency(self, words):
        if not words:
            return 0.0
        return math.exp(self.seq_logprob(words) / (len(words) + 1) / 6.0)

    def save(self):
        with open(f"trigram-lm-{self.lang}.pkl", "wb") as f:
            pickle.dump(self, f)


def load(lang):
    with open(f"trigram-lm-{lang}.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "build":
        lm = TrigramLM(sys.argv[2]).fit_file(sys.argv[3],
                                             int(sys.argv[4]) if len(sys.argv) > 4 else 0)
        lm.save()
        print(f"trigram-lm-{lm.lang}.pkl  uni={len(lm.uni)} bi={len(lm.bi)} "
              f"tri={len(lm.tri)} tokens={lm.total}")
    else:
        FR = load("fr")
        for s in ["laisse dette laisse messe", "blesse le chef blesse la soupe",
                  "mou cède la mousse", "donne un tel mit toux",
                  "le chat mange la souris", "souris la mange chat le"]:
            print(f"{FR.fluency(s.split()):.4f}  {s}")
