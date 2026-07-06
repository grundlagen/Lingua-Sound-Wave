"""Meaning-anchored dual diffusion — anneal BOTH rails toward a fixed meaning.

The idea: the meaning of the input sentence is pinned as an ANCHOR in a
space outside both languages (a multilingual embedding). Then neither the
English nor the French is sacred — an annealing loop mutates BOTH sides
(word swaps, homonym pivots, paraphrase substitutions, insertions,
deletions, poetic expansion) and is rewarded for descending toward:

    meaning(EN') ~ anchor   AND   meaning(FR') ~ anchor   AND   sound(EN', FR') ~ 1

Like diffusion: start from the composer's best guess (structured noise),
iteratively denoise toward the target manifold. The output EN need not be
the input EN — it must only MEAN the same. Length is free: extra words,
poetic paraphrase, one sentence may become three (growth moves are legal;
the meaning anchor is what stops drift).

Meaning space (pluggable):
  1. sentence-transformers multilingual MiniLM if installed+cached (local
     machines: `pip install sentence-transformers`, model
     paraphrase-multilingual-MiniLM-L12-v2) — the real thing.
  2. fallback: corpus meaning space — every word maps to its DUAL pivot
     cluster (cross-language translation neighborhoods from tier-ladder);
     a sentence is a bag of cluster ids; similarity is cosine over the bag.
     Weaker, but language-neutral and dependency-free.

Run:
  python diffusion_compose.py --bench-dir ../homophone-bench \
      "the ocean remembers every vessel that ever sailed" \
      [--iters 400] [--expand 1.6]
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from compose_lattice import SentenceComposer

W_MEANING = 1.0     # per rail
W_SOUND = 1.6
LEN_PENALTY = 0.02  # per word of EN/FR length mismatch


# ---------------------------------------------------------------------------
# meaning spaces
# ---------------------------------------------------------------------------

class STMeaning:
    """Multilingual sentence-embedding space (the real anchor)."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def encode(self, text: str):
        return self.model.encode([text])[0]

    def sim(self, text: str, anchor) -> float:
        import numpy as np
        v = self.encode(text)
        return float(v @ anchor / ((np.linalg.norm(v) * np.linalg.norm(anchor)) or 1))


class CorpusMeaning:
    """Fallback: DUAL pivot clusters as a shared EN/FR meaning vocabulary."""

    def __init__(self, ladder_path: Path):
        import csv
        # union-find over translation edges -> cross-language clusters
        parent: dict[str, str] = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        with open(ladder_path, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["ladder"] in {"DUAL-S", "DUAL-A", "DUAL-B"}:
                    en, fr = r["en"].strip(), r["fr"].strip()
                    if en and fr and en != fr:
                        union("e:" + en, "f:" + fr)
        self.cluster = {w: find(w) for w in parent}

    def _bag(self, text: str, lang: str) -> Counter:
        pref = "e:" if lang == "en" else "f:"
        bag: Counter = Counter()
        for w in text.lower().split():
            bag[self.cluster.get(pref + w, pref + w)] += 1
        return bag

    def sim_bags(self, a: Counter, b: Counter) -> float:
        dot = sum(a[k] * b[k] for k in a)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# the annealer
# ---------------------------------------------------------------------------

class DualDiffuser:
    def __init__(self, bench_dir: Path, pipeline_dir: Path, seed: int = 7):
        self.composer = SentenceComposer(bench_dir, pipeline_dir)
        self.matcher = self.composer.matcher
        self.rng = random.Random(seed)
        self._g2p_cache: dict[tuple[str, str], str] = {}

        try:
            self.meaning = STMeaning()
            self.mode = "minilm"
        except Exception:
            ladder = pipeline_dir / "tier-ladder-cycle3.tsv"
            if not ladder.exists():
                ladder = bench_dir / "tier-ladder.tsv"
            self.meaning = CorpusMeaning(ladder)
            self.mode = "corpus"
        print(f"meaning space: {self.mode}")

        # move inventories
        self.en_mates = defaultdict(list)   # paraphrase mates (EN)
        pp = pipeline_dir / "out-paraphrase" / "paraphrase-en.tsv"
        if pp.exists():
            import csv
            with open(pp, encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    self.en_mates[r["word"]].append(r["mate"])
        # FR homonym classes (free sound-preserving meaning pivots)
        self.fr_homs = defaultdict(list)
        hc = bench_dir / "fr-homophone-classes-lexique.tsv"
        if hc.exists():
            import csv
            with open(hc, encoding="utf-8") as f:
                rd = csv.reader(f, delimiter="\t")
                next(rd, None)
                for row in rd:
                    if len(row) >= 2:
                        ms = row[1].split()
                        for m in ms:
                            self.fr_homs[m] = [x for x in ms if x != m]
        # poetic filler pools (glue words + common poetic lexicon)
        self.en_fillers = ["oh", "so", "still", "soft", "old", "all", "then"]
        self.fr_fillers = ["ô", "si", "or", "tout", "doux", "vieux", "alors"]

    # ---- energy -------------------------------------------------------------
    def _sound(self, en: str, fr: str) -> float:
        def g2p(t, l):
            k = (t, l)
            if k not in self._g2p_cache:
                self._g2p_cache[k] = self.matcher.g2p(t, l)
            return self._g2p_cache[k]
        ia, ib = g2p(en, "en"), g2p(fr, "fr")
        return 0.5 * self.matcher._ngram_channel(ia, ib) \
             + 0.5 * self.matcher._feat_channel(ia, ib)

    def energy(self, en_words, fr_words, anchor) -> tuple[float, dict]:
        en, fr = " ".join(en_words), " ".join(fr_words)
        if self.mode == "minilm":
            m_en = self.meaning.sim(en, anchor)
            m_fr = self.meaning.sim(fr, anchor)
        else:
            m_en = self.meaning.sim_bags(self.meaning._bag(en, "en"), anchor)
            m_fr = self.meaning.sim_bags(self.meaning._bag(fr, "fr"), anchor)
        snd = self._sound(en, fr)
        e = (W_MEANING * (2 - m_en - m_fr)
             + W_SOUND * (1 - snd)
             + LEN_PENALTY * abs(len(en_words) - len(fr_words)))
        return e, {"m_en": m_en, "m_fr": m_fr, "sound": snd}

    # ---- moves --------------------------------------------------------------
    def propose(self, en_words, fr_words, max_len: int):
        en, fr = list(en_words), list(fr_words)
        move = self.rng.random()
        if move < 0.30 and en:                       # EN paraphrase swap
            i = self.rng.randrange(len(en))
            mates = self.en_mates.get(en[i])
            if mates:
                en[i] = self.rng.choice(mates)
        elif move < 0.55 and en:                     # FR re-cell for an EN word
            i = self.rng.randrange(len(en))
            cells = list(self.composer.inv.get(en[i], {}))
            if cells:
                j = min(i, len(fr) - 1) if fr else 0
                if fr:
                    fr[j] = self.rng.choice(cells)
        elif move < 0.72 and fr:                     # FR homonym pivot (sound-free)
            j = self.rng.randrange(len(fr))
            homs = self.fr_homs.get(fr[j])
            if homs:
                fr[j] = self.rng.choice(homs)
        elif move < 0.86 and len(en) < max_len:      # poetic insertion (both rails)
            i = self.rng.randrange(len(en) + 1)
            en.insert(i, self.rng.choice(self.en_fillers))
            fr.insert(min(i, len(fr)), self.rng.choice(self.fr_fillers))
        elif len(en) > 3 and len(fr) > 3:            # deletion
            i = self.rng.randrange(len(en))
            en.pop(i)
            fr.pop(min(i, len(fr) - 1))
        return en, fr

    # ---- the loop -----------------------------------------------------------
    def diffuse(self, sentence: str, iters: int = 400, expand: float = 1.6,
                t0: float = 0.12):
        anchor = (self.meaning.encode(sentence) if self.mode == "minilm"
                  else self.meaning._bag(sentence, "en"))
        max_len = max(6, int(len(sentence.split()) * expand))

        # init from the lattice composer (structured noise)
        _, finals = self.composer.compose(sentence, n_out=1)
        if finals:
            _, fr_line, path, _ = finals[0]
            en_words = sentence.lower().split()
            fr_words = fr_line.split()
        else:
            en_words = sentence.lower().split()
            fr_words = sentence.lower().split()

        e, parts = self.energy(en_words, fr_words, anchor)
        best = (e, list(en_words), list(fr_words), parts)
        print(f"init  E={e:.3f}  sound={parts['sound']:.2f} "
              f"m_en={parts['m_en']:.2f} m_fr={parts['m_fr']:.2f}")

        for it in range(iters):
            t = t0 * (1 - it / iters) + 1e-4
            ne, nf = self.propose(en_words, fr_words, max_len)
            if (ne, nf) == (en_words, fr_words):
                continue
            e2, parts2 = self.energy(ne, nf, anchor)
            if e2 < e or self.rng.random() < math.exp((e - e2) / t):
                en_words, fr_words, e, parts = ne, nf, e2, parts2
                if e < best[0]:
                    best = (e, list(en_words), list(fr_words), parts2)
        return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("."))
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--expand", type=float, default=1.6,
                    help="max growth factor (1 sentence may become ~3 short ones)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("sentence", nargs="+")
    args = ap.parse_args()

    d = DualDiffuser(args.bench_dir, args.pipeline_dir, seed=args.seed)
    sentence = " ".join(args.sentence)
    e, en_w, fr_w, parts = d.diffuse(sentence, iters=args.iters,
                                     expand=args.expand)
    print(f"\nanchor (original): {sentence}")
    print(f"EN': {' '.join(en_w)}")
    print(f"FR': {' '.join(fr_w)}")
    print(f"E={e:.3f}  sound={parts['sound']:.2f}  "
          f"meaning EN'={parts['m_en']:.2f}  meaning FR'={parts['m_fr']:.2f}")


if __name__ == "__main__":
    main()
