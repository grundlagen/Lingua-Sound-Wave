"""SELF-IMPROVING ACCURACY CYCLE -- expert iteration that ties the best pieces of
the old projects into one loop, with no GPU and no money required.

The pieces it integrates (each proven elsewhere in this repo):
  - rule_aware.py     connected-speech realizations (the rules citation judges
                      forget)  -> the SOUND channel that doesn't under-rate.
  - phoneme_map.py    learn an EN->FR phoneme-similarity model FROM GOLD
                      alignments  -> the data-driven 'embeds' the user asked for.
  - semantic_cosine   multilingual MiniLM meaning channel (METHOD, gates dual
                      reading; never wired into the sound judge).
  - prosody.py        stress/rhythm channel (EN stress-timed vs FR syllable-timed).
  - strict_judge.py   adversarial nearest-confusable negatives + AND-logic
                      (geometric-mean) ensemble + strict gold-rate.

The loop (expert iteration / self-distillation):
  0. BOOTSTRAP gold from the v7 GOLD tier with the rule-aware sound channel.
  1. LEARN a phoneme-similarity model from the current gold's IPA alignments
     (co-occurrence -> a learned segment distance that augments panphon).
  2. RE-SCORE every candidate pair with the ensemble {rule-aware ngram, LEARNED
     featural, prosody} x meaning gate; PROMOTE pairs that pass the STRICT gate
     (beat their nearest confusable rival on a held-out-safe basis).
  3. MEASURE the learned channel on a FROZEN eval (never used for promotion):
     strict AUC + gold-rate. As the learned model sharpens on real gold, the
     frozen number should RISE then plateau -- that is the self-improvement.
  4. Repeat. Write strict-gold.tsv + phoneme-sim-learned.json + a metrics log.

This is bounded, deterministic, paper-mode: it spends no money (LLM arbitration
is OFF by default; enable with --llm only if you accept the cost), deploys
nothing, trades nothing.

Run: python self_improve.py            # 3 iterations, offline
     python self_improve.py --iters 4
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np

import bench
import prosody
import rule_aware
from hard_judge import drive_equiv

try:
    from semantic_cosine import semantic_cosine
    _HAVE_SEM = True
except Exception:
    _HAVE_SEM = False

DICT_JSON = "dictionary-v7-integrated.json"
GOLD_TSV = "dictionary-v7-remined.tsv"

BAR = 0.60          # absolute geo-ensemble bar for STRICT-GOLD
MARGIN = 0.10       # must beat nearest confusable rival by this
MEANING = 0.45      # dual-reading meaning gate (semantic cosine), if available


# ----------------------------------------------------------------- segment ops
def segs_vecs(ipa: str):
    s, v = bench.segs_and_vecs(bench.canonical(ipa))
    return list(s), v


def panphon_dist(va, i, vb, j) -> float:
    return bench.feat_dist(va[i], vb[j])


def nw(sa, va, sb, vb, distfn):
    """Needleman-Wunsch; returns (avg per-step cost, aligned segment pairs)."""
    n, m = len(sa), len(sb)
    if n == 0 or m == 0:
        return 1.0, []
    GAP = 0.42
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    ln = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * GAP; bt[i][0] = "U"; ln[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j * GAP; bt[0][j] = "L"; ln[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = D[i - 1][j - 1] + distfn(sa, va, i - 1, sb, vb, j - 1)
            de, ins = D[i - 1][j] + GAP, D[i][j - 1] + GAP
            best = min(sub, de, ins)
            D[i][j] = best
            if best == sub:
                bt[i][j] = "D"; ln[i][j] = ln[i - 1][j - 1] + 1
            elif best == de:
                bt[i][j] = "U"; ln[i][j] = ln[i - 1][j] + 1
            else:
                bt[i][j] = "L"; ln[i][j] = ln[i][j - 1] + 1
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        d = bt[i][j]
        if d == "D":
            pairs.append((sa[i - 1], sb[j - 1])); i -= 1; j -= 1
        elif d == "U":
            i -= 1
        else:
            j -= 1
    return D[n][m] / max(1, ln[n][m]), pairs


# --------------------------------------------------- learned phoneme similarity
class LearnedModel:
    """EN->FR segment similarity learned from gold alignments (the 'embeds').

    Co-occurrence in gold alignments -> P(fr_seg | en_seg). A high-probability
    realization gets a LOW distance; rare/unknown pairs fall back to panphon
    articulatory distance. Blended so the model only sharpens what gold supports.
    """

    def __init__(self):
        self.counts = defaultdict(Counter)
        self.tot = defaultdict(int)

    def fit(self, gold_ipa, distfn):
        self.counts.clear(); self.tot.clear()
        for ei, fi in gold_ipa:
            sa, va = segs_vecs(ei); sb, vb = segs_vecs(fi)
            if not sa or not sb:
                continue
            _, pairs = nw(sa, va, sb, vb, distfn)
            for a, b in pairs:
                self.counts[a][b] += 1
                self.tot[a] += 1

    def dist(self, sa, va, i, sb, vb, j) -> float:
        a, b = sa[i], sb[j]
        base = panphon_dist(va, i, vb, j)
        if a == b:
            return 0.0
        ta = self.tot.get(a, 0)
        if ta < 5:                      # not enough evidence -> trust panphon
            return base
        p = self.counts[a].get(b, 0) / ta
        learned = 1.0 - min(1.0, p / 0.5)      # P(b|a)>=0.5 -> distance ~0
        return 0.5 * base + 0.5 * learned

    def feat_sim(self, en, fr) -> float:
        """Rule-aware: MAX learned-NW similarity over connected-speech realizations."""
        ei, fi = bench.g2p_ipa(en, "en"), bench.g2p_ipa(fr, "fr")
        best = 0.0
        for a in rule_aware.en_realizations(ei)[:8]:
            sa, va = segs_vecs(a)
            for b in rule_aware.fr_realizations(fi)[:6]:
                sb, vb = segs_vecs(b)
                c, _ = nw(sa, va, sb, vb, self.dist)
                best = max(best, 1.0 - c)
        return best

    def to_json(self):
        return {a: dict(c.most_common(8)) for a, c in self.counts.items()}


def geo(vals):
    vals = [max(1e-6, v) for v in vals]
    return float(np.prod(vals) ** (1.0 / len(vals)))


# ----------------------------------------------------------------------- data
def load_dict_ipa():
    pairs = []
    for e in json.load(open(DICT_JSON, encoding="utf-8")):
        if e.get("en") and e.get("fr") and e.get("en_ipa") and e.get("fr_ipa"):
            pairs.append((e["en"], e["fr"], e["en_ipa"], e["fr_ipa"]))
    return pairs


def load_gold_pairs(n):
    rows = []
    for i, line in enumerate(open(GOLD_TSV, encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6 and p[5] == "GOLD" and p[0].isalpha() and p[1].isalpha():
            rows.append((p[0], p[1]))
    random.Random(0).shuffle(rows)
    return rows[:n]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return 0.0
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--train", type=int, default=1200, help="gold pairs to learn from")
    ap.add_argument("--eval", type=int, default=120, help="frozen eval positives")
    ap.add_argument("--llm", action="store_true", help="enable LLM arbitration (COSTS MONEY)")
    args = ap.parse_args()

    ipa_of = {}
    dict_pairs = load_dict_ipa()
    for en, fr, ei, fi in dict_pairs:
        ipa_of[(en, fr)] = (ei, fi)

    # FROZEN eval: positives + nearest-confusable negatives (never used to learn)
    eval_gold = load_gold_pairs(args.eval)
    fr_pool = list({f for _, f in eval_gold})
    eval_cases = []
    for en, fr in eval_gold:
        eval_cases.append((en, fr, 1))
        best, bs = None, -1.0
        for f in fr_pool:
            if f == fr:
                continue
            s = bench.m_combo(en, f)
            if s > bs:
                best, bs = f, s
        if best:
            eval_cases.append((en, best, 0))
    eval_keys = {en for en, *_ in eval_cases}

    # TRAIN gold (disjoint from eval): IPA pairs to learn the phoneme model from
    train_gold = [(en, fr) for en, fr in load_gold_pairs(args.train + args.eval * 3)
                  if en not in eval_keys][:args.train]
    train_ipa = [ipa_of[(e, f)] for e, f in train_gold if (e, f) in ipa_of]
    print(f"frozen eval: {sum(l for *_, l in eval_cases)} pos + "
          f"{sum(1 for *_, l in eval_cases if not l)} nearest-confusable neg")
    print(f"train gold (disjoint, with IPA): {len(train_ipa)} pairs\n")

    model = LearnedModel()
    distfn = panphon_dist_wrap = lambda sa, va, i, sb, vb, j: panphon_dist(va, i, vb, j)
    log = []
    print(f"{'iter':4s} {'eval AUC':>9s} {'gold-rate':>10s} {'meanPos':>8s} {'meanNeg':>8s}  learned-segs")
    print("-" * 64)
    for it in range(args.iters):
        # (1) LEARN from current gold using current distances
        model.fit(train_ipa, distfn if it > 0 else panphon_dist_wrap)

        # (3) MEASURE on frozen eval with the learned featural channel in the ensemble
        pos, neg, gp, gn = [], [], [], []
        pairscore = []
        for en, fr, lab in eval_cases:
            sound = geo([
                rule_aware.rule_aware_ngram(en, fr),
                model.feat_sim(en, fr),
                prosody.prosodic_score(en, fr),
            ])
            (pos if lab else neg).append(sound)
            pairscore.append((en, fr, lab, sound))
        a = auc([s for *_x, l, s in [(e, f, l, s) for e, f, l, s in pairscore] if l == 1],
                [s for e, f, l, s in pairscore if l == 0])
        # strict gold-rate on the frozen eval (positive beats its paired decoy)
        passed = total = 0
        for k in range(0, len(pairscore) - 1, 2):
            if pairscore[k][2] == 1 and pairscore[k + 1][2] == 0:
                total += 1
                ps, ns = pairscore[k][3], pairscore[k + 1][3]
                passed += (ps >= BAR) and (ps - ns >= MARGIN)
        gr = passed / max(1, total)
        mp = float(np.mean([s for *_x, l, s in [(e, f, l, s) for e, f, l, s in pairscore] if l == 1]))
        mn = float(np.mean([s for e, f, l, s in pairscore if l == 0]))
        print(f"{it:<4d} {a:9.3f} {gr:9.1%} {mp:8.3f} {mn:8.3f}  {len(model.tot)} EN phonemes")
        log.append({"iter": it, "auc": a, "gold_rate": gr, "mean_pos": mp, "mean_neg": mn,
                    "learned_segs": len(model.tot)})

        # (2) EXPERT ITERATION: promote new STRICT-GOLD from the dictionary using
        # the *current* learned model, then fold into training gold for next round.
        if it < args.iters - 1:
            cand = [(en, fr, ei, fi) for (en, fr, ei, fi) in dict_pairs
                    if en not in eval_keys and (en, fr) not in set(train_gold)]
            random.Random(100 + it).shuffle(cand)
            added = 0
            # group candidates by EN to build nearest-rival decoys within the batch
            batch = cand[:800]
            by_en = defaultdict(list)
            for en, fr, ei, fi in batch:
                by_en[en].append(fr)
            for en, fr, ei, fi in batch:
                sound = geo([
                    rule_aware.rule_aware_ngram(en, fr),
                    model.feat_sim(en, fr),
                    prosody.prosodic_score(en, fr),
                ])
                if sound < BAR:
                    continue
                # nearest rival = best-sounding OTHER french for this EN in batch
                rivals = [g for g in by_en[en] if g != fr]
                if rivals:
                    rs = max(bench.m_combo(en, g) for g in rivals)
                    if sound - rs < MARGIN:
                        continue
                if _HAVE_SEM and semantic_cosine(en, fr) < MEANING:
                    continue
                train_gold.append((en, fr))
                if (en, fr) in ipa_of:
                    train_ipa.append(ipa_of[(en, fr)])
                added += 1
            print(f"      promoted {added} new STRICT-GOLD pairs "
                  f"(train gold now {len(train_ipa)})")
            distfn = model.dist

    # write artifacts
    with open("phoneme-sim-learned.json", "w", encoding="utf-8") as f:
        json.dump(model.to_json(), f, ensure_ascii=False, indent=1)
    with open("strict-gold.tsv", "w", encoding="utf-8") as f:
        f.write("en\tfr\n")
        for e, fr in train_gold:
            f.write(f"{e}\t{fr}\n")
    with open("self-improve-log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)

    d_auc = log[-1]["auc"] - log[0]["auc"]
    d_gr = log[-1]["gold_rate"] - log[0]["gold_rate"]
    print(f"\nself-improvement over {args.iters} iters: "
          f"AUC {log[0]['auc']:.3f} -> {log[-1]['auc']:.3f} ({d_auc:+.3f}), "
          f"gold-rate {log[0]['gold_rate']:.1%} -> {log[-1]['gold_rate']:.1%} ({d_gr:+.1%})")
    print("wrote phoneme-sim-learned.json (the learned 'embeds'), strict-gold.tsv "
          "(grown corpus), self-improve-log.json (the curve).")
    print("\nThis is the offline expert-iteration core. Wire run_continual.py / "
          "train_selflearn.py on top to add the neural best-of-N generator when a "
          "GPU is available; the learned phoneme model seeds its reward.")


if __name__ == "__main__":
    main()
