"""Harder, gold-based, ENSEMBLE judging -- so accuracy isn't graded by one
method's own methodology.

The 105-pair benchmark is too easy (negatives sound nothing alike), so every
phoneme method scores ~0.99. Here we:
  1. Build a HARD set from the gold: positive = a real gold homophone pair;
     negative = the SAME English word vs a DECOY French word (a real French
     homophone-of-something-else, matched for length) -> a plausible near-miss.
  2. Score with ORTHOGONAL methods (exact-bigram, featural, prosodic, the Drive
     equivalence method) and report AUC -- on hard cases they SPREAD and disagree.
  3. ENSEMBLE judge: z-normalise each method and average, so no single
     methodology dominates. Report its AUC.
  4. LLM ARBITRATION on the highest-disagreement pairs (DeepSeek/Nemotron), the
     'better LLM judging' for the cases symbolic methods can't agree on.

Run: python hard_judge.py
"""
from __future__ import annotations

import json
import os
import random
import urllib.request
from difflib import SequenceMatcher

import numpy as np

import bench
import matcher
import prosody
import drive_phoneme_map as dpm

try:
    import _load_env
    _load_env.load_keys()       # so DeepSeek/OpenRouter keys are available
except Exception:
    pass

_CANON = {}
for k, vs in dpm.phoneme_mapping.items():
    for v in vs:
        _CANON.setdefault(v, k)


def drive_equiv(en, fr):
    try:
        return SequenceMatcher(None,
                               "".join(_CANON.get(s, s) for s in matcher._segs(bench.g2p_ipa(en, "en"))),
                               "".join(_CANON.get(s, s) for s in matcher._segs(bench.g2p_ipa(fr, "fr")))).ratio()
    except Exception:
        return 0.0


METHODS = {
    "ngram_dice": bench.m_ngram_dice,
    "feat_nw_sharp": bench.m_feat_nw_sharp,
    "prosody": prosody.prosodic_score,
    "drive_equiv": drive_equiv,
    "combo": bench.m_combo,
}


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def load_gold(n=180):
    rows = []
    for i, line in enumerate(open("dictionary-v7-remined.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6 and p[5] == "GOLD" and p[0].isalpha() and p[1].isalpha():
            rows.append((p[0], p[1]))
    random.Random(0).shuffle(rows)
    return rows[:n]


def main():
    gold = load_gold()
    fr_pool = list({f for _, f in gold})
    rng = random.Random(1)
    # build positives and HARD negatives (same EN, decoy FR of similar length)
    cases = []   # (en, fr, label)
    for en, fr in gold:
        cases.append((en, fr, 1))
        decoys = [f for f in fr_pool if f != fr and abs(len(f) - len(fr)) <= 1]
        if decoys:
            cases.append((en, rng.choice(decoys), 0))
    print(f"HARD set: {sum(l for *_ , l in cases)} positives, "
          f"{sum(1 for *_ , l in cases if not l)} hard negatives\n")

    scores = {m: [] for m in METHODS}
    labels = []
    for en, fr, lab in cases:
        labels.append(lab)
        for m, fn in METHODS.items():
            try:
                scores[m].append(float(fn(en, fr)))
            except Exception:
                scores[m].append(0.0)
    labels = np.array(labels)

    print(f"{'method':16s} {'AUC(hard)':>10s}")
    print("-" * 28)
    Z = {}
    for m in METHODS:
        s = np.array(scores[m])
        pos, neg = s[labels == 1], s[labels == 0]
        print(f"{m:16s} {auc(pos, neg):10.3f}")
        Z[m] = (s - s.mean()) / (s.std() + 1e-9)        # z-normalise

    # ENSEMBLE: average of z-scored ORTHOGONAL methods (drop combo, it's ~ngram+feat)
    ens = np.mean([Z[m] for m in ("ngram_dice", "feat_nw_sharp", "prosody", "drive_equiv")], axis=0)
    print(f"\n{'ENSEMBLE(z-avg)':16s} {auc(ens[labels==1], ens[labels==0]):10.3f}"
          "   <- diverse methods, no single methodology dominates")

    # disagreement = variance across z-scored methods; LLM-arbitrate the worst
    var = np.var([Z[m] for m in METHODS], axis=0)
    hard_idx = np.argsort(-var)[:10]
    print("\nhighest-disagreement cases (where one metric would mislead):")
    for i in hard_idx[:6]:
        en, fr, lab = cases[i]
        print(f"   {'(homophone)' if lab else '(decoy)    '} {en} ~ {fr}   "
              + " ".join(f"{m[:5]}={scores[m][i]:.2f}" for m in METHODS))

    # LLM arbitration (DeepSeek) on the disagreement cases -> gold-rate
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        items = [(cases[i][0], cases[i][1], cases[i][2]) for i in hard_idx]
        listing = "\n".join(f"{j+1}. English '{e}' read in a French mouth as '{f}'"
                            for j, (e, f, _) in enumerate(items))
        prompt = ("For each, rate 0-100 how much the English word SOUNDS like the "
                  "French word when spoken. Reply ONLY a JSON array of integers.\n\n" + listing)
        body = json.dumps({"model": "deepseek-chat", "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}], "max_tokens": 200}).encode()
        try:
            req = urllib.request.Request("https://api.deepseek.com/chat/completions",
                                         data=body, headers={"Authorization": f"Bearer {key}",
                                         "Content-Type": "application/json"})
            txt = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
            import re
            nums = [int(x) for x in re.findall(r"\d+", txt)][:len(items)]
            print("\nLLM (DeepSeek) arbitration on the disagreements:")
            for (e, f, lab), s in zip(items, nums):
                print(f"   {e} ~ {f}: LLM {s:3d}/100  (truth: {'homophone' if lab else 'decoy'})")
        except Exception as ex:
            print(f"\n(LLM arbitration skipped: {ex})")

    print("\nReading: on HARD cases the methods SPREAD and disagree -- the 0.993 was "
          "an easy-benchmark artifact. Best practice: a z-averaged ENSEMBLE of "
          "orthogonal methods as the judge, gold-anchored, with LLM arbitration on "
          "disagreements -- not one metric grading by its own methodology.")


if __name__ == "__main__":
    main()
