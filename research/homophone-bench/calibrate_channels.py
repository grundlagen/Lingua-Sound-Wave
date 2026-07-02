"""Per-channel logistic calibration on strict-gold (MATHEMATICS.md §6).

Positives: strict-gold.tsv pairs. Negatives: nearest-confusable decoys
(hardest wrong FR by combo sound). Features: the strict_judge channels +
the meaning channel. One logistic regression -> channel-calibration.json.
beauty_compose picks the weights up if the file exists.

Usage:
    python calibrate_channels.py [--n 300] [--no-meaning]
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

import bench
import prosody
from hard_judge import drive_equiv
from strict_judge import auc, geo_mean, nearest_confusable, _safe

FEATURES = {
    "ngram_dice": bench.m_ngram_dice,
    "feat_nw_sharp": bench.m_feat_nw_sharp,
    "prosody": prosody.prosodic_score,
    "drive_equiv": drive_equiv,
}

OUT = "channel-calibration.json"


def load_strict_gold(n):
    rows = []
    for i, line in enumerate(open("strict-gold.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0] and p[1]:
            rows.append((p[0], p[1]))
    random.Random(0).shuffle(rows)
    return rows[:n]


def build_cases(gold):
    fr_pool = list({f for _, f in gold})
    cases = []
    for k, (en, fr) in enumerate(gold):
        cases.append((en, fr, 1))
        decoy, _ = nearest_confusable(en, fr, fr_pool)
        if decoy:
            cases.append((en, decoy, 0))
        if (k + 1) % 50 == 0:
            print(f"  decoys {k + 1}/{len(gold)}")
    return cases


def featurize(cases, use_meaning):
    feats = dict(FEATURES)
    if use_meaning:
        from semantic_cosine import semantic_cosine
        feats["meaning"] = lambda en, fr: max(0.0, semantic_cosine(en, fr))
    names = list(feats)
    X = np.array([[_safe(fn, en, fr) for fn in feats.values()]
                  for en, fr, _ in cases])
    y = np.array([lab for *_, lab in cases])
    return names, X, y


def fit_logistic(X, y, l2=1.0, iters=500, lr=0.5):
    # plain IRLS-free gradient logistic; no sklearn dependency
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0] / len(y)
        w -= lr * g
    return w


def predict(w, X):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return 1.0 / (1.0 + np.exp(-Xb @ w))


def strict_gold_rate(scores, cases, bar, margin=0.10):
    passed = total = 0
    for i in range(0, len(cases) - 1, 2):
        if cases[i][2] != 1 or cases[i + 1][2] != 0:
            continue
        total += 1
        passed += (scores[i] >= bar) and (scores[i] - scores[i + 1] >= margin)
    return passed, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--no-meaning", action="store_true")
    args = ap.parse_args()

    gold = load_strict_gold(args.n)
    print(f"strict-gold sample: {len(gold)} pairs; building decoys ...")
    cases = build_cases(gold)
    names, X, y = featurize(cases, use_meaning=not args.no_meaning)
    print(f"{sum(y)} pos / {sum(y == 0)} neg, features: {names}")

    print(f"\n{'feature':16s} {'AUC':>7s}")
    for j, nm in enumerate(names):
        print(f"{nm:16s} {auc(X[y == 1, j], X[y == 0, j]):7.3f}")

    # 5-fold CV
    idx = np.arange(0, len(cases) - 1, 2)  # positive indices; decoy = i+1
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)
    cv_auc = []
    for f in range(5):
        te = np.concatenate([folds[f], folds[f] + 1])
        tr = np.setdiff1d(np.arange(len(cases)), te)
        w = fit_logistic(X[tr], y[tr])
        p = predict(w, X)
        cv_auc.append(auc(p[te][y[te] == 1], p[te][y[te] == 0]))

    # geo baseline vs calibrated, full-set strict gold-rate (same bar logic
    # as strict_judge: BAR on score + MARGIN over own decoy)
    geo = np.array([geo_mean(list(X[i])) for i in range(len(cases))])
    gp, gt = strict_gold_rate(geo, cases, bar=0.60)
    w = fit_logistic(X, y)
    p = predict(w, X)
    cp, ct = strict_gold_rate(p, cases, bar=0.50)

    print(f"\nlogistic 5-fold AUC: {np.mean(cv_auc):.3f} "
          f"(folds: {' '.join(f'{a:.3f}' for a in cv_auc)})")
    print(f"geo baseline  AUC: {auc(geo[y == 1], geo[y == 0]):.3f}  "
          f"strict gold-rate {gp}/{gt} = {gp / max(1, gt):.1%}")
    print(f"calibrated    AUC: {auc(p[y == 1], p[y == 0]):.3f}  "
          f"strict gold-rate {cp}/{ct} = {cp / max(1, ct):.1%}")

    out = {"features": names,
           "weights": [float(x) for x in w[:-1]],
           "bias": float(w[-1]),
           "cv_auc": float(np.mean(cv_auc)),
           "n_pairs": int(len(gold))}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}: " + ", ".join(
        f"{nm}={wt:+.2f}" for nm, wt in zip(names, w)) + f", bias={w[-1]:+.2f}")


if __name__ == "__main__":
    main()
