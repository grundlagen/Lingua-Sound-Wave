"""Learn the phoneme-substitution cost model from our own certified data.

The matcher's equivalence table (rhotics 0.10, voicing 0.20...) is
hand-curated. We now have better evidence: the alignments of S-tier and
loop-certified entries are EN<->FR phoneme correspondences that survived
every quality gate — including, for the certified subset, a meaning-
preserving round trip. This script mines them into a learned cost table
(the Ristad & Yianilos learned-edit-distance idea, count-based):

  cost(a,b) = 1 / (1 + ln(1 + count(a,b)))     (monotone in evidence)
  gap(s)    = same curve over observed deletions

and validates on the frozen benchmark before anyone trusts it: the learned
table is only worth committing if matcher AUC does not regress. (Honest
caveat recorded here: the alignments come from the matcher itself, so this
is self-distillation — the bench check guards against drift, and the real
payoff is recall on FUTURE mining, where rare-but-real correspondences the
hand table missed now have earned costs.)

Run: python learn_costs.py
Outputs: learned-costs.json (+ benchmark comparison on stdout)
"""
from __future__ import annotations

import json
import math
from collections import Counter


def main():
    entries = json.load(open("dictionary-v5.json"))
    trusted = [e for e in entries
               if (e.get("tier") == "S" or e.get("loop_certified"))
               and e.get("align")]
    subs, gaps, idents = Counter(), Counter(), 0
    for e in trusted:
        for a, b, c in e["align"]:
            if a == "·":
                gaps[b] += 1
            elif b == "·":
                gaps[a] += 1
            elif a == b:
                idents += 1
            else:
                subs[tuple(sorted((a.replace("ː", ""), b.replace("ː", ""))))] += 1

    def curve(n):
        return round(1.0 / (1.0 + math.log(1 + n)), 3)

    pair_costs = {f"{a}|{b}": curve(n) for (a, b), n in subs.items() if n >= 2}
    gap_costs = {s.replace("ː", ""): min(0.42, curve(n))
                 for s, n in gaps.items() if n >= 3}
    out = {"pairs": pair_costs, "gaps": gap_costs,
           "meta": {"trusted_entries": len(trusted), "identities": idents,
                    "distinct_subs": len(subs), "distinct_gaps": len(gaps)}}
    with open("learned-costs.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"learned from {len(trusted)} trusted alignments: "
          f"{len(pair_costs)} pair costs, {len(gap_costs)} gap costs")
    top = sorted(subs.items(), key=lambda kv: -kv[1])[:12]
    for (a, b), n in top:
        print(f"  {a}~{b:3s} x{n:4d} -> cost {curve(n)}  "
              f"(hand table would say: see matcher.EQUIV)")

    # ---- validation: benchmark AUC with vs without learned costs ----
    import matcher
    from dataset import all_pairs

    def auc_run():
        pos, neg = [], []
        for en, fr, label, _t in all_pairs():
            s = matcher.homophone_score(en, "en", fr, "fr")["score"]
            (pos if label else neg).append(s)
        wins = sum(1 for p in pos for n in neg if p > n)
        ties = sum(1 for p in pos for n in neg if p == n)
        return (wins + 0.5 * ties) / (len(pos) * len(neg))

    base = auc_run()
    # merge learned costs (only where they tighten the model)
    n_pair_applied = 0
    for k, c in pair_costs.items():
        a, b = k.split("|")
        key = tuple(sorted((a, b)))
        if c < matcher.EQUIV.get(key, 1.0):
            matcher.EQUIV[key] = c
            n_pair_applied += 1
    n_gap_applied = 0
    for s, c in gap_costs.items():
        if c < matcher.CHEAP_GAP.get(s, matcher.GAP):
            matcher.CHEAP_GAP[s] = c
            n_gap_applied += 1
    # clear caches that captured old costs
    matcher._feat_channel.__dict__.clear() if hasattr(matcher._feat_channel, "__dict__") else None
    learned = auc_run()
    print(f"\nbenchmark AUC: hand-table {base:.3f} -> learned-merged {learned:.3f} "
          f"({n_pair_applied} pair costs, {n_gap_applied} gap costs applied)")
    verdict = "SAFE: no regression" if learned >= base - 0.002 else "REGRESSION: keep opt-in"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
