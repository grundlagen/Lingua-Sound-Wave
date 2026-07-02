"""ONE tier ladder: unify every vetted pair source into a single dataset.

Sources merged (each a column, not a vote -- provenance preserved):
  v5 S/A/B          old LLM-coherence tiers (tier-grader: both sides read + sound>=0.85)
  v7 GOLD/A/B       prosody>=0.70 x cosine>=0.45 (softer, symbolic)
  STRICT-GOLD       geo-ensemble>=0.60 AND beats nearest rival (strict_judge)
  loop-certified    dual atoms certified by >=1 closed loop (x count)
  DUAL-S/A/B        literal MUSE translation AND homophone (dual_mine)

Rank (top first):  DUAL-S > S(v5) > STRICT-GOLD > loop>=2 > DUAL-A > GOLD(v7)
                   > loop=1 > DUAL-B > A > B
The LADDER column is that single rank; per-source columns stay for audit.

Out: tier-ladder.tsv + counts.   Run: python tier_ladder.py
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict


def K(en, fr):
    return (en.strip().lower(), fr.strip().lower())


def main():
    rows = defaultdict(dict)      # (en,fr) -> {source: value}

    # v5 (old S tier)
    for i, line in enumerate(open("dictionary-v5.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 5:
            rows[K(p[3], p[4])]["v5"] = p[0]

    # v7 remined
    for i, line in enumerate(open("dictionary-v7-remined.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6:
            rows[K(p[0], p[1])]["v7"] = p[5]
            rows[K(p[0], p[1])]["sound"] = p[3]
            rows[K(p[0], p[1])]["meaning"] = p[4]

    # STRICT-GOLD
    if os.path.exists("strict-gold.tsv"):
        for i, line in enumerate(open("strict-gold.tsv", encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                rows[K(p[0], p[1])]["strict"] = "STRICT-GOLD"

    # loop-certified (aug preferred)
    lc = "loop-certified-pairs-v7u-aug.tsv" if os.path.exists(
        "loop-certified-pairs-v7u-aug.tsv") else "loop-certified-pairs-v7u.tsv"
    for i, line in enumerate(open(lc, encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            rows[K(p[0], p[1])]["loops"] = int(p[2])

    # DUAL (may be mid-mine; take what exists)
    if os.path.exists("dual-pairs.tsv"):
        for i, line in enumerate(open("dual-pairs.tsv", encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 6:
                rows[K(p[0], p[1])]["dual"] = p[5]
                rows[K(p[0], p[1])]["cognate"] = p[4]

    def ladder(r):
        d, v5, v7 = r.get("dual", ""), r.get("v5", ""), r.get("v7", "")
        loops = r.get("loops", 0)
        if d == "DUAL-S":
            return 1, "DUAL-S"
        if v5 == "S":
            return 2, "S"
        if "strict" in r:
            return 3, "STRICT-GOLD"
        if loops >= 2:
            return 4, f"LOOP{loops}"
        if d == "DUAL-A":
            return 5, "DUAL-A"
        if v7 == "GOLD":
            return 6, "GOLD"
        if loops == 1:
            return 7, "LOOP1"
        if d == "DUAL-B":
            return 8, "DUAL-B"
        if "A" in (v5, v7):
            return 9, "A"
        return 10, "B"

    out = []
    for (en, fr), r in rows.items():
        rank, name = ladder(r)
        out.append((rank, en, fr, name, r))
    out.sort()
    counts = Counter(name for _, _, _, name, _ in out)

    with open("tier-ladder.tsv", "w", encoding="utf-8") as f:
        f.write("rank\ten\tfr\tladder\tv5\tv7\tstrict\tloops\tdual\tcognate\tsound\tmeaning\n")
        for rank, en, fr, name, r in out:
            f.write(f"{rank}\t{en}\t{fr}\t{name}\t{r.get('v5','')}\t{r.get('v7','')}\t"
                    f"{'1' if 'strict' in r else ''}\t{r.get('loops','')}\t"
                    f"{r.get('dual','')}\t{r.get('cognate','')}\t"
                    f"{r.get('sound','')}\t{r.get('meaning','')}\n")

    print(f"tier-ladder.tsv: {len(out)} pairs")
    for name in ("DUAL-S", "S", "STRICT-GOLD", "LOOP12", "DUAL-A", "GOLD",
                 "LOOP1", "DUAL-B", "A", "B"):
        pass
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:12s} {n}")
    top = [x for x in out if x[0] <= 5][:12]
    print("\ntop of the ladder:")
    for rank, en, fr, name, r in top:
        print(f"  {name:12s} {en} ~ {fr}")


if __name__ == "__main__":
    main()
