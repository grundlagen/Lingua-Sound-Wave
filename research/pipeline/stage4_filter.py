"""STAGE 4 — the cross-language filter over the stage-3 expansion.

For each gold pair (en, fr): cross-score every EN variant x FR variant
neighborhood with the AUC-0.993 combo matcher and keep what actually works
as sound in the other language. The original gold pair is excluded — only
NEW pairs are emitted, each with provenance back to its gold parent.

Tiers on the way out (matcher benchmark thresholds):
  STRICT >= 0.60      new gold candidates
  PASS   >= 0.45      usable, bank-grade
  (below 0.45 dropped)

g2p (espeak-ng) is memoized per unique word, so the cross-product costs
alignments, not subprocesses. Output:

  expansion-verified.tsv   en_variant fr_variant score tier gold_en gold_fr src_tier

Run: python stage4_filter.py --bench-dir <dir> --expansion-dir out/ --out-dir out/
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path


def load_matcher(bench: Path):
    spec = importlib.util.spec_from_file_location("matcher", bench / "matcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_expansion(path: Path) -> dict[str, list[str]]:
    exp = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            exp[r["word"]].append(r["variant"])
    return exp


def load_gold(bench: Path, tiers: set[str]):
    rows = []
    with open(bench / "tier-ladder.tsv", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            if r["ladder"] in tiers:
                rows.append((r["en"].strip(), r["fr"].strip(), r["ladder"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--expansion-dir", type=Path, default=Path("out"))
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    ap.add_argument("--tiers", default="DUAL-S,S,STRICT-GOLD,LOOP2,LOOP1,GOLD")
    ap.add_argument("--pass-floor", type=float, default=0.45)
    ap.add_argument("--strict-floor", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0, help="gold pairs to process (0=all)")
    args = ap.parse_args()

    matcher = load_matcher(args.bench_dir)

    # memoize g2p so each unique word costs one espeak call total
    g2p_cache: dict[tuple[str, str], str] = {}
    raw_g2p = matcher.g2p

    def g2p(text: str, lang: str) -> str:
        key = (text, lang)
        if key not in g2p_cache:
            g2p_cache[key] = raw_g2p(text, lang)
        return g2p_cache[key]

    def score(en: str, fr: str) -> float:
        ia, ib = g2p(en, "en"), g2p(fr, "fr")
        return 0.5 * matcher._ngram_channel(ia, ib) + 0.5 * matcher._feat_channel(ia, ib)

    gold = load_gold(args.bench_dir, set(args.tiers.split(",")))
    if args.limit:
        gold = gold[: args.limit]
    exp_en = load_expansion(args.expansion_dir / "expansion-en.tsv")
    exp_fr = load_expansion(args.expansion_dir / "expansion-fr.tsv")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "expansion-verified.tsv"
    seen: set[tuple[str, str]] = set()
    n_strict = n_pass = n_tested = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        f.write("en\tfr\tscore\ttier\tgold_en\tgold_fr\tsrc_tier\n")
        for gi, (gen, gfr, gtier) in enumerate(gold):
            en_side = [gen] + exp_en.get(gen, [])
            fr_side = [gfr] + exp_fr.get(gfr, [])
            for ev in en_side:
                for fv in fr_side:
                    if (ev, fv) == (gen, gfr) or (ev, fv) in seen:
                        continue
                    seen.add((ev, fv))
                    n_tested += 1
                    try:
                        s = score(ev, fv)
                    except Exception:
                        continue
                    if s >= args.strict_floor:
                        tier = "STRICT"
                        n_strict += 1
                    elif s >= args.pass_floor:
                        tier = "PASS"
                        n_pass += 1
                    else:
                        continue
                    f.write(f"{ev}\t{fv}\t{s:.3f}\t{tier}\t{gen}\t{gfr}\t{gtier}\n")
            if (gi + 1) % 500 == 0:
                print(f"  {gi+1}/{len(gold)} gold pairs | tested {n_tested} "
                      f"| STRICT {n_strict} PASS {n_pass} "
                      f"| {time.time()-t0:.0f}s", flush=True)

    print(f"\ndone: {len(gold)} gold pairs, {n_tested} candidates tested, "
          f"{n_strict} STRICT + {n_pass} PASS new pairs -> {out} "
          f"({time.time()-t0:.0f}s, {len(g2p_cache)} espeak calls)")


if __name__ == "__main__":
    main()
