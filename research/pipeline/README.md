# pipeline/ — the staged generator, cycle 1 executed

Implementation of `PIPELINE.md` stages 2–4 (the no-GPU build order), **run to
completion on 2026-07-06** against the gold data from
`research/homophone-bench` (branch `claude/phrase-weave-multiword`).

## Cycle-1 results (in `out/`)

Seed: 14,887 gold pairs (tiers DUAL-S, S, STRICT-GOLD, LOOP2, LOOP1, GOLD
from `tier-ladder.tsv`) → 12,211 EN / 12,000 FR words.

| stage | output | result |
|---|---|---|
| 2 | `ladders-{en,fr}.json` | 24,211 word ladders; 518 EN / 3,258 FR carry homonym buckets (from the extant homophone-class TSVs) |
| 3 | `expansion-{en,fr}.tsv` | 8,092 EN inflections (LemmInflect) + 10,619 FR forms (rule-generated, Lexique-validated), Zipf-ranked |
| 4 | `expansion-verified.tsv` | 50,494 candidates scored with the AUC-0.993 combo → **27,086 STRICT (≥0.60) + 19,377 PASS (≥0.45) new pairs**, 213 s total |

Non-cognate STRICT pairs (spelling-distance > 0.45): **13,797** — genuinely
new material, e.g. `kept~quêtes 0.78`, `freezer~frisez 0.82`,
`peak~piques 1.00`, `ceasing~six 0.75`, `plucking~plaques 0.61`. Every row
carries provenance (`gold_en`, `gold_fr`, `src_tier`) so tier_ladder.py can
absorb it as a new column per the stage-1 rule.

The cycle-1 STRICT harvest alone nearly **triples** the high-trust pair
inventory (14,887 → ~42k) without any GPU or model — confirming the
build-order bet that inflections of proven words are the highest-probability
new homophones.

## Cycle-2 results (in `out-cycle2/`)

Seed: cycle-1 gold **+ 13,874 EXPANSION-1 pairs** (cycle-1 STRICT rows at a
stricter ≥ 0.70 admission bar — stricter seed, unchanged 0.60/0.45 output so
nothing is discarded) → 28,761 pairs, 13,235 EN / 14,980 FR words.

| stage | result |
|---|---|
| 2 | 28,216 ladders; FR homonym coverage 3,258 → **5,538** words |
| 3 | 10,933 EN + 14,839 FR variants (new gold brought new neighborhoods) |
| 4 | 13,702 candidates (cycle-1's 46,463 pairs excluded) → **+7,653 STRICT / +1,872 PASS genuinely new pairs**, 94 s |

**The flywheel compounds**: 4,536 of the new STRICT pairs are non-cognate,
and nearly all descend from EXPANSION-1 parents — cycle-1 discoveries seeded
cycle-2 discoveries (`very~vernie 0.82`, `lesser~laissés 0.78`,
`salted~soldé 0.68`, `distress~stressées 0.71`, `two~toue 1.00`).

Running inventory after two cycles: 14,887 seed → **~50k verified pairs**
(34,739 STRICT-tier discoveries), zero GPU.

## Scripts

| file | stage | run |
|---|---|---|
| `stage2_ladder.py` | 2 | `python stage2_ladder.py --bench-dir <homophone-bench> --out-dir out/` |
| `stage3_expand.py` | 3 (deterministic half) | `python stage3_expand.py --bench-dir <hb> --ladder-dir out/ --out-dir out/` |
| `stage4_filter.py` | 4 | `python stage4_filter.py --bench-dir <hb> --expansion-dir out/ --out-dir out/` |

Deps: `pip install panphon wordfreq lemminflect pandas` + `espeak-ng` on
PATH. (If panphon's `unicodecsv` dependency fails to build on newer Ubuntu,
install it manually from the sdist and `pip install --no-deps panphon`.)
`matcher.py` is imported from `--bench-dir` — the combo scorer stays the
single source of sound truth; stage 4 memoizes g2p so the cross-product
costs alignments, not subprocesses (21k espeak calls for the full run).

## Honest caveats

- Roughly 40% of verified pairs are cognate-ish (`accept~accepter`): real
  homophones and useful glue, but easy. Filter on spelling distance (as
  above) when you want only the surprising material.
- Stage 4 here scores **sound only**. Meaning placement is stage 5's problem
  (the ladders' homonym/sense buckets exist for exactly that swap-search).
- FR expansion is suffix-rule + Lexique-membership, not true morphology; a
  Lefff pass would add irregular forms the rules miss.

## Next (cycle 2)

1. Absorb `expansion-verified.tsv` STRICT rows into `tier_ladder.py` as a
   provenance column (`EXPANSION-1`).
2. Re-run stages 2–4 on the enlarged gold (new words bring new inflection
   neighborhoods).
3. Start the learned half of stage 3 (paraphrase/phrase generation,
   `research/qwen-finetune/`) on the enlarged corpus.
