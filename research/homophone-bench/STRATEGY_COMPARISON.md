# Unbiased strategy comparison on nursery rhymes — the frontier does NOT dominate

`compare_strategies.py` runs the same public-domain Mother Goose lines through two
extant strategies, scored identically (matcher combo = sound, bigram coherence,
dual = combo × coherence). All French is the engine's own generated output.

- **Baseline** = the decoder as it shipped (`soramimi` settings): default trie,
  `MIN_WORD_SEGS=2` (no fillers), default penalties, ranked by the decoder's own
  similarity.
- **Frontier** = whole-line carve: poetry trie (1-seg fillers), coverage-forcing,
  arbiter (combo × coherence) ranking.

## Result

| line | baseline (dual) | frontier (dual) | dual winner |
|---|---|---|---|
| Humpty Dumpty | épidémie (0.16) | un petit un petit (0.48) | **frontier** |
| Humpty Dumpty sat on a wall | épidémie (0.12) | un petit un petit et on vol (0.42) | **frontier** |
| Pat a cake | but équipe (0.30) | but y (0.35) | **frontier** |
| Little Jack Horner | logique (0.19) | let gendarmes (0.20) | tie |
| Jack and Jill | desquels (0.13) | cicatrices (0.08) | **baseline** |
| Hickory dickory dock | critiques (0.26) | — (0.00) | **baseline** |

**Tally: dual wins frontier 3, baseline 2, tie 1. Raw-sound wins 3–3.**

## The honest reading (not biased to the frontier)

- **The frontier is a specialist, not a strict upgrade.** It wins *decisively* on
  lines that carve into a fluent filler phrase (Humpty 0.48 vs 0.16), which is the
  van Rooten style. But it does **not** beat the baseline everywhere.
- **The baseline is more robust on hard lines.** For lines dense in non-lexical
  words (Jack and Jill, Hickory dickory dock), the baseline's "grab one tight
  word" strategy still returns a usable match (Hickory → "critiques", combo 0.58),
  while the frontier's coverage-forcing returns **nothing** (0.00) — it demands a
  full carve that does not exist, and gives up. The simpler, "worse" strategy wins
  exactly where you predicted it might.
- **Raw sound is a 3–3 tie**: the baseline's single tight word is often a *closer*
  sound match than a multi-word carve (the épidémie effect) — it just reads worse,
  which is why it loses on dual where coherence matters and wins where it doesn't.

## Conclusion: ensemble, not replacement

No single strategy dominates the nursery-rhyme task. The right system **runs both
(and the chain/weave routers too) and lets the matcher arbiter pick per line** —
frontier for carve-able lines, baseline for dense ones, with the synonym-swap
layer for the lines neither can carve. Treating the frontier as the one true
method would have *lost* Hickory dickory dock and Jack and Jill outright. The
arbiter-picks-per-line ensemble strictly dominates any single strategy.

## Not yet compared (runtime)

The chain/weave routers (`chain_translate.py`; codex `recursive_poet.py`,
`round_rabbit.py`) need the 194k-node graph (~500 s build) and weren't run here.
They are a *different* category (per-word transfer through sound+meaning chains,
and semantic-radius lattices) and should be added to the ensemble bake-off next —
on the same lines, same scoring, no thumb on the scale.
