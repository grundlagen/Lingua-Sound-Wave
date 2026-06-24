# Homophonic-writing generation: which method combo works (measured)

`generate_pipeline.py` combines the best extant pieces — the v7-integrated
dictionary, a both-sides-coherence chain composer, and Round Rabbit theming — on
light compute (dictionary + bigram LMs, no rebuilds) and tests which combination
and order produces the best bilingual homophonic writing.
`joint = sound × EN-coherence × FR-coherence`.

## Result: non-cognate chain wins (cognates dropped)

| config | best joint | example |
|---|---|---|
| **chain, NON-COGNATE** | **0.56** | EN "to inform or in" / FR "tout formes or in" |
| chain, cognate | 0.23 | (small pool, sparse chains) |
| Round Rabbit → chain | ≤ 0.23 / empty | theming collapsed the pool |

**Non-cognate is the method.** Per the call, cognates are dropped: the
sound+meaning subset is tiny (235 of 4,220 pairs) and composes *worse*, not
better — too few pairs to build fluent lines. Pure homophony (frame-evasion, the
actual art) is both the goal and the higher-scoring path.

## What works, in what order

1. **Pool = v7-integrated, non-cognate, score ≥ 0.80** — the large pool (3,985
   single-word pairs) is what gives the composer diversity to find fluent chains.
2. **Chain-compose** maximizing `sound × EN-coh × FR-coh`, no repeated words —
   the writing step. Best lines reach joint 0.56 with EN-coherence 0.87.
3. **Phoneme-carve verify** the winning lines (the whole-line carve as the final
   acoustic check) — run only on the top few, to save compute.

## What did NOT help (yet)

- **Round Rabbit theming** narrowed the pool too far on this branch's meaning
  graph (no MUSE), collapsing chains. It needs rich meaning edges to help; today
  it hurts. Defer until the meaning layer is grown (MUSE / multilingual encoder).
- **Cognate restriction** — dropped, as above.

## The limiter (unchanged)

FR-side coherence tops out ~0.68 (bigram LM). The lines read as fluent *word
sequences* on both sides, not full sentences — the real L2 model is still the one
gating component. But the working recipe is settled: **large non-cognate pool →
chain-compose on both-sides coherence → carve-verify the winners.**
