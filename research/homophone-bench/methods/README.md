# Methods — Experimental Approaches

These files represent genuinely different algorithmic strategies that are not
part of the canonical pipeline. They are preserved because each explores a
distinct approach to homophonic generation. None are "wrong" — they're different
paths through the same problem.

## homophone_writer/ — Multi-Channel Generation

Four iterations of a universal homophone writer. Each represents a different
strategy for picking the best French rendering of an English input.

| File | Generation strategy |
|---|---|
| `homophone_writer.py` (v2) | 11 channels (cognate, dual, ladder, glue, babel, homophone-class, synonym-chain, metaphor, carve) + 7-channel scoring + climb/improve |
| `homophone_writer_v3.py` | 9 channels, training corpus build, iterative self-improve |
| `homophone_writer_v4.py` | 11-channel decay reduction, strict judge, triple-verify, training corpus |
| `homophone_writer_v5.py` | Cross-accent dual-ear generation (EN ear + FR ear via speech synthesizer), training corpus build from v7 + strict_gold + chain-web |

Key insight across versions: the v5 cross-accent approach (matching through
two ears rather than direct IPA edit distance) is fundamentally different from
the earlier channel-based approaches. Neither approach has been rigorously
compared head-to-head.

## topological_flow/ — Persistent Homology Approaches

Mathematical formulations using graph theory and algebraic methods.

| File | Approach |
|---|---|
| `topological_flow.py` (v1) | Bipartite matching + filtered graph (persistent homology) — sound and meaning as dual matchings |
| `topological_flow_v2.py` | Refined bipartite flow with cost optimization |
| `topological_flow_v3.py` | Polysemy + periphrastic + many-to-many algebraic flow. Homophone classes as cover amplifiers. Poetic drift edges. |
| `topological_sequence.py` | Sequence-based topological constraints |

Key insight: v3's many-to-many formulation (x EN words → y FR words, not 1:1)
is the most general. The homophone-class cover amplifier insight (one sound
unlocks 5+ meaning sets) is unique to this approach and not present in the
canonical pipeline.

## poets/ — Generative Poem Writers

| File | What it does |
|---|---|
| `recursive_poet.py` | Multi-pass recursive expansion over typed graph, beam search, semantic/rhythm QC. Most complete poem output. |
| `constrained_poet.py` | Constraint-based poem generation |
| `web_poet.py` | Themed web-walk poet (exploration) |
| `dual_poet.py` | Dual-reading poem generation |

## composers/ — Line Composition Strategies

| File | What it does |
|---|---|
| `compose_lots.py` | Multi-granularity composer with fragment lots. This is the canonical composer — moved back to main directory. |
| `beauty_compose.py` | Beauty-weighted composition |
| `compose_smoke.py` | Smoke test composer |
| `dual_scale_composer.py` | Dual-scale composition |
| `full_dict_composer.py` | Full-dictionary composition strategy |

## Why These Aren't Canonical

Each approach was built to explore a specific hypothesis. They weren't abandoned
because they failed — they were succeeded by a different exploration. The
canonical pipeline (`compose_lots.py` + `mapping_web.py` + `round_rabbit.py`)
represents the most thoroughly validated path, but the insights from these
methods (cross-accent scoring, many-to-many flow, homophone-class amplification)
are valuable and may be reintegrated.
