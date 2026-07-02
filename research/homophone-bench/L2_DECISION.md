# L2 decision: embeds or no? — No. Trigram alone.

Decided by AUC on held-out OpenSubtitles French (never seen by the LM), two
honest tests (`l2_bakeoff.py`):

| L2 | A: real vs shuffled (grammar) | B: real vs salad (sense) |
|---|---|---|
| bigram | 0.666 | 0.952 |
| **trigram** | **0.912** | **0.994** |
| embed (MiniLM cohesion) | 0.500 | 0.784 |
| tri×embed | 0.889 | 0.992 |

- **Embeds are word-order-blind**: 0.500 = coin flip on shuffled sentences.
  Sentence embeddings measure topic, not syntax — they cannot be the fluency
  model. Even the product only dilutes the trigram.
- **Embeds keep exactly one job**: the MEANING channel (semantic_cosine on the
  dual pairs), where topic is the question.
- **Trigram is the L2**: `trigram_lm.py`, 18.7M tokens, wired into
  `final_verse.py`. The old bigram was the documented ceiling (0.666 on
  grammar); the trigram nearly closes it (0.912).

## Calibration against the canon (Mother Goose)

Scoring the one fair-use gold line (`gold-dual-readings.tsv`, van Rooten's
Humpty Dumpty) against alternatives:

| candidate | combo | prosody | L2 | joint |
|---|---|---|---|---|
| van Rooten (gold) | 0.58 | 0.79 | 0.31 | 0.143 |
| our engine carve | 0.54 | 0.79 | 0.52 | 0.224 |
| literal translation | 0.52 | 0.72 | 0.58 | 0.218 |
| fluent decoy | 0.40 | 0.69 | 0.59 | 0.160 |

Two lessons:
1. The judge separates gold from decoy (0.58 vs 0.40 combo, 0.79 vs 0.69
   prosody) ✓.
2. **Rooten-class art lives at combo ≈ 0.55–0.70, not 0.85+.** The canon trades
   sound precision for full French coherence. So the engine has TWO operating
   regimes: tight verse (tiles, combo ≥ 0.8 — the FINAL_VERSE gallery) and
   Rooten-class (whole-line carve, combo 0.55–0.70, L2-maximised).

## The Rooten-regime deliverable (ours, machine-selected from candidates)

> **EN**: Humpty Dumpty sat on a wall; Humpty Dumpty had a great fall
> **FR**: *Un petit, un petit, cette âme au vol ; un petit, un petit, a des
> regrets, folle.*
> ("A little one, a little one — this soul in flight; a little one, a little
> one has regrets — mad.")

combo 0.56 / 0.58 (the gold band), joint 0.268 / 0.276 — **above the gold
line's own joint (0.143)** because the French stays coherent. Selected by
combo×L2 over hand-written candidates; not a word of van Rooten's rendering
reused.

## JOKER CLEF status

Checked: Zenodo (papers only), GitHub (participant repos outside session repo
scope; API 403), HuggingFace (no mirror). The corpus is registration-gated —
needs a CLEF signup by Rupert to become the held-out wordplay eval. Until then
Mother Goose gold + held-out subtitles are the external checks.
