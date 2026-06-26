# Best-practice self-teaching for the homophonic engine

How to make the system learn well — gold-standard-anchored, embedding-grounded,
generative over phoneme streams, with Claude in the loop. This is the design the
training/reward code follows.

## 1. Gold standards are the spine

The mined **sound∧meaning gold** (prosody ≥ 0.70 AND semantic-cosine ≥ 0.45 —
`*-remined.tsv`, GOLD tier) is the reference. Three distinct uses, never mixed:

- **Held-out eval** — freeze a gold slice; *never train on it*; measure the model
  against it every round (does it reproduce/approach the gold carves?).
- **Few-shot / SFT seeds** — the train side of gold are the warm-start targets.
- **Reward calibration** — set thresholds so gold scores high and known-bad scores
  low (the `experiment.py` separation metric is exactly this).

Best practice: a gold set is *small, clean, frozen, and external to training*.
Grow it by remining the **best dataset** (full v7 dictionary, JOKER CC-BY corpus),
not by hand.

## 2. The reward = two calibrated channels

`reward = sound × meaning`:
- **sound** = stress-weighted **prosody** (espeak stress, EN-fall vs FR-rise,
  unstressed cheaped out) — perceptual homophony.
- **meaning** = **semantic-cosine** of the multilingual embedding (source vs
  output) — kept as a *method* now; fold into the reward only deliberately.

Embeddings do double duty: the meaning reward, **gold-exemplar retrieval**
(nearest gold pair to a query), and **sense disambiguation** (cluster a word's
translations).

## 3. Generate over PHONEME STREAMS (the right output space)

Don't generate French text blind. Best practice: the model emits a target
realization **conditioned on the source phoneme stream**, so the output is
phoneme-aligned by construction (the carve). Two viable forms:
- **seq2seq carve** — input = English IPA stream, output = French words whose IPA
  reconstructs it (supervise with our carves; reward with prosody match);
- **constrained decode** — the phonetic decoder restricts the LM to French words
  that fit the next phoneme chunk (sound guaranteed; LM picks the fluent/meaningful
  one). The phoneme stream is the contract; meaning + fluency are the free choices.

## 4. Contrastive learning (the JOKER lesson)

From the reward, mint **preference pairs**: gold/high-reward (positive) vs
random/low-reward French for the same English (negative). Train a small **reward
model** on these so per-sample scoring is fast and learned (no per-sample LLM
call), and/or DPO the generator on them. Hard negatives = *sound-alike but
wrong-meaning* (the cases the plain scorer over-rates).

## 5. Self-improvement loop (expert iteration)

`generate (best-of-N) → score with reward → keep gold-quality → SFT → repeat`,
raising `keep_thresh` over time. `run_continual.py` already does this; the upgrade
is to (a) gate "keep" on BOTH sound and meaning (gold definition), (b) eval on the
frozen gold each round, (c) stop chasing the local reward once gold-eval plateaus.

## 6. Claude in the loop (judgment where reward is blind)

Automated reward is blind to wit, register, and real coherence. So each round
pushes samples to `selflearn-status`; **Claude reviews them against gold**, flags
systematic failures (e.g. "function-word salad," "wrong-sense rung"), and proposes
reward/rule changes — committed to the branch for the next round to pull. The
council (DeepSeek/Nemotron) supplies linguistic clues; Claude curates and decides.
Human/Claude judgment is the outer loop; the reward is the inner loop.

## The pipeline, end to end

```
best datasets (v7 dict, JOKER) ──remine(prosody×meaning)──▶ GOLD set
        │                                                     │
        ├── frozen gold eval ◀───────────────────────────────┤
        ▼                                                     ▼
  SFT warm-start ─▶ generate phoneme-stream carves ─▶ reward(sound×meaning)
        ▲                                                     │
        │                  keep gold-quality ◀────────────────┤
        └──── SFT on self-bests ◀── Claude/council review ◀───┘
```

Everything additive and preserved: new files, frozen gold, branches — never an
in-place overwrite of working code.
