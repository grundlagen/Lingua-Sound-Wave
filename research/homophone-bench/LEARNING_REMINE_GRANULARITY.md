# Learning the coverage penalty; do we re-mine?; and granularity

Three questions, each tested.

## 1. The learning system (the coverage penalty) — built

`learned_coverage.py` replaces the blunt global deletion knob with a **per-line
autotuned** one: sweep the gap-scale, keep the carve that maximises the **dual**
objective (coverage-aware combo × bigram coherence), and fit a small model
`gap_scale ≈ a + b·stream_length` so a new line gets a one-shot scale.

Because the objective is dual (not raw coverage), it never picks the over-forced
single-long-word collapse. Results:

| line | learned scale | carve | dual |
|---|---|---|---|
| Humpty Dumpty | 1.6 | un petit un petit | 0.48 |
| Humpty Dumpty sat on a wall | 1.6 | un petit un petit et on vol | 0.42 |
| Little Jack Horner | 1.6 | let gendarmes | 0.20 |
| Jack and Jill | 2.8 | (poor) | 0.09 |
| Hickory dickory dock | 2.2 | (poor) | 0.04 |

The learner adapts and locks the clean lines to a gentle 1.6. It also **honestly
exposes** that Jack/Hickory have *no* good carve at any scale — their best dual is
~0.05, so the problem is not the penalty. (The fitted line is rough — 5 noisy
points — so per-line autotune is the real mechanism; the model is a warm-start.)

The next-deeper learn is the substitution/gap **costs** themselves — that machinery
already exists (`learn_costs.py`; the cycle-consistency objective in
`cycle_consistency.py`), so the learning ladder is: gap-scale (here) → per-segment
costs (learn_costs / cycle) → the L2 coherence model.

## 2. Do we need to re-mine? — No, not for generation

The whole-line carve engine decodes **live** off the lexicon trie + matcher +
fillers. It does **not** consume the static v5/v6 dictionary entries at all — it
re-mines *per query*. That is the "re-mine, not re-label" principle (the old
code's good ethos) now **embodied in the live engine** rather than frozen into a
file. So:

- **For generation: a static re-mine is not required.** The engine produces the
  carve from the lexicon on demand, with the arbiter ranking it — exactly what a
  re-mine pass did offline, now online.
- **The static dictionary still earns its keep, as three other things:** a fast
  **cache** of known-good word-for-word pairs (v5/reranked, near-ceiling); the
  **fragment source** (its alignments built `fragments.tsv`); and the
  **composition primitives** (`v6-fillers.tsv`). None of those is the generation
  substrate; they are supports.

Net: keep the good re-mining *principles* (proposer + arbiter, re-mine not
re-label) — they're already running live — and don't burn a full offline re-mine
for generation. Re-mine only to refresh the cache or the fragment index.

## 3. Granularity — how fine can the units go?

Your model is right: **v6 words are content words (≥2 phoneme segments) +
1-phoneme fillers.** Two clarifications, one tested.

- **The matcher already operates at single-phoneme grain.** Alignment is
  segment-by-segment (the decoder walks one phoneme at a time through the trie);
  only the *output units* are whole words. So matching is already as fine as it
  gets; there is no coarser alignment to refine.
- **The output floor is the real word — the 1-phoneme filler is the finest
  legitimate unit.** Below a real word you get non-words ("un p't"), which breaks
  the iron constraint that every output token be a real French word (the
  warblish principle: mimic the stream with *existing* words, not coined ones).
  So a ≥2-phoneme word **cannot** be "deconstituted" into smaller *output* pieces
  and stay legal — its phonemes can only be re-matched, which the decoder already
  does.
- **Finer-grain OUTPUT was tested and does not help.** Capping admitted words to
  ≤5 and ≤3 segments (forcing many short pieces) made the failing lines produce
  **no** ≥70%-coverage carve at all — the whole-word "incrédulité"/"cicatrices"
  was the *only* option, and removing it left nothing. Those lines fail for lack
  of any sound-matching French sequence, **not** for coarse units. Finer grain is
  not the lever for them.

Where finer grain *does* live is the **fragments layer** — sub-word phoneme chunks
(`st`, `ɹi`, `tɹ`) used to **compose** candidate streams / reach words, i.e. for
*generation and proposal*, not for emitting sub-word output. That layer is already
built (`fragments.py`); it composes chunks into whole words, never below.

### So what *does* fix Jack and Jill / Hickory dickory dock?
Not granularity and not the penalty — **the synonym-swap layer**. Those lines are
dense in non-lexical words (jack, jill, hickory, dickory) that have no French
sound-match at any grain. The fix is to change *what is said* (a near-synonym /
rephrasing whose phoneme string carves), which is the future layer already hooked
at the GAP verdict — the only remaining lever for the lines that genuinely don't
carve.
