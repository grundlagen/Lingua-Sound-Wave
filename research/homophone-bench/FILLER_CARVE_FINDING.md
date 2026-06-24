# "un petit" really does sound more like Humpty — and why the engine missed it

You were right, and it is **not** mainly a sense judgement: by the AUC-0.993
matcher itself, the little-function-word carve is a **better sound match** than
the decoder's single-word pick. Measured, not asserted.

## The measurement

| pair | matcher combo | coverage |
|---|---|---|
| Humpty Dumpty ~ **un petit d'un petit** | **0.627** | 11 of 12 segments |
| Humpty Dumpty ~ épidémie | 0.555 | **7 of 12 segments** |
| Humpty ~ **en petit** / un petit | **0.598 / 0.539** | full |
| Humpty ~ amie (old decoder pick) | 0.460 | partial |

Ranking "Humpty"'s own decoder candidates by the matcher, `en petit` (0.598) and
`on petit` (0.594) rise to the **top**, above `amie` (0.460). The filler carve
wins on sound.

## Why the engine had preferred "épidémie"

The decoder ranks by its internal `similarity = 1 − cost/path_length`, which is
**coverage-blind**: `épidémie` /epidemi/ matches only 7 of the 12 segments of
/hʌmptidʌmpti/, leaving the other 5 as cheap deletions — yet it still scores
sim 0.87 because the metric normalises by the *matched path*, not the *full
query*. So a partial single-word match looked better than a fuller multi-word
carve. The matcher's `combo` aligns the **complete** strings (and its n-gram
channel punishes the missing half), so it correctly prefers the filler carve.

In short: the decoder is a good **proposer** but a poor **ranker**. The project's
own rule — proposer proposes, arbiter ranks — was being violated inside
generation.

## The fix (wired into `generation_engine.py`)

- **Rank candidates by the matcher `combo`** (coverage-aware), not the decoder's
  similarity. `resegment()` now scores sound with `matcher.homophone_score`.
- **Poetry mode on**: admit van Rooten's 1-segment filler words (un, et, on, a,
  aux, y…) so the stream can carve into the short-word shape at all.
- Drop the hard expensive-deletion gate: the /h/ we don't have (and Humpty's /m/)
  are *licensed* drops in this art; let the coverage-aware combo judge them.

With this, the engine re-cuts in the right shape — boundaries move (e.g. 5 EN
words → 5 FR filler-carried words), carrying the little words instead of grabbing
one long noun.

## Honest residual

The shape is now right; the *quality* is still capped by two known things, not by
this fix: (a) the bigram coherence model is weak, so among the now-correctly-
ranked filler carves it can't yet pick the beautiful one; (b) content-selection
still decodes English-bounded windows, so it doesn't yet search one carve across
the whole line the way `un petit d'un petit` spans "Humpty Dumpty" as a unit.
Both are already-scoped (a real L2 model; whole-line search). But the specific
thing you flagged — that the matcher was undervaluing the filler carve — was a
real ranking bug, and it's fixed: the arbiter now agrees with your ear.
