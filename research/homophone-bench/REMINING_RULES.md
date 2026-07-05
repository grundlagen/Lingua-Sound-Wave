# Re-mining rules from the June 11–12 commits (useful for v6)

Surveyed the June 11–12 re-mining machinery; the disciplined gates there are worth
carrying into v6 (and the live engine). Sources: `phonetic_decoder.py` (v4 augment,
Jun 11), `fragments.py` + the chain/weave layer (Jun 12), `learn_costs.py`.

## 1. The augment acceptance gates (the core re-mining rule)

`phonetic_decoder.augment_dictionary()` / `reverse_augment()` decode every English
word whose best existing score < 0.90 and keep a re-mined entry ONLY if:

```
similarity >= 0.90  AND  words >= 2  AND  coverage >= 0.85
AND  expensive_deletions == 0  AND  max_substitution <= 0.45
tier = S if (max_substitution <= 0.25 AND coverage >= 0.90) else A
```

These are far stricter than v6's `coverage >= 0.60`. The key disciplines v6 was
missing: **`expensive_deletions == 0`** (no licensed-deletion abuse) and
**`max_substitution <= 0.45`** (no single bad phoneme swap carrying an entry).

**Applied to v6 (this commit):** each entry now carries a **`gold`** flag = passes
those three gates. So v6 keeps its filler/poetry material (which *needs* licensed
deletions) but marks the subset that meets v5's strict word-for-word bar — the
re-mining rule used as a quality label, not a blanket cut.

Note: the gold gate and the filler carves are in tension by design — filler carves
(only→on lie) use licensed deletions, so they are *not* gold. That's correct: gold
= clean word-for-word; filler = composition scaffolding. v6 holds both, tagged.

## 2. Only re-mine the weak (don't re-label the strong)

The augment skips any word already at score ≥ 0.90. Re-mining targets the gaps —
the honest-negative lesson (`b5a985a`): growth comes from decoding NEW entries with
a better matcher, not re-labelling existing topology. v6's `novel` flag mirrors
this (entries absent from v5).

## 3. The flywheel (certify → densify → re-grow)

The June 12 chain layer: certified alignments → `learn_costs.py` tightens the
substitution/gap table (AUC 0.989 → 0.994) → re-run the decoder with the learned
costs + bigger trie → new entries → denser graph → re-weave. The growth step is
**re-mining with the improved matcher**, applied to generation, not recycling.
This is exactly what the live carve engine now does per query.

## 4. The fragments generative probe (Jun 12)

`fragments.py` chains attested chunk EN-sides into candidate English words, looks
them up in the pronunciation lexicon, and re-mines a French side — gated at
`similarity >= 0.88, coverage >= 0.85, expensive_deletions == 0,
max_substitution <= 0.30` (even stricter). The proposal-grammar + arbiter shape.

## What to carry forward

- **Adopt the augment gates as v6's gold tier** (done — `gold` flag).
- **Recompute `learn_costs.py` from the integrated v5+v6 certified alignments** so
  the matcher tightens further before the next re-mine (and pair it with the
  cycle-consistency objective — label-free).
- **The live engine already embodies the flywheel** (decode + arbiter per query);
  the static gates make its *kept* output cleaner.
- Keep filler/poetry material separate from gold — both are needed, the gates tell
  them apart.
