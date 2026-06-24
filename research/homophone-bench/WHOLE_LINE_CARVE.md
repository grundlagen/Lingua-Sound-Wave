# Whole-line carve: the full pipeline generates a spanning carve end to end

`whole_line_carve.py` decodes a line's **entire** phoneme stream in one pass
(no English-word windows), through the v6 poetry trie (1-segment fillers),
ranked by the coverage-aware matcher arbiter — and with one more fix it produces
French carves that span the whole line in the homophonic-verse shape. All French
below is the engine's own generated output on public-domain English input.

## The missing piece: force coverage in the search

The whole-line decode first collapsed to **low-coverage** carves — "Humpty Dumpty
sat on a wall" came back as a 2-word partial covering 43% — because the beam's
deletion costs are cheap, so it would rather **skip half the stream** than force a
long French sequence. Coverage-aware *ranking* can't fix that: the beam never
*generates* the full carve to rank.

Fix (`force_coverage()`): raise the deletion penalty (keep /h/ cheap — French has
no /h/), so the beam must cover the stream. Effect on the long line:

| | coverage | carve |
|---|---|---|
| default gaps | 0.62 | 2-word partial ("on bénéfice…") |
| deletion penalised | **0.95** | 7-word full carve |

## End-to-end results (public-domain source, engine-generated French)

| EN line | engine carve | cov | coh | fillers |
|---|---|---|---|---|
| Humpty Dumpty | un petit un petit | 0.92 | 0.85 | 2 |
| Humpty Dumpty sat on a wall | un petit un petit et on vole | 0.90 | 0.79 | 4 |

The engine, from scratch, re-cuts "Humpty Dumpty" (2 English words) into "un petit
un petit" (4 French words) — boundaries falling on French words, little filler
words carrying the meter — the exact shape of the art form. It converges on the
known style without being given it.

## Honest limitations

- **Coverage-forcing is a blunt global knob.** It nails the Humpty lines but
  over-forces others: "Hickory dickory dock" → a single long word
  "incrédulité" (coherence 0.09), and "Jack and Jill" finds no ≥75%-coverage
  carve at all. A per-line or learned deletion penalty (tie it to stream length /
  vowel density) is the right version.
- **Coherence is still bigram-grade.** The top carves are sound-true and
  French-shaped, with high bigram fluency, but "un petit un petit et on vole" is
  evocative-nonsense, not crafted verse. Selecting the *beautiful* full carve
  among the now-many full-coverage candidates is the L2-coherence model's job —
  the one gating component, unchanged.
- It works best where the English carves cleanly; lines dense in non-lexical
  words still need the synonym-swap layer (the future hook).

## What now stands, assembled

The pipeline is now whole and each piece is the session's:
`espeak G2P → continuous stream → resegmentation decoder (free boundaries, June 11)
→ v6 poetry trie (1-seg fillers) → coverage-forced beam → coverage-aware matcher
ranking → bigram coherence`. It generates spanning, filler-carried, boundary-
re-cut French for a whole English line — the operation a word-for-word dictionary
structurally cannot do. The remaining distance to *verse* is exactly the
L2-coherence model and a smarter (learned) coverage penalty — both scoped, neither
a dictionary problem.
