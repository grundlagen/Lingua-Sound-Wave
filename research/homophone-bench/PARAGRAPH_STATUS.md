# How close is paragraph production? (measured)

Periphrastic chain translation (`periphrastic_translate.py`) on an 11-content-
word paragraph, *"the pale moon lights the quiet sea and a small bird sings
of love in the cold night"*:

| metric | value |
|---|---|
| mean sound similarity | **0.96** |
| mean meaning (anchor) | 0.58 |
| strong-sound (≥0.85) | **11/11 = 100%** |
| dual-strong (sound ≥0.85 **and** meaning ≥0.5) | **7/11 = 63%** |

## Reading the number

**Homophonic paragraph production is already solved (100%).** Every content
word gets a high-sound French rendering, so a whole paragraph that *sounds*
like the English is producible today — that is the Van Rooten goal, met at
paragraph length.

**Dual (sound + meaning) production is ~63% per word** and climbing through
three mechanisms now in place:
- direct loanish hits: `love→love`, `lights→light` (1.00 meaning)
- synonym transfer: `tiny→petits` (0.92), `small→soft`, `cold→cool`,
  `pale→pâli (pallid)`
- description transfer: `she→femelle` (via "female")
- periphrasis as last-resort rescue: `night→soft night→sauvent natte`

The 37% that carry sound but weak meaning (`moon→des riment`, `bird→sauvent
bête`) are the words whose synonym/description set didn't contain anything
the French lexicon could sound-match. Two unlocks raise this:
1. **LLM paraphrase** (measured +0.25–0.29 sentence semantics earlier) —
   replaces Datamuse's flat synonym list with fluent, context-aware
   rephrasings the decoder can carve.
2. **Re-mine cycle** (learned costs + CMUdict trie, queued) — more French
   reachable per English sound stream = more words clearing the meaning bar.

## Honest ceiling

Per-word dual translation is near its practical limit. The remaining gap is
**sentence-level fluency** — the French side reads as word-salad because
word order follows the English sound stream (inherent to homophony) and the
decoder optimizes per-word, not per-line. Closing that is a generation
problem (arrange/select for both-language grammar), i.e. the LLM-judge layer,
not a data problem. The data layer is essentially done.

## Form polish applied
- poetic periphrasis demoted to rescue-only (stops "soft X" beating real
  synonyms)
- Datamuse free API (no key) for synonyms + glosses, fully cached offline in
  `api-cache.json` so reruns need no network
