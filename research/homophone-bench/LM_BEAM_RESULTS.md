# Bigram LM inside the beam: results

Date: 2026-06-22.  Three components landed together:

1. **`phonetic_decoder.py`** — `decode()` now accepts `lm=BigramLM, lm_weight=0.35`.
   The beam key rewards word-boundary paths whose bigram log-probability exceeds a
   baseline of −6 nats/word (fluent) and penalises those below it (salad).  The
   LM adjustment is only active after the first complete word has been emitted so
   it does not bias single-word vs. multi-word splits at the start of the beam.

2. **`corpus_phrases.py`** — extracts the top-300 word bigrams per language from
   the trained BigramLM (both EN and FR), converts each to canonical IPA via
   espeak-ng, filters out language-fallback artefacts, and writes
   `corpus-phrases-{en,fr}.tsv`.  Output: 600 clean seed streams, avg 7.2 segs.

3. **`fragment_weave.py`** — `--phrases` loads the phrasebook; phrase seeds (full
   bigram IPA, 4–14 segs) are decoded DIRECTLY as candidate streams rather than
   chained as sub-blocks (too long).  Short seeds (≤6 segs) are also folded into
   the chaining pool at 2× weight.  `best_decode()` always passes the loaded LM
   to `pd.decode()`.

## Sample run (100 s, --lm --phrases, round 1)

117 novel fluent pairs, 31 multi-word.  Top results:

| EN | FR | snd | en_flu | fr_flu | joint |
|---|---|---:|---:|---:|---:|
| us if | **as if** | 0.88 | 0.87 | **1.00** | 0.73 |
| into | in tout | 0.88 | 0.80 | 0.76 | 0.57 |
| in some | in savent | 0.87 | **0.98** | 0.68 | 0.55 |
| but of | but savent | 0.91 | 0.91 | 0.64 | 0.51 |
| but love | man love | 0.90 | 0.80 | 0.70 | 0.47 |
| and love | in fleuve | 0.92 | 0.84 | 0.53 | 0.45 |

## What changed vs zipf-only (same budget, no LM)

Without LM the same 100 s window found 324 pairs (218 multi-word) with top joint
0.87 ("it steel / elles il").  The LM run finds fewer pairs (117) at somewhat
lower peak joint (0.73) but dramatically better FR fluency:

| metric | zipf-only | LM beam |
|---|---:|---:|
| avg fr_flu top-10 | 0.74 | **0.82** |
| fr_flu = 1.0 results | 0 | **1** ("as if") |
| "cessent cours"-grade salad in top-10 | present | absent |

The LM beam correctly rejects "cessent cours" (frflu 0.23) from top positions
while surfacing "as if" (frflu 1.00).  The trade-off: fewer total pairs because
the tighter beam key prunes salad-starting paths earlier, reducing the diversity
of explored IPA streams.

## Residual gap and next steps

- Phrase seeds are currently decoded independently; wiring them as CHAIN STARTERS
  (one seed + 1-2 short blocks appended) would grow the multi-word count while
  keeping phrase-level anchoring.
- The FR corpus is 19th-century (Monte Cristo, Candide); adding subtitle/web text
  would calibrate common colloquial bigrams ("c'est", "il y a") that are currently
  under-scored.
- "us if / as if" should be verified with espeak round-trip and added to
  certified-phrase-pairs.tsv.
