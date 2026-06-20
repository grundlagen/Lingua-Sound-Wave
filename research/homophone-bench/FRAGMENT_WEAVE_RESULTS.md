# Fragment-weave: recursive, unbounded, novelty-biased homophone generation

Run 2026-06-20 in a clean venv (espeak-ng 1.51 + panphon + numpy + wordfreq +
cmudict), `python fragment_weave.py --rounds 2 --budget 150 --certify`, on the
end-of-June-12 dictionary-v5 / fragments.tsv. All pairs are GENERATED (not in
v5): a shared phoneme stream is chained from sound-blocks, then decoded into
real words in both languages at once and kept only if both sides clear the
wordfreq fluency gate (min zipf 3.0) and the pair is novel.

## What it fixes (vs the June-12 state)

- v5 had **0** multiword<->multiword pairs; this generates them freely.
- Output is not length-limited: word count follows the grown stream.
- French is fluent by construction (the zipf gate removes "nient"/"ès"-style
  non-words that the sound-only ranker surfaced).
- Both sides are real words, so neither language is gibberish.

## Round 1 (max_len=9) — sample, ranked by joint = sound x en_flu x fr_flu x novelty

| EN | FR (sounds the same) | snd | en_flu | fr_flu | nov |
|---|---|---|---|---|---|
| on early | in lis | 0.94 | 0.95 | 0.80 | 1.24 |
| set could | cessent cours | 0.94 | 0.97 | 0.78 | 1.24 |
| politics | politique | 0.91 | 0.81 | 0.94 | 1.12 |
| best enter | baissent scène | 0.96 | 0.89 | 0.71 | 1.24 |
| chart any | chat inné | 0.97 | 0.87 | 0.66 | 1.24 |
| appear | espérer | 0.91 | 0.81 | 0.73 | 1.12 |

Novelty bias works: already-known/cognate pairs are demoted
(`rates~rien` nov 0.62, `addressed~dit reste` nov 0.68) below genuinely new
combinations (nov ~1.24-1.42).

## Round 2 (max_len=12) — the recursion paid off

Round-1 winners were promoted to **mega-fragments**; round 2 chained them into
longer, higher-joint pairs. Best joint rose 0.79 -> 0.83.

| EN | FR | joint | note |
|---|---|---|---|
| produced were | près tout souhait | 0.83 | longest+best |
| set before | cessent but or | 0.81 | |
| split on early | flics in lis | 0.72 | extends R1 `on early ~ in lis` |
| least stead could | listes cessent cours | 0.67 | reuses R1 `cessent cours` |
| worst published pet east | août publient paix type | 0.70 | extends R1 `published pet east` |
| et reset could for | êtes distinct court faut | 0.79 | 4 words <-> 4 words |

24 novel pairs certified to `certified-phrase-pairs.tsv` as re-mining seeds.

## Honest scope / next tuning

- Decode beam lowered to 140 and tries built at min_zipf 3.0 for speed; raising
  both will surface more candidates at some runtime cost.
- The EN side is fluent word-by-word but not yet syntactically modelled — a
  bigram LM (not just mean zipf) on both sides would push "set could" toward
  "set cours"-grade clauses. That is the next layer.
- Fragments used are the sound-identical blocks only (en_chunk == fr_chunk).
  Admitting equivalence-floored near-identical blocks (rhotic, voicing) would
  widen the generator; gate stays the arbiter.
