# What the bigram LM does (and what it reveals)

Run 2026-06-20. `python bigram_lm.py /tmp/corpus` then `python lm_compare.py`.
Word-bigram model, stupid-backoff, trained on public-domain running text:
EN = Pride & Prejudice, Sherlock, Moby Dick, Frankenstein (528,734 tokens);
FR = Le Comte de Monte-Cristo, Candide (505,771 tokens). Backs off to wordfreq
for OOV. The point: replace mean-zipf fluency (which says any common words in
any order are fine) with one that knows "in the sea" >> "set could".

## Calibration (per-word logprob -> fluency in [0,1])

| phrase | lp/word | fluency |
|---|---:|---:|
| in the sea (EN) | -3.39 | 1.00 |
| on the moon (EN) | -4.53 | 0.95 |
| dans la mer (FR) | -4.62 | 0.94 |
| il est (FR) | -4.76 | 0.93 |
| set could (EN salad) | -7.54 | 0.68 |
| chart any (EN salad) | -9.24 | 0.52 |
| cessent cours (FR salad) | -12.42 | 0.23 |
| chat inné (FR salad) | -13.78 | 0.11 |

The model separates real phrases (~0.9-1.0) from common-word salad (~0.1-0.5)
cleanly -- exactly the axis mean-zipf was blind to.

## Re-ranking the SAME 350 generated pairs: zipf vs bigram

Biggest losers under the bigram LM (zipf loved them, the LM rejects them):

| pair | zipf joint | LM joint | why |
|---|---:|---:|---|
| politics / politique | 0.72 | **0.06** | isolated rare-ish tokens, no fluent context |
| set could / cessent cours | 0.77 | **0.16** | "cessent cours" is not French (frL 0.23) |
| best enter / baissent scène | 0.66 | **0.08** | "baissent scène" not a bigram (frL 0.14) |
| worst were / août souhait | 0.71 | **0.14** | "août souhait" salad (frL 0.23) |
| published pet east / publient paix type | 0.65 | **0.08** | three-word salad both sides |

## The real finding (honest)

The bigram LM is the right scorer, but its **top** re-ranked pair only reaches
0.40 and most sit at 0.16-0.22. That is the signal: **the fragment generator is
producing common-word salad, not phrases**, and mean-zipf was giving it false
confidence (0.6-0.79). Re-ranking after the fact can only pick the least-bad of
a salad pool.

So the bigram's true value is not as a re-ranker -- it is as a **generation
objective**. The next layer is to put the bigram term *inside* the search:

1. add an LM score to `phonetic_decoder.decode`'s beam so the decoder prefers
   word sequences that are both sound-true AND grammatical (sound x LM), not
   sound-then-filtered;
2. seed `fragment_weave` chains from frequent corpus bigrams instead of uniform
   block sampling, so streams start life near fluent phrases;
3. keep the gate, keep novelty -- the LM just becomes the thing the beam climbs.

Caveat: the corpus is literary (19th-century novels), so colloquial phrases
("tout doux" -> 0.39) are under-scored; a modern subtitle/web corpus would
calibrate the FR side better. The relative ranking signal holds regardless.

`fragment_weave.py --lm` already swaps the re-ranking objective to the bigram;
the deeper win is item (1) above.
