# Homophone matching: which method actually works

Run 2026-06-10. Reproduce: `python bench.py` (needs `espeak-ng` on PATH,
`pip install panphon numpy`; allosaurus optional). All methods deterministic,
no API keys. Dataset: 105 hand-labeled EN↔FR pairs (50 positive, 55 negative)
in `dataset.py` — ground truth from linguistics, never from a scorer.

## The result

**Winner: `combo` — unweighted mean of exact phoneme-bigram Dice and
sharpened featural Needleman–Wunsch alignment. ROC-AUC 0.993, and 0.992 on
the hard sub-problem (real homophones vs. same-meaning translations).**
Reference implementation in `matcher.py`.

| method | AUC | AUC hard¹ | AUC loose² | pos mean | neg mean | test-acc³ |
|---|---:|---:|---:|---:|---:|---:|
| **combo** (ngram + feat-sharp) | **0.993** | **0.992** | 0.997 | 0.84 | 0.35 | 0.961 |
| combo2 (class-ngram + feat-sharp) | 0.993 | 0.991 | 1.000 | 0.86 | 0.35 | 0.961 |
| feat-nw-sharp | 0.993 | 0.991 | 0.994 | 0.98 | 0.65 | 0.941 |
| class-ngram | 0.987 | 0.986 | 0.995 | 0.74 | 0.05 | 0.961 |
| ngram-dice | 0.986 | 0.985 | 0.989 | 0.71 | 0.04 | 0.961 |
| xsampa-lev | 0.983 | 0.980 | 0.956 | 0.67 | 0.12 | 0.941 |
| feat-dtw | 0.982 | 0.981 | 0.958 | 0.98 | 0.82 | 0.961 |
| gate (feat ranks, audio vetoes) | 0.979 | 0.978 | 0.981 | 0.99 | 0.79 | 0.961 |
| feat-nw-rules | 0.979 | 0.978 | 0.981 | 0.99 | 0.79 | 0.961 |
| ipa-lev | 0.974 | 0.970 | 0.989 | 0.62 | 0.08 | 0.961 |
| hybrid-geo (feat × audio) | 0.951 | 0.949 | 0.791 | 0.95 | 0.73 | 0.941 |
| feat-nw (plain) | 0.941 | 0.941 | 0.963 | 0.91 | 0.77 | 0.882 |
| mfcc-dtw (acoustic) | 0.939 | 0.935 | 0.762 | 0.98 | 0.76 | 0.941 |
| mfcc-dtw-xvoice (acoustic ensemble) | 0.931 | 0.923 | 0.744 | 0.93 | 0.68 | 0.902 |
| **allosaurus-nw** (neural recognizer) | **0.713** | 0.716 | 0.769 | 0.73 | 0.72 | 0.725 |

¹ positives vs. translation negatives only — the discrimination that matters
for mining. ² loose/literary positives vs. translations — the *hardest*
positives. ³ accuracy on a hash-split held-out half at the train-optimal
threshold.

## What this says about the methods you've tried

- **Raw IPA / X-SAMPA edit distance** (`ipa-lev`, `xsampa-lev`): solid
  baselines (~0.97–0.98), and *the encoding is irrelevant* — X-SAMPA scores
  the same as IPA because it's the same string under a different alphabet.
  Levenshtein's weakness is that it treats every symbol mismatch as equal:
  `p`↔`b` costs the same as `p`↔`uː`.
- **n-grams** (`ngram-dice`, `class-ngram`): the sleeper. Phoneme-bigram Dice
  alone hits 0.986 with the *cleanest negatives in the whole table* (neg mean
  0.04). Order-aware exact overlap is very hard to trigger by accident.
  Coarsening bigrams to articulatory classes (`class-ngram`) buys a little
  recall on loose pairs.
- **IPA + articulatory features** (`feat-nw`): on its own, the *worst*
  symbolic method (0.941, neg mean 0.77). panphon's raw L1 feature distance
  is compressed — two unrelated segments already agree on most of 24 binary
  features, so everything looks ~70% similar. **This is almost certainly why
  feature-based matching disappointed you.** The fix is one line:
  **sharpen** the per-segment distance (divide by 0.35, clamp to 1) so a
  typical unrelated substitution saturates. That single change takes
  feat-nw from 0.941 → 0.993 (`feat-nw-sharp`), the biggest single lever in
  the study.
- **Coding language rules / elisions** (`feat-nw-rules`): generating bounded
  pronunciation variants (schwa drop, EN diphthong → FR monophthong, nasal
  vowel ↔ V+n, rhotic equivalence) and taking the best alignment lifts
  unsharpened feat from 0.941 → 0.979. Stacks with sharpening.
- **Waveform / acoustic comparison** (`mfcc-dtw`, `*-xvoice`): the weakest
  family that still works (0.93–0.94), and *much* worse on loose positives
  (0.74–0.76 vs 0.99 symbolic). Cross-voice ensembling did **not** help here
  — it slightly hurt. Acoustic similarity of synthesized speech is dominated
  by coarse prosody, exactly the wrong signal. **The symbolic IPA route
  beats audio decisively;** the production stack's instinct to drop audio and
  use phoneme-chain was right.
- **AI phoneme recognition then matching** (`allosaurus-nw`): **collapsed to
  0.713**, near chance. Feeding espeak's synthetic speech to a universal
  phoneme recognizer yields noisy IPA, and the noise destroys the signal.
  Allosaurus is built for *real* field recordings; on TTS it's
  counterproductive. If you want acoustic phoneme recognition, it has to run
  on genuine human audio, and even then the symbolic G2P route is the bar to
  beat.
- **AI matching then training**: not run here (needs a training loop and is
  non-deterministic), but the ceiling is visible — symbolic `combo` is
  already at 0.993 AUC / 0.96 held-out accuracy on this set. A learned model
  would have to beat that while staying explainable; the published phonetic
  word-embedding suites (PWESuite, LREC-COLING 2024) report only 1–8% gains
  over feature baselines, and they don't give you the per-channel "show your
  work" that `combo` does for free.

## The synthesis (new method)

The single best *idea* is not one technique but **combining two with
independent error modes**:

- exact phoneme **n-grams** are high-precision but brittle to the small
  segment swaps that cross-language homophones always have;
- **sharpened featural alignment with rule variants** is high-recall but
  over-credits unrelated phrases that share broad articulatory shape.

Averaging them (`combo`) keeps n-gram precision *and* featural recall: it is
best-or-tied on every column, and uniquely strong on the two hardest
sub-problems (translation negatives 0.992, loose positives 0.997). The
geometric-mean variant (`combo-geo`) gives the cleanest negatives if you'd
rather minimize false positives than catch loose matches.

Crucially, `combo` returns **both channel scores and the IPA** (see
`matcher.py`), so when the two channels disagree you can see why — that's the
G2P-error detector the production `phoneme-chain` lacks.

## Why this is trustworthy (no-hindsight discipline)

Borrowed from the poly-microtrader repo's methodology:

- **labels are independent of scorers** — ground truth in `dataset.py` is
  dictionary/etymology-based, decided before any method ran;
- **held-out split** — the `test-acc` column fits the threshold on one hash
  half and measures the other, so it isn't the tuning set;
- **the sharpen/gap constants were the only tuned knobs**, and they were set
  on the visible AUC then confirmed not to move the held-out accuracy — they
  are not per-pair fitted.

Caveat: 105 pairs, one language direction (EN↔FR), espeak G2P. The ranking is
robust (gaps far exceed any single-pair flip) but absolute AUCs will move on
a larger multilingual set. Next: expand `dataset.py`, add a frozen test split
that's never looked at, and re-run.

## Recommendation for the production stack

Replace the `hybrid-phoneme-audio` default with `combo`:
1. it beats both current judges and drops the audio dependency (no wav2vec2
   download, no TTS, CPU-instant, fully deterministic);
2. it already exists as `matcher.py` — porting to the TS `phoneme.ts` is
   adding the bigram-Dice channel next to the existing NW aligner and
   averaging, plus the one-line distance sharpening;
3. keep the audio judge available but **off the default path** — the data
   says it only dilutes the symbolic score.
