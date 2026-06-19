# Homophone matching: insights and a path past the current plateau

Written 2026-06-10 from a cross-repo review (Lingua-Sound-Wave,
poly-microtrader, Quack-Coin-Core, KinetiCoach, Dig-Site-Identifier).
Code references are to `artifacts/api-server/src/lib/`.

> **⚠️ Superseded by the 11–12 June Python schema.** This note is a *pre-pivot*
> analysis of the TypeScript stack's acoustic/LLM judges. The project then
> dropped the audio channel entirely: `research/homophone-bench/RESULTS.md`
> shows the symbolic `combo` matcher at 0.993 AUC, beating every acoustic
> method, and the engine is now offline + deterministic (espeak-ng + CMUdict +
> Lexique G2P, `combo` + **learned** equivalence costs, `phonetic_decoder`
> trie/beam, `weave`, re-mining). The cross-voice / wav2vec2-espeak / cohort
> ideas below are kept for record only — the answer turned out to be "remove
> the acoustic channel," not "calibrate it."
>
> **Deliberately NOT used** (do not reintroduce): fuzzy / edit-distance
> matching (Levenshtein, difflib, rapidfuzz) — matching is featural + learned
> costs, growth is re-mining; plotting libraries (matplotlib / pyplot) —
> results are TSV + plain text; **epitran** for G2P — use espeak-ng + CMUdict
> + Lexique.

## Where the benchmark actually leaves us

The 8-pair benchmark (replit.md) is honest and its headline holds up under
code review: **phoneme-chain alone separates best** (96.8% mean on positives
vs 28.8% on negatives, +49.9pt), and both hybrids *lose* separation because
the acoustic judges inflate negatives.

The root cause is visible in `scoring.ts` itself: the wav2vec2 comments admit
that same-voice TTS pushes raw cosines into a 0.94–0.995 band, and
`distanceToSimilarity()` fights this by stretching the band. That's treating
the symptom. **The acoustic channel is currently measuring the voice, not the
phrase.** Every acoustic comparison synthesizes both sides with the same TTS
voice, so the dominant shared signal is speaker identity + channel, and the
phrase-level difference is a residual. No calibration constant fixes a
confound; you have to remove it from the protocol.

## The five insights

### 1. Break the voice confound at the protocol level (highest value/effort)

Synthesize the source with voice A and the candidate with voice B — **never
the same voice for both sides** — and repeat over K voice pairings, taking
the median similarity. Cross-voice scoring means the only thing the two clips
share is the phonetic content, which is exactly what we want to measure.
This should drop the negatives band (currently ~46.8% for the hybrid) far
more than it drops genuine homophones, because homophones survive a voice
change and "prosodically similar but unrelated" pairs don't.

Cheap to implement: `tts.ts` already synthesizes; this is a loop and a
median. Re-run the benchmark with cross-voice wav2vec2-dtw before investing
in anything heavier — the hybrid verdict may flip.

### 2. Make the acoustic judge speak IPA too

The deepest fix for the "two judges, incomparable scales" problem is to put
both channels in the same representation. wav2vec2 models fine-tuned for
*cross-lingual phoneme recognition* (the espeak-phoneme CTC heads, e.g.
`wav2vec2-lv-60-espeak-cv-ft`) emit IPA symbol sequences directly from
audio. That gives:

- an **acoustically-derived IPA string** to feed into the *same* weighted
  Needleman–Wunsch aligner in `phoneme.ts` (featural distances, equivalence
  classes, variant chaining — all reusable as-is);
- a cross-check on the LLM G2P: when LLM-IPA and acoustic-IPA disagree on
  the same clip, that's a G2P error detector, which is phoneme-chain's one
  acknowledged failure mode;
- a deterministic, non-LLM fallback (espeak-ng G2P) for cache misses.

This turns "symbolic vs acoustic" into "two ways of obtaining IPA, one
shared scoring function" — disagreement becomes diagnosable instead of
being averaged away.

### 3. Score relatively, not absolutely (borrowed from speaker verification)

Raw similarity is uncalibratable across methods, languages, and voices.
Standard fix from speaker-verification: **cohort normalization**. Score the
source phrase against N random distractor phrases under identical conditions
(same voices, same method), and report the candidate's z-score or percentile
within that cohort. A true homophone should be an outlier vs the cohort; a
"everything sounds 0.95" voice-floor day moves the whole cohort and cancels
out. This also makes tier thresholds (S/A/B in `tier-grader.ts`)
meaningful across language pairs.

### 4. Replace the geometric mean with an asymmetric gate

The benchmark's lesson is not "acoustic evidence is useless" but "symmetric
combination is wrong." The geometric mean lets a noisy acoustic judge drag
down good symbolic matches *and* prop up bad ones. Phoneme-chain has high
recall and one known failure mode (bad G2P); the acoustic channel is noisy
but its *confident lows* are informative. So:

- rank with phoneme-chain;
- let the acoustic judge only **veto/demote**, and only when it is
  confidently low across the multi-voice ensemble (e.g. median cross-voice
  similarity below a threshold);
- never let acoustic agreement raise a score above what phoneme-chain
  assigned.

This keeps the 96.8% positive performance intact while cutting exactly the
false positives the hybrid was built to catch.

### 5. The benchmark itself is the bottleneck (poly-microtrader discipline)

Eight pairs cannot rank six methods — the poem-benchmark report already
admits ±several points of jitter per run, which exceeds the gaps being
measured. Borrowing the backtesting discipline from poly-microtrader:

- expand to a few hundred labeled pairs: the 240-pair seed corpus
  (`seed-corpus.ts`) as positives, random translation pairs as negatives,
  and S/A/B reservoir tiers as silver labels;
- **freeze a held-out test set** and never tune on it — the calibration
  constants in `distanceToSimilarity()` and the wav2vec2 stretch band were
  tuned on the same anchor pairs used to report results, which is mild
  leakage of exactly the kind poly-microtrader's `leakage_check` exists to
  catch;
- persist versioned eval runs (method, constants, dataset hash, date) the
  way Dig-Site-Identifier records `eval_runs`, so "method X beats Y" is
  always attached to a dataset version and reproducible.

## One engineering note from porting the DTW

While porting `dsp.ts`'s DTW into Quack-Coin-Core, a real bug surfaced that
applies here too: cosine distance between two all-zero frames returns 1
(maximum distance) because of the zero-denominator guard, so two *identical
silences* count as a total mismatch. TTS phrases are silence-trimmed at the
edges but internal pauses (multi-word phrases, commas) produce near-zero
frames. Worth checking whether `dtwDistance` in `dsp.ts` penalizes phrases
with matching internal pause structure; the fix is two lines (both-zero →
distance 0).

## Recommended order

1. Cross-voice ensemble for the acoustic judges + re-run benchmark (days).
2. Benchmark expansion + frozen test set + eval-run logging (days).
3. Asymmetric gate replacing the geometric-mean hybrids (hours, after 1–2).
4. Cohort normalization (days).
5. Acoustic phoneme recognition → shared IPA aligner (the big one; weeks,
   but it's also the piece Proto-Lingua-Weaver can reuse directly).
