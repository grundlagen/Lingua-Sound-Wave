# Improving the Whisper path

## The finding: the production Whisper encoder is a stub

`artifacts/api-server/src/lib/whisper-phonetic-cluster.ts` advertises
"sub-phonemic cluster matching using Whisper encoder, ≥0.95 cosine = Perfect
Match." But `extractWhisperEncoderFeatures()` is a **stub**: it returns
`Math.random()` 768-dim vectors (with a `TODO: Replace with actual Whisper encoder
call`). So the Perfect-Matching Reservoir compares **random vectors** — the 0.95
threshold fires only by chance, and any "perfect match" stored is noise.

Measured (`whisper_improve.py`): the random stub's cosine between any two words
centres on **0** (−0.07…+0.02) — no signal at all.

## The fix (deterministic, runnable now)

Replace the random features with **real MFCC encoder features** (mean+std pooled
over frames, 78-dim, L2-normed) — the same acoustic signal the bench's MFCC-DTW
uses, reduced to a cosine-comparable vector. It is a drop-in for the stub and it
carries real acoustic information:

| pair | real-feature cos | stub cos |
|---|---|---|
| shoe~chou (homophone) | 0.995 | 0.015 |
| key~qui (homophone) | 0.999 | 0.012 |
| two~tout (homophone) | 0.994 | −0.068 |

The real features are meaningful where the stub is noise. **Honest caveat:**
pooled-vector cosine alone barely *separates* homophones from non-pairs
(+0.004) — all short words pool to similar vectors. Real discrimination needs
**frame-level DTW** (what `mfcc-dtw` already does) or a **true Whisper encoder**,
not pooled cosine. So the cluster-matching design should use DTW over the frame
sequence, not a single pooled vector.

## Recommendation (priority order)

1. **Delete the random stub** — it is actively harmful (records noise as "perfect
   matches"). At minimum swap in the real MFCC features.
2. **Score with frame-level DTW**, not pooled cosine — the pooled vector loses the
   temporal detail homophony lives in.
3. **Wire the real Whisper encoder** when the model is available — strictly better
   features than MFCC, but the same two rules apply.
4. **Keep it as a re-ranker / round-trip validator on finished output** with real
   or multi-voice audio (`REPRESENTATION.md` §2), never the load-bearing scorer —
   `RESULTS.md` showed neural recognizers on *synthesized* speech collapse (0.713).

`whisper_improve.py` is the runnable demonstration of (1)–(2).
