# Speech / wav tests run (ffmpeg-free) + the Hickory diagnosis

## Why the frontier was blank on "Hickory dickory dock"

Not a bug in the idea — a fixed knob. The bake-off ran the frontier at a **fixed**
coverage-forcing scale (1.6) with a hard **coverage ≥ 0.70** filter. At scale 1.6
the best Hickory candidate reaches only **0.62 coverage**, so every candidate is
filtered out → empty result. The **learned per-line penalty** (`learned_coverage.py`)
picks scale **2.2** for that line and returns "incrédulité" (cov 0.77) — still a
single long word that the baseline's "critiques" (combo 0.58) beats, so **baseline
still wins Hickory**, but the frontier should show that ~0.04 result, not a blank.
Fix for the ensemble: drive the frontier with the learned penalty, not a fixed
scale. (Conclusion unchanged: ensemble, arbiter-picks-per-line.)

## The older speech/wav tests now run here

espeak writes WAV; Python's stdlib `wave` reads it; a NumPy linear resample to
16 kHz feeds the bench's MFCC. So `run_wav_tests.py` runs the acoustic methods
**without ffmpeg** (which is absent in this env). On the 105-pair benchmark:

| method | AUC | AUC_hard | pos | neg | separation |
|---|---|---|---|---|---|
| feat-dtw | 0.982 | 0.981 | 0.98 | 0.82 | +0.16 |
| mfcc-dtw | 0.939 | 0.935 | 0.98 | 0.77 | +0.21 |
| mfcc-dtw-xvoice | 0.932 | 0.925 | 0.94 | 0.70 | +0.23 |
| hybrid-geo | 0.953 | 0.952 | 0.96 | 0.74 | +0.22 |
| gate | 0.979 | 0.978 | 0.99 | 0.79 | +0.20 |
| **combo (symbolic)** | **0.993** | **0.992** | 0.84 | 0.34 | **+0.50** |

**Re-confirmed on the spot:** the acoustic (synthesized-speech MFCC-DTW) scorers
all trail symbolic `combo`, exactly as `RESULTS.md` reported. The tell is
**separation**: the audio methods rate *everything* high (neg means 0.70–0.82),
so they discriminate weakly (+0.16–0.23) where combo separates cleanly (+0.50).
Acoustic similarity of synthesized speech is dominated by coarse prosody — the
wrong signal for homophone discrimination.

## On the Whisper/wav2vec already in Lingua

The production app does carry the neural acoustic path —
`artifacts/api-server/src/lib/whisper-phonetic-cluster.ts` and `wav2vec.ts`
(plus the OpenAI audio integration). That is the **neural** version of the same
acoustic idea these MFCC-DTW tests embody, and `RESULTS.md` already measured its
family: feeding *synthesized* speech to a universal recognizer (allosaurus)
collapsed to 0.713 — ASR's internal LM hallucinates toward plausible text rather
than reading the acoustics. The honest standing guidance (`REPRESENTATION.md` §2):
keep Whisper/wav2vec as a **validator / re-ranker on finished output with real or
multi-voice audio**, never as the load-bearing scorer — the symbolic route wins
discrimination decisively. Running the TS Whisper path needs the app + models, so
it's outside this pure-Python env; the deterministic MFCC-DTW family above is the
runnable stand-in and it reproduces the verdict.

`run_wav_tests.py` is the artifact; re-run anytime to reconfirm.
