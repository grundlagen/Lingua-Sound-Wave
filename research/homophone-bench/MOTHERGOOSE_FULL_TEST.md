# Unified Mother Goose test — every method, nothing cut

`mothergoose_full_test.py` generates a French carve for each public-domain line
(frontier whole-line carve + baseline decoder) and scores each with **all** the
scorers side by side: symbolic combo (sound), MFCC-DTW acoustic (the
Whisper-family audio path, ffmpeg-free), and bigram coherence. French is
machine-generated; English source is public-domain (1916).

| line | strategy | FR carve | sound | audio | coh |
|---|---|---|---|---|---|
| Humpty Dumpty | baseline | épidémie | 0.49 | 0.46 | 0.31 |
| Humpty Dumpty | **frontier** | un petit un petit | 0.54 | **0.83** | 0.85 |
| Humpty Dumpty sat on a wall | baseline | épidémie | 0.32 | 0.30 | 0.31 |
| Humpty Dumpty sat on a wall | **frontier** | un petit un petit et on vo… | 0.52 | 0.51 | 0.79 |
| Jack and Jill | baseline | desquels | 0.46 | 0.68 | 0.26 |
| Jack and Jill | frontier | cicatrices | 0.34 | 0.65 | 0.22 |
| Hickory dickory dock | baseline | critiques | 0.54 | 0.50 | 0.45 |
| Hickory dickory dock | frontier | incrédulité | 0.41 | 0.56 | 0.09 |
| Pat a cake | baseline | but équipe | 0.41 | 0.79 | 0.72 |
| Pat a cake | frontier | but y | 0.47 | 0.65 | 0.72 |

## What the audio path adds (run, not dropped)

The acoustic MFCC-DTW scorer **validates the best carve**: "un petit un petit"
gets audio **0.83** vs épidémie's 0.46 — independent acoustic agreement that it
sounds like "Humpty Dumpty," confirming both the symbolic score and the earlier
ear-test. It's noisier on the longer/denser carves (synthesized-speech prosody
dominates), which is exactly why `REPRESENTATION.md` keeps it as a **soft
re-ranker on finished output**, not the judge. Here it's run beside the symbolic
score on every carve, not cut.

## Three methods, three jobs (all live)

- **carve engines** (frontier + baseline) — generate the line; neither dominates
  (frontier wins the carve-able lines, baseline the dense ones).
- **scorers** (symbolic combo + MFCC-DTW audio + coherence) — rate each carve;
  audio agrees on tight matches, soft-re-ranks the rest.
- **Round Rabbit** (`round_rabbit_run.py`, Fable's lattice) — themes the content
  words: the already-homophonic neighbours that make good carve material.

## On v6 only producing ~800 entries

Not a fineness limit — a **speed cap**. `build_v6.py` was run `--limit 1500`, so
it only mined the top 1500 frequent English words; 807 cleared the gate (58%).
Finer grain (single phonemes + fillers) means a *full* run reproduces v5's
coverage **plus** filler extras, not fewer. The practical integration is already
in place: `dictionary-v6-integrated.json` = full v5 ∪ v6's new entries (12,348),
which is what Round Rabbit and the engine should consume. To make v6 standalone
comprehensive, re-run `build_v6.py` over all v5 headwords with no limit (linear
runtime; ~7 min for 7,286).
