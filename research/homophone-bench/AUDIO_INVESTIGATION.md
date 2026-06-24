# Is the audio path's "high AUC" real? Investigated — no.

The audio methods score AUC 0.93–0.98 on the 105-pair benchmark, close to symbolic
combo's 0.993. That looked promising. It does not survive investigation.

## The tell in the benchmark: separation, not AUC

| method | AUC | separation (pos−neg) | neg mean |
|---|---|---|---|
| combo (symbolic) | 0.993 | **+0.50** | 0.34 |
| feat-dtw (audio) | 0.982 | +0.16 | 0.82 |
| gate (audio) | 0.979 | +0.20 | 0.79 |

Audio's AUC is high but its **separation is tiny** and its **negative mean is
0.65–0.82** — it rates *everything* high. High AUC here just means it ranks the
trivially-different translation negatives slightly below positives on an easy set.

## The real test: retrieval (rank the lexicon, find the true homophone)

Ranking 150 French words for each English query, where does the documented
homophone land?

| EN | true FR | **audio rank** | audio noise floor | combo rank | combo noise |
|---|---|---:|---:|---:|---:|
| shoe | chou | **66** | 91 | 1 | 0 |
| key | qui | 4 | 28 | 1 | 0 |
| set | cette | 3 | 11 | 1 | 0 |
| two | tout | 9 | 28 | 1 | 0 |

Audio buries the true homophone (rank 3–66) under a **huge noise floor** — for
"shoe", 91 of 150 French words score as high as the real match "chou". Symbolic
combo ranks the true homophone **#1 every time, noise floor 0**.

## Verdict

The audio "high results" are an **easy-benchmark artifact**. On the task that
matters — retrieval / dictionary-building / ranking carves — audio is a weak
discriminator: it can neither pick the right word nor set a usable threshold.
Confirmed mechanism behind `RESULTS.md`'s warning, and consistent with
`whisper_train.py` (the trained ensemble plateaus ~0.92–0.95 and overfits) and
`WHISPER_IMPROVE.md` (the production encoder is a random stub).

**So, to the question "incorporate and improve audio as a new high-result method?"
— no.** It does not need more investment as a primary or additive signal:
- it does **not** retrieve (rank 66 for shoe~chou),
- it does **not** add to combo (trained ensemble correlates +0.795 but doesn't beat it),
- it **plateaus** ~0.95 on synthesized-speech prosody.

Its one legitimate role (unchanged): a **soft re-ranker / round-trip validator on
a shortlist the symbolic stage already produced**, ideally on real or multi-voice
audio, never the retriever or the judge. Everything upstream — v5/v6/v7, the carve
engine, Round Rabbit — correctly ranks on the symbolic combo.
