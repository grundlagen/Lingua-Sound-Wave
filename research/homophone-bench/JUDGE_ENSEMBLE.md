# Why "0.993 on everything" was misleading — and how to judge honestly

You were right to distrust the number. A single matcher reporting AUC ~0.993 on
the 105-pair benchmark is **grading itself by its own methodology against easy
negatives**. Two changes fix it: an **ensemble** of orthogonal methods, and a
**strict** evaluation that removes every easy win. The judge of record stays
`combo` (unchanged) — these are diagnostics that tell us how much to trust it.

## 1. The easy benchmark inflates every method (`hard_judge.py`)

On the original pairs, negatives sound nothing like the target, so *every*
phoneme method clears ~0.99. Rebuilt as a gold-anchored set with decoy French
words (length-matched, real homophones-of-something-else):

| method            | AUC (easy decoys) |
|-------------------|------------------:|
| ngram_dice        | 0.989 |
| feat_nw_sharp     | 0.998 |
| prosody           | 0.972 |
| drive_equiv       | 0.965 |
| combo             | 0.997 |
| **ENSEMBLE (z-avg)** | **0.999** |

Still high — but the methods **genuinely disagree** on individual hard cases
(e.g. `dicky~tiquer`: ngram 0.00 but feat 0.91; `baud~bote`: ngram 0.50 but
prosody 0.98). A single metric would mislead on exactly those. The ensemble
z-normalises each orthogonal method and averages, so no one methodology
dominates; DeepSeek arbitrates the highest-variance (most-disagreed) cases.

## 2. STRICT mode — the honest, lower number (`strict_judge.py`)

The ensemble still flatters because the decoys are random. Strict mode tightens
three screws:

- **Adversarial negatives**: each English word's negative is the French word in
  the pool that *sounds most like it but is the wrong meaning* (argmax `combo`).
  Every positive competes against its single most-confusable rival.
- **AND-logic ensemble**: geometric mean instead of average — **all** methods
  must agree; one sceptic drags the score down.
- **Gold-rate, not just AUC**: fraction of positives that clear a high absolute
  bar (geo ≥ 0.60) **and** beat their nearest decoy by a margin (≥ 0.10).

Result (140 gold positives vs 140 nearest-confusable negatives):

| method            | AUC (strict) | vs easy |
|-------------------|-------------:|--------:|
| ngram_dice        | 0.732 | −0.257 |
| feat_nw_sharp     | 0.873 | −0.125 |
| prosody           | 0.697 | −0.275 |
| drive_equiv       | 0.694 | −0.271 |
| **GEO-ENSEMBLE**  | **0.760** | −0.239 |

- **Strict gold-rate: 56.4%** — a "GOLD" pair clearly beats its closest rival
  only about half the time.
- **DeepSeek harsh rubric**: true homophones mean **24/100**, nearest decoys
  **0/100** — separated, but compressed low, because many short single-word GOLD
  pairs (`thread~raide`, `on~âne`) are *loose* homophones, not near-perfect.

### What this tells us
1. The real discriminative power lives in **`feat_nw_sharp`** (0.873 strict);
   `ngram_dice` exact-match collapses (0.732) when rivals share bigrams.
2. The **GOLD tier of the v7 dict is softer than its name** — it was defined by
   `prosody ≥ 0.70 ∧ meaning ≥ 0.45`, which admits loose phonetic matches. For
   self-learning, treat strict gold-rate (not AUC) as the metric to move, and
   consider a `STRICT-GOLD` sub-tier (geo-ensemble ≥ 0.60 ∧ beats-nearest-rival)
   as the frozen eval set.
3. **Corpus of creation need not be word-to-word.** The winning signals score
   *phoneme streams*, not lemma equality — so whole-line carvings that no single
   French word matches are still admissible, judged by the geo-ensemble.

## How to use it
- `python hard_judge.py` — ensemble + LLM-on-disagreement (calibration).
- `python strict_judge.py` — adversarial + AND-logic + strict gold-rate + harsh
  LLM (the number to optimise against).
- The production judge remains `combo`; promote a change only by a deliberate
  edit once strict gold-rate improves.

> API keys (DeepSeek/OpenRouter) live in gitignored `.env.local`; rotate them.
