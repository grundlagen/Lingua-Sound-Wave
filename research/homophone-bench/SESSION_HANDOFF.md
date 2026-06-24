# Session handoff — homophonic generation engine (save point)

Branch: `claude/phrase-weave-multiword`. All work below is committed + pushed.
Read this first to resume. Public-domain sources only; copyrighted renderings
(van Rooten etc.) were never ingested — they are targets to match.

## The objective (corrected, locked)

ONE phoneme stream that reads as **sensical text in both languages at once**.
Per item: `sound(L1)≈sound(L2)` AND L1 coherent AND L2 coherent. **No
meaning-equivalence term** — the art (van Rooten Mother Goose, Fraser's
"frame-evasion") deliberately does not preserve meaning. Theme-matching was
explicitly retracted. See `OBJECTIVE_AND_GOLD.md`, `CENTRAL_PROBLEM.md`.

## The pipeline now assembled (all reuse June 11–12 primitives)

`espeak G2P → continuous phoneme stream → resegmentation decoder (free word
boundaries, phonetic_decoder.py v4) → v6 poetry trie (1-seg fillers) →
coverage-forced beam (learned per-line penalty) → coverage-aware MATCHER ranking
(combo, AUC 0.993) → bigram coherence`.

It GENERATES whole-line carves that re-cut English boundaries into French ones:
- "Humpty Dumpty" → "un petit un petit" (cov 0.92, coh 0.85) — engine's own output
- "Humpty Dumpty sat on a wall" → "un petit un petit et on vol" (cov 0.90, 4 fillers)

## Key findings (each tested, each committed)

1. **v5 is near-ceiling word-for-word** (86% coverage of documented homophones;
   v5-FR ≥ documented in 81%). `PAPERS_AND_V5_TEST.md`.
2. **It can be topped only by re-ranking**: v5 selected by saturated build-score;
   re-rank its own candidates by matcher+frequency → 18% improve (wee→oui not
   ouïe). `dictionary-v5-reranked.tsv`, `RERANK_V5_RESULTS.md`.
3. **Generation ≠ dictionary**: Mother Goose needs RESEGMENTATION (re-cut the
   stream), which a word-keyed table structurally cannot do.
   `MOTHERGOOSE_GEN_PROBLEM.md`.
4. **"Extra words" = 1-segment French fillers** (un/et/on/a/aux/y/eau) that
   `MIN_WORD_SEGS=2` excluded. Admitting them (poetry mode) enables the carve.
   `JUNCTURE_AND_FILLERS.md`, `poetry_mode.py`.
5. **Decoder ranking bug**: its similarity is coverage-blind (rated 58%-covering
   "épidémie" above the fuller "un petit"). Fix: rank by the matcher combo.
   CONFIRMED un petit d'un petit 0.627 > épidémie 0.555 by sound.
   `FILLER_CARVE_FINDING.md`.
6. **Coverage-forcing**: the beam deleted half the stream; penalising deletion
   (keep /h/ cheap) → coverage 0.62→0.95, surfaced the full carve.
   `WHOLE_LINE_CARVE.md`.
7. **Learned coverage penalty**: per-line autotune of the deletion scale
   maximising dual. `learned_coverage.py`. Clean lines lock to 1.6.
8. **Single-phoneme composition beats 2-phoneme chunks for GENERATION**: 50% vs
   16% dual-decodable yield (2-phoneme chunks are cluster-biased, French avoids
   clusters). Matching already runs at single-phoneme grain.
   `SINGLE_PHONEME_COMPOSITION.md`.
9. **v6** = v5 ethos + 1-seg fillers + arbiter ranking + coherence field +
   re-mine. `build_v6.py`, `dictionary-v6.tsv`, `v6-fillers.tsv`, `V6.md`.
10. **Cycle-consistency** is a valid label-free signal (AUC 0.872 vs trusted).
    `cycle_consistency.py`, `EXPERIMENTS_N1_N2_RABBIT.md`.

## Do we need to re-mine? No (for generation)

The engine decodes LIVE off the trie + matcher + fillers = re-mine-per-query.
The static dict is a cache + fragment source + filler primitives, not the
generation substrate. `LEARNING_REMINE_GRANULARITY.md`.

## Open next steps (priority order)

1. **The L2-coherence model** — THE gating component, everywhere. Bigram is too
   weak to pick the beautiful carve among coverage-correct ones. Need KenLM
   5-gram (Leipzig/OpenSubtitles) min, an LLM ideally. It must rank a
   looser-sounding COHERENT phrase above a tighter-sounding noun.
2. **Fine/mixed inventory** (in progress): single phonemes + all 2–4 chunks +
   the equivalence rules (odd stops etc.) in the generation pool, arbiter-ranked.
3. **Synonym-swap layer** — for lines that don't carve (Jack and Jill, Hickory
   dickory dock fail at any grain/penalty). Preserve rhyme meaning, substitute a
   near-synonym whose phoneme string carves. Hook = the GAP verdict.
4. **Learn the gap/sub COSTS** — extend learn_costs.py with the cycle-consistency
   objective (label-free).
5. **Passage-level held-out eval** + grow the fair-use gold (Mother Goose now,
   Catullus/Arabic later). `dual_reading_eval.py`, `gold-dual-readings.tsv`.

## Run commands

```
python whole_line_carve.py        # whole-line generation
python learned_coverage.py        # per-line learned deletion penalty
python poetry_mode.py             # filler-word carving
python build_v6.py --limit N      # mine v6
python dual_reading_eval.py       # score gold homophonic verse
python rerank_v5.py               # the improved word-for-word table
```

Env: panphon + numpy + wordfreq + espeak-ng (132 voices). Heavy libs
(kenlm/pynini/torch/faiss) pip-only, not installed; pure-Python core runs
anywhere. bigram-lm-{en,fr}.pkl present.
