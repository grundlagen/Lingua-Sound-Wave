# Homophone Bench — EN↔FR Homophonic Translation Pipeline

Build text that, read aloud, **sounds like** English but **reads as** coherent
French (and vice-versa) — the Van Rooten *Mots d'Heures* effect — from data we
build and own, with no paid APIs.

The canonical pipeline has 9 stages, all offline and deterministic. Every stage
works and is measured. The recurring limiter is sentence-level fluency (the bigram
LM scores word-adjacency, not meaning).

## Quick Start

```bash
cd research/homophone-bench
pip install -r requirements.txt

# Build the dictionary (one-time, if not already built)
python build_dictionary.py

# Compose a dual-language line
python compose_lots.py
```

## Pipeline Stages

```
 0  MATCHER     combo scorer (ngram-dice + featural NW) — AUC 0.993
 1  DICTIONARY  build v5 lexicon (10,846+ entries) from Lexique + WikiPron
 2  FRAGMENTS   extract sub-word sound chunks (2,641 clean runs)
 3  MERGE       add ~944 generated fragment-chain matches
 4  GLUE        add function-word entries (the→de, and→end)
 5  FINALIZE    compute composition gates, gap ratios, B_safe/B_reservoir split
 6  MAPPING     build typed multi-graph (15,682 nodes, 19,121 edges, 5 types)
 7  RABBIT      semantic-component → homophonic-hop lattice
 8  COMPOSE     compose dual-language lines with QC gates
```

## Stage 0 — Matcher (the scoring foundation)

The homophone score `combo = 0.5·ngram_dice + 0.5·sharpened_featural_NW` (AUC 0.993
on 105 labeled EN↔FR pairs). Everything ranks on it. Implemented in `matcher.py`
with learned phoneme-substitution costs from `learn_costs.py`.

| file | role |
|---|---|
| `matcher.py` | the canonical matcher (combo, EQUIV floors, variants, gaps) — 68 files import it |
| `bench.py` | offline bake-off: 15 methods compared on 105-pair dataset |
| `dataset.py` | labeled EN↔FR benchmark pairs |
| `learn_costs.py` | learn phoneme-substitution costs from certified alignments |
| `lexicon_g2p.py` | dictionary-based G2P (Lexique 245k FR, WikiPron 65k EN) |
| `bigram_lm.py` | word-bigram LM (stupid backoff) for fluency scoring |

See `RESULTS.md` for full method comparison. `combo` wins at 0.993; audio methods
(mfcc-dtw 0.939) and allosaurus (0.713) score worse.

## Stage 1 — Dictionary (v5, 10,846+ entries)

Block+rank retrieval pipeline: phonemize lexicons → block with cheap Dice → rank
with full combo. From 561 EN × 3,950 FR: 27 S-tier (≥0.90) + 53 A-tier (0.78–0.90)
= 14% yield per 100 EN words. Only 3/80 are cognates — sound-alikes dominate.

| file | role |
|---|---|
| `build_dictionary.py` | scalable dictionary builder (block+rank) |
| `build_dictionary_full.py` | full-scale build (8,000 EN × 15,000 FR) |
| `dictionary-v5.json` | **canonical dictionary** — 10,846+ entries, 33-column schema |
| `dictionary-v5.tsv` | TSV export (11,789 rows including header) |

### v7 Upgrade Path

`build_v7_decode.py` adds 4,420 decoder-generated entries to v5, producing
`dictionary-v7-integrated.json` (16,208 entries). The v7 dict is canonical but
the v5 pipeline is the one wired end-to-end. To upgrade, point `compose_lots.py`
and `mapping_web.py` at `dictionary-v7-integrated.json`.

| file | role |
|---|---|
| `build_v7_decode.py` | v5 retrieval + single-phoneme decoder additions → 16,208 entries |
| `dictionary-v7-integrated.json` | v5 (11,788) ∪ 4,420 decoder additions |

## Stage 2 — Fragments

Extracts maximal cleanly-matched alignment runs (cost < 0.30, length ≥ 2) from
every usable dictionary entry. Builds a fragment index that lets the pipeline
handle words not in the dictionary by chaining sub-word sound chunks.

| file | role |
|---|---|
| `fragments.py` | extract fragments, chain-generate novel matches |
| `fragments.tsv` | 2,641 sub-word sound chunks (e.g. st→st, ɹi→ɹi) |

## Stage 3 — Merge Generative

Merges fragment-chain-generated novel matches into the dictionary, recomputes
independent combo scores, assigns tiers.

| file | role |
|---|---|
| `merge_generative.py` | merge ~944 validated generated matches into dictionary |
| `generative-matches.tsv` | novel matches found by chaining fragments |

## Stage 4 — Function Glue

Adds composition-only weak function-word entries with minimum 0.55 score.

| file | role |
|---|---|
| `function_glue.py` | add the→de, and→end glue entries |

## Stage 5 — Finalize

Derives junction fields (onset/coda, gap ratios, syllable deltas), splits B tier
into B_safe/B_reservoir, computes `usable_for_composition` gate. Builds the
composition index used by the composer.

| file | role |
|---|---|
| `finalize.py` | compute composition gates, write composition-index.json |
| `enrich.py` | enrich entries with NW alignment, pivot, syllable counts |

## Stage 6 — Mapping Web

Builds a typed multi-graph from dictionary entries and fragments. Five edge types
are never collapsed — walks can distinguish sound echoes from semantic echoes.

| file | role |
|---|---|
| `mapping_web.py` | build the typed multi-graph |
| `mapping-web.json` | **canonical graph** (15,682 nodes, 19,121 edges) |
| `mapping-walks.tsv` | permutation walks of sound→meaning→sound paths |

## Stage 7 — Round Rabbit

Collapses meaning edges into semantic components, then BFS over sound edges from
each component, attaching dictionary entries at every reachable node. Output is a
lattice for a poem generator, not a poem itself.

| file | role |
|---|---|
| `round_rabbit.py` | semantic-component → homophonic-hop lattice |
| `round-rabbit.json` | 42 rows from 6 components (seeded run) |

## Stage 8 — Compose

Multi-granularity composer: builds lots from fragments, whole entries, and
multiword entries. Composes EN→FR deterministically with QC gates (coverage ≥ 0.80,
rhythm budget, hiatus prevention). Produces dual-language lines.

| file | role |
|---|---|
| `compose_lots.py` | **the composer** — EN→FR dual lines with QC gates |
| `composition-lines.json` | composed output (2/4 test lines pass QC) |
| `recursive_poet.py` | recursive semantic-sound poem generator (multi-pass) |
| `soramimi.py` | sentence-level Van Rooten renderer (phoneme stream → FR words) |
| `phonetic_decoder.py` | beam decoder over Lexique trie (Knight–Graehl style) |
| `poetry_mode.py` | relaxed trie admitting 1-segment filler words |

## Multi-Agent Systems

### Three-Agent Loop (`three_agent_v2.py`)
Agent A (EN→FR carver) + Agent B (FR→EN hearer) + Agent C (deterministic judge
using combo scorer + semantic similarity). 4-iteration max, span-level repair
targeting. Proven architecture — Agent C's two-comparison fix is verified. Agent A
is currently lookup-only (6,143 pairs), limiting repair to known words.

### Qwen Fine-Tune Kit (`qwen-finetune/`)
Upgrades Agent A from lookup to generative. Full pipeline scripted: SFT data from
dictionary → repair-data synthesis → LoRA training → DPO optimization. Two
implementations: `QwenCarver` (GPU, transformer) and `CharLSTMCarver` (CPU,
char-level seq2seq). **Not yet run on GPU** — the scripts are complete and waiting
for execution.

### Self-Learn (`selflearn/`)
GPU-based self-improvement loop: sample → reward-score (combo × French validity) →
SFT on best. Qwen2.5-1.5B with LoRA. Includes eval harness, continual run supervisor,
and neural carver fallback (bi-GRU, CPU-only).

## Guardrails (deliberately NOT used)

- **Fuzzy / edit-distance string matching** (Levenshtein, difflib, rapidfuzz).
  Raw symbol-edit distance treats p↔b the same as p↔uː. Matching is the
  featural+n-gram `combo` with learned equivalence-floored costs.
- **Plotting libraries** (matplotlib, pyplot, seaborn). Findings are reported as
  TSV + plain-text tables so they stay diffable and terminal-reviewable.
- **epitran** for G2P. Pronunciations come from espeak-ng + CMUdict + Lexique
  — curated lexicons, not a rule-only transcriber.

## Key Data Artifacts

| file | what | size |
|---|---|---|
| `dictionary-v5.json` | canonical dictionary (v5 pipeline) | 708k lines |
| `dictionary-v7-integrated.json` | v5 + 4,420 decoder additions | 16,208 entries |
| `mapping-web.json` | typed multi-graph | 272k lines |
| `composition-index.json` | indexed entries by pivot/syllables/tier | 16KB |
| `fragments.tsv` | 2,641 sub-word sound chunks | — |
| `mapping-walks.tsv` | permutation walks | 501 lines |
| `round-rabbit.json` | semantic-component lattice | 2,231 lines |

## What Doesn't Exist Yet

These modules are described in architectural documents (`REPRESENTATION.md`) but
have not been written:

- **Phrase bank** — carving frequent EN phrases sound-first into bankable units
- **Bank composer** — chaining bank units (the current composer is `compose_lots.py`)
- **Corpus bank** — carving public-domain nursery-rhyme corpus
- **French anchor** — growing fr→en reverse coverage (currently 104 entries)
- **LLM fluency pass** — the README says this is the "one piece that wants an API key"

## What's in `methods/`

Experimental approaches that are not part of the canonical pipeline but represent
genuinely different algorithmic strategies. See `methods/README.md` for a catalog.

## Dictionaries Archive

Older dictionary versions (v2–v6) are in `dictionaries/archive/`. Only v5 and v7
are canonical. See `dictionaries/archive/README.md` for version history.
