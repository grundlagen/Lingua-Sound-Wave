# Homophonic Engine — dual-channel translation (sound *and* sense)

A translation engine for the form Van Rooten and Luis van Rooten played by
hand: render a text in another language so it **sounds like the original**
while **carrying its meaning**. Drop in a paragraph, get the dual rendering.
Built to be tinkered with — for poetry, lyrics, puns, or translation-prize
experiments.

```bash
python engine.py --text "the pale moon lights the quiet sea"
echo "your poem here" | python engine.py --drift 0.7 --show-work
python engine.py --pair en-es --text "the sea is cold"
```

```
ORIGINAL    the pale moon lights the quiet sea
HOMOPHONIC  Pâli des riment light soft si      [fr, drift=0.55]
```

## One run, ordinary resources — and that is the finding

You do **not** need a parallel corpus, a trained model, or paid APIs. The
whole engine stands on four free, offline-cacheable resources:

| resource | role | free? |
|---|---|---|
| **espeak-ng** | grapheme→IPA for ~100 languages | yes, apt |
| **wordfreq** | frequency word lists per language | yes, pip |
| **panphon** | articulatory features (language-agnostic phoneme distance) | yes, pip |
| **MUSE** bilingual dicts | meaning edges (en-fr, en-es, en-he, …) | yes, download |
| Datamuse (optional) | synonyms + glosses for periphrasis | yes, no key, cached |

**Is the seed enough to learn pronunciation rules?** Yes — proven. panphon
gives a working phoneme-distance baseline for *any* pair with no training.
Running `seed_learn.py` on a freshly mined pair then refines it: from the
en-es seed alone it recovered textbook English→Spanish rules — lax vowels
/ɪ ɛ/ → /e i/ (Spanish has none), reduced /ʌ ə æ/ → /a/, diphthongs →
monophthongs — with **no parallel corpus and no hand-coding**. The seed *is*
the training set. Bigger curated lexicons help recall at the margins, but
nothing is *required* beyond the four resources above.

## The knobs (it's a toy as much as a tool)

- `--drift` 0→1: pure sound (surreal soramimi) ↔ hold meaning hard. The dial
  the whole project turns on.
- `--pair en-fr | en-es | …`: any pair in `multilang.PAIRS` with a built
  target lexicon.
- `--show-work`: per-word sound/sense scores and the chain/periphrasis used.

## How a word transfers (the pipeline in one breath)

source word → {itself, synonyms, glossed descriptions, last-resort poetic
periphrases} → each candidate decoded homophonically into the target
language (phoneme beam search over the target pronunciation trie, with
learned + featural costs) → keep the rendering with the best
`sound · (sound↔sense by --drift)` blend, anchored to the source meaning by
multilingual embeddings so it can't drift into nonsense.

## The 11–12 June schema (authoritative pipeline)

This is the canonical methodology; everything else in the repo (the
TypeScript `artifacts/` explorer, the May handover notes) predates it. The
pipeline is fully offline and deterministic:

1. **G2P** — espeak-ng + CMUdict (ARPAbet→IPA via `lexicon_g2p.py`) for
   English, Lexique for French. Curated lexicons, broad canonical IPA.
2. **Matcher** — `combo` in `matcher.py`: sharpened panphon featural
   Needleman–Wunsch + phoneme n-gram Dice, with substitution costs floored
   by **learned** phoneme equivalences (`learn_costs.py` → `learned-costs.json`,
   Ristad–Yianilos counts mined from certified alignments).
3. **Decoder** — `phonetic_decoder.py`: a Lexique pronunciation **trie + beam
   search** that generalizes 2-chunk concatenation to true multi-to-multi
   rendering (`--augment`/`--reverse` grow the dictionary).
4. **Graph + weave** — `chain_game.py`/`weave.py` build the 3-layer graph
   (sound ≈, translation =, semantic ~) and beam-search chains under strict
   alternation; `explode_web.py` exposes every sub-path and loop-certifies
   the meaning-preserving sound pairs.
5. **Growth = re-mining.** Promoting/relabelling certified pairs cannot
   compound — loops are made of edges that were already usable. New edges
   come only from re-running the decoder/augment passes with the learned
   costs + the bigger CMUdict trie, then re-weaving (the documented bootstrap
   cycle).

### Deliberately NOT used (guardrails)

These approaches are explicitly out of the schema and must not be
reintroduced:

- **Fuzzy / edit-distance string matching** (Levenshtein, difflib
  `SequenceMatcher`, `rapidfuzz`/`fuzzywuzzy`). Raw symbol-edit distance
  treats `p`↔`b` the same as `p`↔`uː` (see `RESULTS.md`) and inflates
  negatives. Matching is the featural+n-gram `combo` with learned
  equivalence-floored costs; multi-to-multi is the decoder trie. Coverage
  grows by **re-mining**, never by relaxing the match.
- **Plotting libraries** (matplotlib, pyplot, seaborn). Findings are
  reported as TSV + plain-text tables (`bench.py`, `chain_methods.py`, the
  `*-report.txt` files) so they stay diffable and terminal-reviewable.
- **epitran** for G2P. Pronunciations come from espeak-ng + CMUdict +
  Lexique — curated lexicons, not a rule-only transcriber.

## Add a language in one line

`multilang.PAIRS` holds every pair-specific fact (espeak voices, wordfreq
codes, MUSE url, RTL flag). Then:

```bash
python multilang.py en it          # build lexicons + mine the dictionary
python seed_learn.py en-it         # learn it->en pronunciation rules from the seed
python engine.py --pair en-it --text "..."
```

Status today: **en-fr** fully built (10.6k-pair dictionary, woven chain-web,
loop-certified gold layer); **en-es** proven (783 pairs mined, Spanish rules
learned from seed); **en-ga / en-he / ar-he** registered with honest
per-pair notes in `MULTILANG.md`.

## How close is it? (measured, honest)

On an 11-content-word paragraph: **homophonic rendering 100%** (every word
gets a high-sound target rendering — a paragraph that *sounds* like the
source is producible now), **dual sound+sense ≈63% per word**. The gap to
full literary dual translation is **sentence-level fluency** — the target
reads word-salad because word order follows the source sound stream
(inherent to homophony). Closing it is a *generation* problem (arrange the
rendered material into target-grammatical lines), best done by an LLM
sitting on top of this deterministic engine as proposer/judge — the engine
owns the phonetics and never hallucinates them. That's the one piece that
wants an API key; everything else runs offline.

## File map (engine)

- `engine.py` — the button (drop text → dual rendering)
- `multilang.py` — language-pair registry + lexicon/dictionary builder
- `seed_learn.py` — learn a pair's pronunciation rules from its seed
- `matcher.py` — language-agnostic phoneme matcher (panphon + n-gram + costs)
- `phonetic_decoder.py` — homophonic beam decoder (Knight–Graehl style)
- `chain_game.py` / `weave.py` — the sound+meaning graph and chain-web
- `RESULTS.md`, `MULTILANG.md`, `PARAGRAPH_STATUS.md`, `LLM_RECIPE.md` — the
  measured findings and design notes

---

# Homophone bench — the EN↔FR homophonic-translation pipeline

Build text that, read aloud, **sounds like** English but **reads as** coherent
French (and vice-versa) — the Van Rooten *Mots d'Heures* effect — generatively,
from data we build and own.

The pipeline has six stages. Each stage's files are listed; the **spine** (the
files you actually run end-to-end) is marked ▶, the rest are research/diagnostics.

```
0 MATCHER     score whether two words sound alike  (the AUC-0.993 combo)
1 DICTIONARY  build the EN↔FR homophone lexicon     (v2 → v7)
2 WEB/WEAVE   wire sound+meaning into one graph, harvest loop-certified atoms
3 ROUTING     reach/route/measure the graph; proofs and webbing density
4 CARVE       cut an English phoneme stream into French words (the real homophone)
5 ENGINES     translation + generative poets built on the carve / web
6 BANK        a phrase bank of verified homophones → compose Van Rooten lines
```

Copyright note: we train/compose only on **public-domain** English (traditional
nursery rhymes, Gutenberg-derived frequency models) and **generate** the French.
The homophonic-translation canon (Van Rooten 1967, de Kay 1980, Hulme 1981) is in
copyright and is **not** used or reproduced. See `CORPUS_AND_COPYRIGHT.md`.

---

## Stage 0 — Matcher (the scoring foundation)

The homophone score `combo = 0.5·ngram_dice + 0.5·sharpened_featural_NW`
(AUC 0.993). Everything ranks on it.

| file | role |
|---|---|
| ▶ `matcher.py` | the benchmark-winning matcher (combo, EQUIV floors, variants, gaps) |
| ▶ `bench.py` | offline bake-off: which matching method discriminates (symbolic + audio) |
| `dataset.py` | labeled EN↔FR benchmark pairs |
| ▶ `bigram_lm.py` | word-bigram LM (stupid-backoff) for French/English fluency |
| ▶ `lexicon_g2p.py` | dictionary-based grapheme→phoneme |
| `learn_costs.py` | learn phoneme-substitution costs from our own certified data |

## Stage 1 — Dictionary (v2 → v7)

Retrieval-built EN→FR homophone pairs, refined over versions; **v7-integrated**
(16,208 entries) is the one in use.

| file | role |
|---|---|
| `retrieval.py` | can we BUILD a dictionary (not just classify)? the retrieval test |
| `build_dictionary.py`, `build_dictionary_full.py` | scale the dictionary build |
| `refine_dictionary.py`, `enrich.py`, `finalize.py` | v3 refine → composition-ready v5 |
| `rerank_v5.py`, `retrieve_vs_decode.py` | improve/contrast word-for-word selection |
| `build_v6.py` | v6: v5 ethos + 1-segment fillers + arbiter rank |
| ▶ `build_v7.py`, `build_v7_decode.py` | v7 = best of v5 (retrieval) + v6 (single-phoneme decoder) |
| `allophone_layer.py`, `fragments.py`, `fragment_weave.py` | allophone/fragment enrichment |
| `function_words.py`, `fine_inventory.py`, `learned_coverage.py` | rescue/inventory/coverage knobs |

## Stage 2 — Web & weave (the graph)

Wire three typed edges — homophone `≈`, translation `=`, synonym `~` (real
multilingual embeddings) — into one graph, then harvest **loop-certified pairs**:
sound edges sitting inside a strictly-alternating cycle that returns home, so
meaning round-trips (the dual atoms).

| file | role |
|---|---|
| `web.py`, `mapping_web.py` | one graph over EN/FR words, two edge types |
| ▶ `chain_game.py` | alternation chains (sound/meaning telephone) — builds the graph |
| ▶ `weave.py` | loop all chains into one web; emit chain-web + loops |
| ▶ `explode_web.py` | all-step explosion + **loop-certified-pairs** (the atoms) |
| `round_rabbit.py`, `round_rabbit_run.py`, `rabbit_walk.py` | semantic→homophonic lattice (Fable's engine) |
| `sound_meaning.py` | grade each sound-pair by cross-lingual semantic similarity |
| ▶ `cache_graph.py` | pickle the unedited graph → `graph-v7u.pkl` (instant reuse) |
| ▶ `cache_vecs.py` | encode all 195k node embeddings → `node-vecs.npy` |

## Stage 3 — Routing, proofs & webbing

| file | role |
|---|---|
| `hoproute.py` | unified shortest-hop routing + ideal-hop-distance (inflation) |
| `routes.py` | node profiles + any-word routing + interchange hubs |
| `reach.py` | **Proof 1**: any word reaches any word (99.6% giant component) |
| `dict_coverage.py`, `web_coverage.py` | **Proof 2**: dictionary coverage of the web |
| `transfer_distance.py` | rank transfers by sound × semantic-distance (far-meaning bridges) |
| ▶ `webbing.py` | tile-graph density: dual-atoms don't co-chain (both-rail = 0) |
| `chain_hop.py`, `chain_paragraph_hop.py`, `chain_translate.py`, `set_match.py` | chain-hop transfer translators |
| `cycle_consistency.py`, `rhythm_channel.py` | label-free quality / prosody experiments |

## Stage 4 — Carve (the actual homophone)

A graph walk only *stutters* on a sound; real homophony = **carving** an English
phoneme stream into French words so the French spoken rebuilds the English.

| file | role |
|---|---|
| ▶ `phonetic_decoder.py` | beam decoder: segment an EN phoneme stream into FR words |
| ▶ `poetry_mode.py` | the French unit trie + filler whitelist (the carve pool) |
| ▶ `whole_line_carve.py` | one coverage-forced carve across a whole line (`carve_line`) |
| ▶ `carve_quality.py` | **sound-first decoding** lever (combo 0.30→0.54) |
| `generation_engine.py`, `generate.py`, `generate_pipeline.py` | the 1+2+3 generation engine |
| `soramimi.py`, `demo_homophonic_writing.py` | sentence renderers / demos |

## Stage 5 — Engines (translation & generative poets)

| file | role |
|---|---|
| `translate_engine.py` | tiered EN→FR homophonic+semantic translator (S/A/P/literal) |
| `vanrooten.py` | joint drift+carve: one French line, source allowed to drift |
| `homophonic_poet.py` | generative + **verified** homophone (themed EN → carve → combo proof) |
| `web_poet.py` | themed web-walk poet (exploration; NOT a true spoken homophone) |
| `ladder.py` | dual-reading ladder: typed slots + loop-tiles + sense-split |
| `dual_translate.py`, `paraphrase_translate.py`, `phrase_weave.py` | dual/paraphrase/phrase translators |

## Stage 6 — Phrase bank & composer (current frontier)

The escape from the both-rail wall: bake the homophone **inside each unit**, then
only the seams need fluency.

| file | role |
|---|---|
| ▶ `phrase_bank.py` | carve frequent EN phrases sound-first → `phrase-bank(-balanced).tsv` |
| ▶ `bank_composer.py` | chain bank units into dual-reading lines (`--theme`, `-n`, seeds) |
| ▶ `corpus_bank.py` | carve the PD nursery-rhyme corpus → `corpus-carves.tsv` gallery |
| ▶ `french_anchor.py` | French-anchor the FR rail (grow loop-tiles 813→896) |

## Tests / diagnostics

`mothergoose_full_test.py`, `test_against_v5.py`, `test_mothergoose_gen.py`,
`compare_strategies.py`, `dual_reading_eval.py`, `compose_smoke.py`,
`lm_compare.py`, `content_neighbours.py`, `corpus_phrases.py`,
`build_poem_dataset.py` — corpus tests, ablations, and the dual-reading evaluator.

## Audio path (investigated, parked)

`audio_investigate.py`, `whisper_improve.py`, `whisper_train.py`,
`run_wav_tests.py`, `run_all_tests.py` — MFCC/Whisper acoustic re-ranking.
Verdict (`AUDIO_INVESTIGATION.md`): high AUC is an easy-benchmark artifact; audio
does not retrieve. Keep symbolic combo as the judge; audio only as a soft
re-ranker on a shortlist.

---

## Data artifacts (key)

| file | what |
|---|---|
| `dictionary-v7-integrated.json` | the 16,208-entry homophone lexicon (in use) |
| `graph-v7u.pkl` | the unedited weave graph (sound+trans+sem edges) — built by `cache_graph.py` |
| `node-vecs.npy` / `node-ids.json` | all 195k node embeddings — built by `cache_vecs.py` |
| `loop-certified-pairs-v7u(-aug).tsv` | the dual-atom alphabet (813 / 896) |
| `phrase-bank.tsv` / `phrase-bank-balanced.tsv` | 1,034 homophone phrase units (best-combo / fluent) |
| `corpus-carves.tsv` | PD nursery-rhyme homophonic gallery |
| `fr-anchored-pairs.tsv` | French-anchored sound pairs (FR rail) |

Big derived artifacts (`graph-v7u.pkl` 41 MB, `node-vecs.npy` 300 MB) are not
committed; rebuild with `cache_graph.py` / `cache_vecs.py`.

## End-to-end (the spine)

```bash
# 0–1. dictionary already built: dictionary-v7-integrated.json
# 2. graph + embeddings (one-time, ~10 min each)
python cache_graph.py            # -> graph-v7u.pkl
python cache_vecs.py             # -> node-vecs.npy, node-ids.json
# (loop atoms: weave.py --full ; explode_web.py  -> loop-certified-pairs-*.tsv)

# 4. find the carve quality lever
python carve_quality.py          # sound-first decoding wins

# 6. build the bank and compose
python phrase_bank.py 1100 balanced     # -> phrase-bank-balanced.tsv
python bank_composer.py                 # dual-reading Van Rooten lines
python bank_composer.py --theme sea -n 5
python corpus_bank.py                   # PD rhyme gallery

# verified single-line generation
python homophonic_poet.py love night
```

## The standing ceiling

Every stage works and is measured; the recurring limiter is the **fluency / L2
model**. The bigram LM scores word-adjacency, not sense — so lines are
sound-true and French-shaped but not yet *meaningful* verse. Swapping in a real
French language model (the only missing dependency) turns the same pipeline's
output from sound-true strings into Van Rooten-class verse. Stage-specific notes
live in the per-topic `*.md` files (e.g. `CARVE`, `WEBBING`, `PHRASE_BANK`,
`BANK_COMPOSER`, `GENERATIVE_POET`, `PROOFS_REACH_AND_COVERAGE`).
