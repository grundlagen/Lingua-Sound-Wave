# Lingua Weaver idea trawl

_90 modules in tree, 22 more on other branches. Read-only scan; originals untouched._


## Modules in the current tree

- `_load_env.py` — Load .env.local (gitignored) for local work; Claude Code doesn't commit it.
- `allophone_layer.py` — Allophone layer -- a SEPARATE enrichment that borrows the existing match-time machinery (matcher._variants + EQUIV) and adds the English allophonic processes th
- `audio_investigate.py` — Investigate the audio path's "high results": is the high AUC real, or an artifact of an easy benchmark? AUC measures ranking on the 105-pair set, whose negative
- `bank_composer.py` — bank_composer -- chain phrase-bank entries into a whole Van Rooten line. Each bank entry is an EN phrase that sounds like a FR phrase (a homophone unit). We bea
- `bench.py` — Offline benchmark: which homophone-matching method actually discriminates? Everything is deterministic and runs without API keys: - G2P + TTS: espeak-ng (determ
- `bigram_lm.py` — Word-bigram language model with stupid-backoff, for scoring phrase fluency on both the English and French sides of the homophone weave. Why: mean wordfreq-zipf 
- `build_dictionary.py` — Build a candidate cross-lingual homophone dictionary at scale, and measure how many *legitimate* word-to-word entries we can actually expect. This is the real f
- `build_dictionary_full.py` — Full-scale EN->FR homophone dictionary builder. Same architecture as build_dictionary.py (block with phoneme bigrams, rank with the benchmark-winning `combo` ma
- `build_poem_dataset.py` — Test: how to ACHIEVE the homophonic-poem dataset, given the books are in copyright. Reality check first. Mots d'Heures: Gousses, Rames (van Rooten 1967) and N'H
- `build_training_set.py` — Consolidate every legitimately-generated homophonic artifact into ONE instruction-style training set for the restructuring task (input English -> output French 
- `build_v6.py` — Build dictionary v6: v5's re-mining ethos + 1-segment filler words + the coherence/ranking fixes established this session. v5 ethos kept verbatim: real lexicon 
- `build_v7.py` — Dictionary v7 = best of v5 + v6. v5's method is the better base: RETRIEVAL (block the whole French lexicon, rank by the rule-rich combo), exhaustive, multiple c
- `build_v7_decode.py` — v7, efficiently: v5 IS the retrieval leg (richer than a re-run -- multiword, pair-bank, overrides), so DON'T redo retrieval. Run ONLY the single-phoneme decoder
- `cache_graph.py` — Build the unedited v7 graph once and pickle it so routing/analysis is instant. Imports chain_game.build_graph VERBATIM (no engine edit); min_sound=0 = full grap
- `cache_vecs.py` — Cache full-scope v7 node embeddings (same MiniLM the weave used) for on-demand semantic NN in the translation engine.
- `carve_quality.py` — Push carve quality: measure which levers actually raise the homophone score. Carve combo tops out ~0.5 on whole lines. We test, on a fixed line set, the levers 
- `chain_game.py` — Alternation chains: the translation-telephone game over the web. The user's spec (key = clé ≈ clay = argile ≈ ...): chains must ALTERNATE between meaning moves 
- `chain_hop.py` — Resolve a 'gap' word by HOPPING the chain until a useful FR landing. No word is a dead-end (giant component): walk ≈snd/=trans/~syn until you reach a fluent Fre
- `chain_paragraph_hop.py` — Long-chain paragraph translation over hops-all.tsv (June-12 chain_paragraph method, no embeddings). Per content word: best-product walk up to 40 hops from the w
- `chain_translate.py` — Chain-routed dual translation: every word transfers to French through an alternation chain, not a direct lookup. The user's architecture: transfer capacity come
- `compare_strategies.py` — Unbiased strategy comparison on the nursery-rhyme task. Runs the SAME public-domain Mother Goose lines through two extant strategies and scores them on the SAME
- `compose_smoke.py` — Composition smoke test using S/A (usable_for_composition) entries only. Takes English sentences, looks up each word in the working dictionary (usable entries, b
- `content_neighbours.py` — Content selection via Round Rabbit neighbours: the lever for lines that don't carve. The failing nursery lines (Jack and Jill, Hickory dickory dock) are dense i
- `corpus_bank.py` — Carve the PUBLIC-DOMAIN nursery-rhyme corpus into French -- our own legal homophonic gallery (the tradition's English is centuries old / PD; we generate the Fre
- `corpus_phrases.py` — Extract frequent bigram phrases from the trained bigram LMs, convert them to canonical IPA, and save as seed blocks for fragment_weave. Instead of seeding the w
- `cycle_consistency.py` — N1 experiment: cycle-consistency as a LABEL-FREE quality signal. Idea (DEPS_RABBIT_AND_NOVEL.md N1): a true homophone round-trips. Decode an English word's phon
- `dataset.py` — Labeled EN<->FR benchmark pairs for homophone-matching method comparison. Ground truth comes from linguistic knowledge, NOT from any scoring method: - word posi
- `demo_homophonic_writing.py` — Demonstration: end-to-end homophonic writing, sensical in BOTH languages. This ties the whole stack together and shows progress toward the goal: fragments (atte
- `dict_coverage.py` — PROOF 2: compare the real EN and FR dictionaries against our dataset, and match ANY dictionary word into the dataset. "Our dataset" = the v7 homophone dictionar
- `dual_reading_eval.py` — Dual-reading evaluation: the CORRECT objective for homophonic verse. Corrected target (per the project owner): the value is NOT two poems sharing a theme -- any
- `dual_translate.py` — Dual-track translation: literal AND homophonic, with a meaning-graded blend — the goal the user's original homophone-agent-audio skeleton named. For an English 
- `enrich.py` — Enrich the dictionary into the composition-ready v5 representation. The dual-language writing stage needs entries to be *chainable*: machine- readable alignment
- `explode_web.py` — Explode the chain-web so ALL steps are connection points, and certify the loop interiors as semantic+homophonic translation units. Answers two questions about c
- `finalize.py` — Finalize v5 into the composition-ready release the review specified: 1. junction columns surfaced in the TSV (they existed in JSON only); 2. gap/deletion ratios
- `fine_inventory.py` — Fine / mixed generation inventory: single phonemes + all 2-4 phoneme chunks + the full rule set (every phoneme incl. odd stops and the EN<->FR equivalence phone
- `fr_coherence.py` — French coherence scoring via a real LLM (the L2-model upgrade) -- multi-provider. Replaces the bigram fluency as a FINAL RE-RANK only (a handful of calls, not o
- `fragment_weave.py` — Fragment weave: grow a shared phoneme stream from sound-blocks, then read it off as real words in BOTH languages at once. Unbounded length, recursive, novelty-b
- `fragments.py` — Fragment layer: turn v5's alignments into a generative chunk grammar. Insight (user's): v5 entries are not just word matches — each alignment is a sequence of s
- `french_anchor.py` — French-anchor the homophone layer: symmetric to the English-anchored dataset. The v7 dataset is EN-headword -> FR-homophone, so the FR sound side is sparse (onl
- `function_words.py` — Rescue pass for English function words. The composition smoke test showed function words (the/is/of/and...) are the coverage holes: they are 1-3 segments long, 
- `generate.py` — Generative dual-language lines from pattern "lots". The user's spec: organize the material by structural pattern so a generative system can chain units of every
- `generate_pipeline.py` — Homophonic-writing generation: combine the best extant methods and test which combos / orders work best, cognate vs non-cognate. Light compute -- uses the alrea
- `generation_engine.py` — Resegmentation-under-coherence: the 1+2+3 generation engine. Builds ONLY on what June 11-12 perfected -- reinvents none of it: (1) RESEGMENTATION [reused: phone
- `homophonic_poet.py` — homophonic_poet -- GENERATIVE Van Rooten: the output really is a homophone. The fix to web_poet: a graph walk mixing sound+meaning hops does NOT read aloud as o
- `hoproute.py` — Unified hop routing + ideal hop distance for the translation engine. Two ideas the engine needs: 1. UNIFIED routing: at every hop take WHICHEVER edge is shortes
- `ladder.py` — The dual-reading LADDER: build writing where the English reading and the French reading SOUND the same and MEAN the same -- both rails unanchored. Implements th
- `learn_costs.py` — Learn the phoneme-substitution cost model from our own certified data. The matcher's equivalence table (rhotics 0.10, voicing 0.20...) is hand-curated. We now h
- `learned_coverage.py` — Learned coverage penalty: replace the blunt global deletion knob with a per-line tuned one, and fit a tiny model so it generalises. whole_line_carve.py raised e
- `lexicon_g2p.py` — Dictionary-based G2P from the user's recovered lexicon data (checkcehck/homophone/homophone-agent-audio): - data/lexique.tsv Lexique 3-derived French word -> IP
- `llm_remine.py` — LLM-in-the-loop remining: let the model READ our fragment data + current elision/liaison/allophonic rules and propose concrete improvements to how we carve Engl
- `lm_compare.py` — Show what the bigram LM does: generate ONE candidate pool, then rank it two ways -- by mean wordfreq-zipf (the old fluency) and by the word-bigram LM -- and pri
- `mapping_web.py` — Typed mapping web for v5 sound and meaning walks. Sound, fragment, and meaning edges stay separate. Walks can therefore say whether a result is a pure sound ech
- `matcher.py` — Reference implementation of the benchmark-winning homophone matcher. Method: `combo` — the synthesis that won the bench.py comparison (AUC 0.993, hard-negative 
- `mothergoose_full_test.py` — Unified Mother Goose test: every method, no cutting. For each public-domain Mother Goose line we GENERATE a French carve two ways (frontier whole-line carve + b
- `paraphrase_translate.py` — Paraphrase-search dual translation: unfreeze the English side. The measured ceiling of sound-first decoding (~0.49 sentence semantics) comes from fixing the Eng
- `phonetic_decoder.py` — Phonetic decoder: segment an English phoneme stream into French word sequences — the Knight & Graehl (1997) transliteration idea applied to homophone generation
- `phrase_bank.py` — Build a big PHRASE BANK of sensical homophone matches to compose from. For a large set of real English phrases (the most frequent English bigrams + public-domai
- `phrase_weave.py` — Phrase-to-phrase homophonic weave with a fluency prior on BOTH sides. Motivation (the gap this fills) ------------------------------- `dictionary-v5` has 1,299 
- `poetry_mode.py` — Van Rooten / poetry mode for the resegmentation engine. What "messing with the matching" on Humpty revealed: - the filler words van Rooten leans on -- un /œ̃/, 
- `prosody.py` — Prosody-aware sound scoring -- stress-weighted, and DIVERGED per language. Homophony lives in the stressed syllables; the ear forgives a fudged unstressed 2nd/3
- `rabbit_walk.py` — Round Rabbit Fix 1, given a go: min-hop BFS vs best-PRODUCT Dijkstra. DEPS_RABBIT_AND_NOVEL.md Fix 1 claims round_rabbit.py's bfs_component keeps the fewest-HOP
- `reach.py` — PROOF 1: any word can reach any word in the dataset. Loads the cached unedited v7 graph (graph-v7u.pkl) and proves connectivity three ways: 1. WCC weakly-connec
- `refine_dictionary.py` — Refined homophone dictionary (v3): lexicon pronunciations + pair bank. v3 on top of v2: G2P now comes from curated dictionaries first — Lexique 3 for French (24
- `rerank_v5.py` — An alternative to v5's word-for-word selection: re-rank each headword's FR choice by the matcher (combo) with a frequency tiebreak. Why: v5 picks the "best" FR 
- `retrieval.py` — The real test: can we BUILD a dictionary, not just classify given pairs? Classification (bench.py) scores a pre-selected pair. A dictionary is the inverse, much
- `retrieve_vs_decode.py` — Why v6 finds fewer/worse word-for-word matches than v5 even with looser gates. Hypothesis: v5 (build_dictionary.py) is RETRIEVAL -- block the whole French lexic
- `rhythm_channel.py` — N2 experiment: a PROSODY/RHYTHM channel for the matcher, measured honestly. Hypothesis (from DEPS_RABBIT_AND_NOVEL.md N2): every existing channel scores *segmen
- `round_rabbit.py` — Round-rabbit semantic/sound lattice. This builds the structure the poem generator was missing: 1. Collapse meaning edges into semantic components. 2. From every
- `round_rabbit_run.py` — Round Rabbit, branch-native: semantic anchor -> homophonic radius lattice, run on this branch's mapping-web.json (sound + meaning adjacency) with the best-PRODU
- `routes.py` — Q2: understand NODES and ROUTES over the whole v7 web (sound + meaning). Loads the cached unedited graph (graph-v7u.pkl: every node -> [(node, quality, label, f
- `run_all_tests.py` — Run the FULL test suite -- every method, symbolic + audio together, ffmpeg-free -- as one table. The final bake-off across the whole bench. Run: python run_all_
- `run_wav_tests.py` — Run the older Lingua Weaver SPEECH/WAV tests (bench.py audio methods) without ffmpeg -- espeak already writes WAV, and stdlib `wave` reads it. Patches bench.loa
- `reward.py` — Reward for self-learning homophonic carving. reward(en, fr) = sound_match (combo, our matcher) * french_validity (real French words) -- both LOCAL and free, so 
- `train_selflearn.py` — Self-learning homophonic carver: SFT warm-start then best-of-N self-improvement. Designed to run on ONE consumer GPU (Colab T4 / RunPod 3090-4090). Trains a sma
- `set_match.py` — Advanced set matching: chain English sets to French sets through the sound+meaning web, with multi-hop paths and optimal assignment. The user's idea: matching n
- `soramimi.py` — Sentence-level homophonic renderer: English text -> French word sequences that sound like it (the Van Rooten generator). Same decoder, whole-utterance phoneme s
- `sound_meaning.py` — Sound + meaning: grade every sound-pair by cross-lingual semantic similarity, so "sounds the same AND means something close" becomes a queryable column instead 
- `test_against_v5.py` — Test legitimate, openly-sourced homophone material against dictionary v5. Two open/moral sources only: - dataset.py positives: dictionary/etymology-documented E
- `test_mothergoose_gen.py` — Try to GENERATE homophonic Mother Goose with the reranked v5 table, and diagnose why it is hard. Public-domain source lines only (The Real Mother Goose, 1916). 
- `transfer_distance.py` — Q1: the OPPOSITE of a loop. Loops certify pairs whose meaning round-trips home (safe, but biased to near-cognate morphology like feel~files). The interesting ar
- `translate_engine.py` — Tiered homophonic+semantic translation engine. Input: an English phrase. For each content word, render it in French that SOUNDS like the English and MEANS the s
- `vanrooten.py` — Van Rooten joint search: ONE French line that SOUNDS like the English and is coherent French -- letting the ENGLISH SOURCE DRIFT semantically until a clean homo
- `weave.py` — Weave: loop all chains into one interconnected transfer web. Phase 1 (--stats): connectivity census of the full graph (sound + trans + semantic-kNN edges). Unio
- `web.py` — The mapping centre: one graph over EN and FR words with two edge types. sound edges usable v5 entries (weight = phonetic score; cognate edges carry cognate=true
- `web_coverage.py` — Coverage of EN/FR dictionaries against the FULL 195k web node set (homophone pairs + MUSE meaning layer), not just the homophone dataset side.
- `web_poet.py` — web_poet -- generative homophonic poetry by THEMED WALK on the v7 web. Outside-the-box move: don't translate a fixed sentence (that forces both rails to chain, 
- `webbing.py` — Webbing density of the dual-atom alphabet: how do the loop-tiles interconnect, and what are the best SETS to compose within? Two connection directions (the 'ver
- `whisper_improve.py` — Improving the Whisper path: replace the random-vector stub with REAL acoustic encoder features. Finding: artifacts/api-server/src/lib/whisper-phonetic-cluster.t
- `whisper_train.py` — Train the audio/Whisper path to match the symbolic results. We can't fine-tune neural Whisper here, but we can TRAIN a small learned ensemble of acoustic featur
- `whole_line_carve.py` — Whole-line coverage-aware carve: one French re-cut across an ENTIRE line, not per-English-window fragments. The fix to generation_engine's content_select: do NO

## Ideas living only on OTHER branches (worth porting)

- `research/homophone-bench/chain_analysis.py` @ `origin/claude/tender-lovelace-hvkftg` — Alternative chain analysis: explore what the current pipeline misses. Standalone — reads only from the TSV data files, never touches the existing scripts. Imple
- `research/homophone-bench/chain_compose.py` @ `origin/claude/tender-lovelace-hvkftg` — chain_compose.py — translation built AROUND chain_translate, fed with real synonyms (not context-drift), with poetic descriptions as the fallback. The breakthro
- `research/homophone-bench/chain_engine.py` @ `origin/claude/tender-lovelace-hvkftg` — chain_engine.py — the engine routed through the breakthrough: the woven chain-web, used to its furthest-developed form. Every content word transfers to the targ
- `research/homophone-bench/chain_methods.py` @ `origin/claude/tender-lovelace-hvkftg` — New chain methods: graduated tiers, density certification, deep fragments. Standalone — reads dictionary-v5.json, fragments.tsv, and the chain TSVs. Never impor
- `research/homophone-bench/chain_paragraph.py` @ `origin/claude/tender-lovelace-hvkftg` — chain_paragraph.py — translate a paragraph through chain hops. Uses the FULL chain-hop system: every content word transfers to French through alternation chains
- `research/homophone-bench/compose_lots.py` @ `origin/codex/v5-composition-mapping-web` — Multi-granularity composer for v5 material. Builds pattern lots from: - partial units: reusable fragment chunks; - whole units: single dictionary entries; - mul
- `research/homophone-bench/engine.py` @ `origin/claude/tender-lovelace-hvkftg` — engine.py — the drop-a-paragraph homophonic+semantic translation engine. python engine.py --text "the pale moon lights the quiet sea" python engine.py --pair en
- `research/homophone-bench/fragment_route.py` @ `origin/claude/github-projects-review-haxivl` — fragment_route.py — chunk tunnelling: transfer a word with NO whole-word sound row by routing its IPA chunks through fragment_edges and recomposing the French s
- `research/homophone-bench/function_glue.py` @ `origin/codex/v5-composition-mapping-web` — Targeted composition-only rescue for core function-word glue. The strict function-word decoder intentionally rejects very short weak forms when they do not clea
- `research/homophone-bench/homophone_bench/__init__.py` @ `origin/codex/v5-composition-mapping-web` — CLI package for the homophone bench release scripts.
- `research/homophone-bench/llm_layer.py` @ `origin/claude/github-projects-review-haxivl` — llm_layer.py — the fluency layer (LLM_RECIPE Job 3), DeepSeek-ready. The deterministic engine guarantees the SOUND and the per-word meaning options; it cannot m
- `research/homophone-bench/merge_generative.py` @ `origin/codex/v5-composition-mapping-web` — Merge vetted fragment-generated matches into dictionary-v5. The fragment probe writes candidates with a generator score. This pass turns them into audited dicti
- `research/homophone-bench/multilang.py` @ `origin/claude/github-projects-review-haxivl` — multilang.py — the button. One language-pair config in, the whole homophonic+semantic loop out. python multilang.py en es # build + mine + translate, en->es pyt
- `research/homophone-bench/periphrastic_translate.py` @ `origin/claude/tender-lovelace-hvkftg` — Periphrastic chain translation: when a word won't transfer, DESCRIBE it. The user's architecture, completed: greatest transfer comes from chain complexity, and 
- `research/homophone-bench/quarantine/chain_compose.py` @ `origin/claude/github-projects-review-haxivl` — chain_compose.py — translation built AROUND chain_translate, fed with real synonyms (not context-drift), with poetic descriptions as the fallback. The breakthro
- `research/homophone-bench/quarantine/chain_engine.py` @ `origin/claude/github-projects-review-haxivl` — chain_engine.py — the engine routed through the breakthrough: the woven chain-web, used to its furthest-developed form. Every content word transfers to the targ
- `research/homophone-bench/quarantine/engine.py` @ `origin/claude/github-projects-review-haxivl` — engine.py — the drop-a-paragraph homophonic+semantic translation engine. python engine.py --text "the pale moon lights the quiet sea" python engine.py --pair en
- `research/homophone-bench/quarantine/periphrastic_translate.py` @ `origin/claude/github-projects-review-haxivl` — Periphrastic chain translation: when a word won't transfer, DESCRIBE it. The user's architecture, completed: greatest transfer comes from chain complexity, and 
- `research/homophone-bench/query_chains.py` @ `origin/claude/tender-lovelace-hvkftg` — Query tool for chain/loop/pair TSV data files. python query_chains.py search cross # search all datasets for "cross" python query_chains.py chains cross # chain
- `research/homophone-bench/recursive_poet.py` @ `origin/claude/github-projects-review-haxivl` — Recursive semantic-sound poem generator over v5. This is deliberately not a lookup composer. It uses: - dictionary-v5.json: whole and multiword sound rows; - fr
- `research/homophone-bench/seed_learn.py` @ `origin/claude/github-projects-review-haxivl` — seed_learn.py — learn a pair's pronunciation-correspondence rules from its own mined seed dictionary. Answers "is the seed enough to learn the rules?" For any p
- `research/homophone-bench/smoke_test.py` @ `origin/codex/v5-composition-mapping-web` — Small smoke test for the composition-ready v5 release.

## Nemotron: underused ideas to experiment with next

1. Implement prosody-aware reranking using phrase-level pitch and duration contours from TTS to penalize homophones with unnatural speech rhythm, improving naturalness without semantic loss.  
2. Integrate semantic role labeling (SRL) to preserve thematic agent-patient relations during homophonic substitution, ensuring meaning coherence beyond surface fluency.  
3. Add phonotactic legality filters for French syllable structure (e.g., no word-initial /ŋ/) during candidate generation to eliminate phonologically implausible outputs.  
4. Experiment with contrastive learning on dual-reading pairs (literal vs. homophonic) to train embeddings that explicitly separate sound and meaning spaces.  
5. Use differentiable phoneme editing (via G2P gradients) to continuously optimize homophone paths toward higher semantic fidelity during beam search.  
6. Incorporate morphological segmentation (e.g., unsupervised morpheme splitting) to allow homophonic matching at subword level, increasing coverage for rare/compound words.  
7. Apply hierarchical reranking: first filter by phonetic similarity (n-best), then rerank by joint semantic+prosodic score using a lightweight transformer scorer.  
8. Leverage cross-lingual puns and wordplay databases (e.g., from Oulipo or schizophonia) as priors to bias generation toward linguistically valid homophonic constructs.