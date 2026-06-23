# Roadmap: from the v5 dictionary to a multilingual homophone generator

Research + planning doc, 2026-06-22. No new generator code here — this reads the
whole codebase, reconstructs how v1→v5 happened, distills the invariants those
experiments paid for, and lays out the highest-leverage avenues forward, with a
north-star architecture for a *multilingual* ("multi-LM") homophone writer.

Sources read: `RESULTS.md`, `DICTIONARY_FEASIBILITY.md`, `REPRESENTATION.md`,
`LLM_RECIPE.md`, `LM_COMPARISON.md`, `FRAGMENT_WEAVE_RESULTS.md`,
`HOMOPHONE_PRODUCTION_DEMO.md`, the `matcher` / `phonetic_decoder` /
`refine_dictionary` / `finalize` / `weave` / `fragments` / `fragment_weave` /
`bigram_lm` source, and the dictionary v2–v5 JSON.

---

## 1. What actually happened, v1 → v5 (and the layer after)

The project is not one artifact — it is a **stack of layers, each built because
the previous one exposed the next bottleneck.** The dictionary version number
only tracks the middle of that stack.

### Layer 0 — the matcher (the substrate). `bench.py`, `matcher.py` (Jun 10)
The founding question was "which homophone scorer actually works?" 16 methods,
105 hand-labeled EN↔FR pairs, ground truth from linguistics not from a scorer.
Winner: `combo` = ½·phoneme-bigram-Dice + ½·sharpened-featural-NW, **AUC 0.993**.
Three findings that everything else inherits:
- **symbolic beats audio/neural decisively** — every MFCC/Allosaurus method was
  worse, and Allosaurus on synthetic speech *collapsed to 0.713*;
- **the "sharpen" trick** (÷0.35 clamp on panphon distance) was the single
  biggest lever in the study (feat-NW 0.941→0.993);
- **two channels with independent error modes** beat any single one.

### v1 — full-scale naive build. `build_dictionary_full.py` (Jun 10)
15,366 entries by blocking the whole EN lexicon against the whole FR lexicon.
Proved scale was possible and immediately exposed the **noise floor**:
~82 French words clear the 0.45 classification threshold *per query*. A
threshold-built dictionary is ~98% garbage (`DICTIONARY_FEASIBILITY.md`).
Lesson: dictionary-building is **top-k retrieval with a high bar**, not
classification. The 0.993 AUC measured an easier task than the real one.

### v2 — refinement & hygiene. `refine_dictionary.py` (Jun 10) → 8,319 entries
*Fewer* entries than v1, deliberately. Added: G2P overrides for the ~218 French
words espeak language-switches on (menthe, rythme, dos…); multiword FR phrases
(`mayday ~ m'aider`) with a per-word frequency gate so phrases stay natural;
cognate tagging via the MUSE bilingual dictionary (the free shared-sound-AND-
meaning subset); NW **alignment decomposition** on kept entries; B-tier hygiene
(best-per-word + require a shared bigram). The dictionary got smaller and
*much* cleaner.

### v3 — real lexicons + the equivalence layer. `lexicon_g2p.py` (Jun 11) → 9,684
Replaced espeak-only G2P with curated lexicons: **Lexique 3** (241k FR) and
**WikiPron** (65k EN, UK + US variants). Ported the **EN↔FR equivalence-cost
layer** (p~b, i~ɪ, θ~s, rhotic map, nasal split) and cheap deletions
(offglide/schwa/h) into the matcher as a floor under the sharpened distance.
Merged a 30k historical pair-bank, provenance-tagged. This is where the matcher
became genuinely *cross-lingual* rather than a generic phone-distance.

### v4 — the decoder (generation, not just retrieval). `phonetic_decoder.py` (Jun 11) → 10,652
The pivot from *lookup* to *generation*. A **Knight & Graehl transliteration
decoder**: a pronunciation **trie** over the full Lexique + **beam search** that
walks an English phoneme stream and reads off French word *sequences* —
multi-to-multi, with word-penalty/frequency-bonus language dynamics and liaison
arcs. Also fixed the Lexique ASCII-/ɡ/ corruption (every g-word had been indexed
without its /ɡ/). This is the engine `fragment_weave` and the demo still use.

### v5 — composition-ready representation. `finalize.py` (Jun 11) → 11,788
Made the dictionary **chainable**. Every entry got a machine-readable `align`
(segment pairs + costs) — up from display strings on 29% to **100%** — plus a
`pivot` skeleton (CVC…), junction fields (`en_onset/coda`, `fr_onset/coda`,
`en_syll/fr_syll`), `direction`, and the `usable_for_composition` acceptance
rule (5,596 entries pass). The representation verdict (`REPRESENTATION.md`):
**store orthography + broad IPA + equivalence-class pivot; never allophones,
never anchor on audio.** Allophony is *code paths at match time*, not data.

### The layer after v5 — generative grammar + meaning + the honest negative (Jun 12)
- **`fragments.py`** turned v5 alignments into a chunk grammar: extract attested
  EN→FR sound-blocks; the high-count ones are *sound-identical* across languages
  (st×372, ɹi×244), so chaining them yields one IPA stream readable as both.
- **`sound_meaning.py`** graded all 5,596 usable pairs with multilingual
  embeddings: **1,811 sound-same AND mean-same-or-close**, of which 1,117 are the
  "close" prize band (`pled~plaide`, `teepee~tipi`, `to~tout`).
- **`weave.py`** showed the transfer graph is **one giant component** (99.58% of
  194k nodes), 7,285 chain-web edges, 539 loops.
- **The honest negative** (`b5a985a`): re-weaving after promoting loop-certified
  pairs returned *identical* numbers. Certification re-labels existing topology;
  it cannot add edges. **Growth must come from re-mining, not re-labeling.**

### This session (Jun 20–22) — fluency in the loop
`phrase_weave` (dual fluency prior) → `fragment_weave` (recursive, unbounded,
novelty-biased) → `bigram_lm` (stupid-backoff word bigram, replaces mean-zipf
"any common words in any order are fine") → **LM wired into the decoder beam** +
corpus-bigram phrase seeds → end-to-end demo with arbiter verification. Measured
state: 70/117 generated pairs pass the AUC-0.993 arbiter; 37 dictionary
word→phrase homophones are sound-true AND fluent; **phrase↔phrase with both
sides fluent is the frontier**, gated by French phrase-fluency.

---

## 2. The invariants (what the experiments already settled — don't relitigate)

These are paid for in negative results. Any future design must respect them.

1. **Symbolic phonetics, always.** Broad IPA is the accent-independent
   interlingua; audio/ASR is a *validator at most*, never a source
   (`REPRESENTATION.md` §2). Quantitatively confirmed twice.
2. **The LLM never invents or scores sound.** It hallucinates IPA and overrates
   spelling. Give it *meaning* work only — propose synonyms, judge sense, order
   units (`LLM_RECIPE.md`). Sound stays deterministic.
3. **Proposal grammar + independent arbiter.** Every generative layer (decoder,
   fragments, weave, fragment_weave) proposes; the matcher re-scores. The arbiter
   was tuned on hand labels, never on generator output — that's what makes a high
   score on a generated pair real evidence.
4. **Top-k with a high bar, not a threshold.** The noise floor is real; ~82
   false matches clear 0.45 per query.
5. **Growth = re-mining, not re-labeling.** Closed loops re-label topology that
   already existed. New edges come from running the decoder with a better matcher
   / bigger trie.
6. **Three lexicons, three sizes.** shared-sound (abundant, ~14/100 words) ≫
   shared-sound-AND-meaning (rare, ~4%) ≫ forced phrase puns (need a human).
   Name which one a goal targets before optimizing.
7. **No-hindsight discipline** (borrowed from poly-microtrader): labels
   independent of scorers, held-out splits, only the sharpen/gap constants tuned.

---

## 3. Where we actually are (the measured frontier)

| capability | state | evidence |
|---|---|---|
| EN↔FR sound scoring | **solved**, AUC 0.993 | `RESULTS.md`, reproduced this session |
| EN word → FR phrase homophones | **strong**, 37 arbiter-confirmed sensical | `HOMOPHONE_PRODUCTION_DEMO.md` |
| EN fixed → FR free (Van Rooten) | **working** | `soramimi.py` |
| FR fixed → EN free | **working** (104 entries) | `phonetic_decoder --reverse` |
| sound + meaning lexicon | **graded**, 1,811 pairs | `sound_meaning.py` |
| phrase ↔ phrase, both fluent | **frontier**, ~3 clear strict bar | this session |
| both-free meaning-on-both (regime 3) | **designed, unbuilt** | `REPRESENTATION.md` §3 |
| multilingual (beyond EN↔FR) | **architecturally ready, untested** | matcher is voice-agnostic |

The bottleneck is no longer *sound* and no longer *retrieval*. It is **fluency
and meaning on the generated side(s)** — exactly the two things the deterministic
pipeline can't supply and the LLM/LM must.

---

## 4. North star — a multilingual homophone generator

Picture the destination so the avenues have somewhere to point. The general
object is a **weighted lexicon transducer per language** plus a **shared
phonetic interlingua**, searched under fluency + meaning priors:

```
Given source language L1, target language L2, and (optionally) a seed
meaning/topic, emit text T1 in L1 and T2 in L2 such that:
    pronounce_L1(T1) ≈ pronounce_L2(T2)        (phonetic_cost low — our arbiter)
    LM_L2(T2) high                              (T2 reads as real L2)
    LM_L1(T1) high                              (T1 reads as real L1)
    optionally  embed(T1) · embed(T2) high      (they also mean something close)

score = λ1·LM_L1(T1) + λ2·LM_L2(T2) − φ·phonetic_cost + μ·meaning(T1,T2)
```

Why this is reachable as *multi*-lingual and not just EN↔FR:
- the matcher's substrate (panphon features + sharpen) is **already language-
  agnostic**; only the EQUIV floor is hand-curated per pair;
- pronunciation lexicons exist for many languages (Lexique-style / WikiPron /
  espeak fallback);
- **multilingual embeddings already put all languages in one meaning space**, so
  `meaning(T1,T2)` generalizes for free;
- the cost of adding language N is **O(1) per language**, not O(N) per pair, *if*
  we route through a shared interlingua instead of building N² equivalence tables.

The "multi-LM" reading is literal too: the loop needs **one fluency LM per
language** in the beam (cheap, deterministic) and **one shared LLM** as the
meaning proposer/judge across all of them.

---

## 5. The avenues, by leverage

Ordered by expected payoff per unit effort, each with the idea, why it's the
right lever, the concrete first step, and the failure mode to watch.

### A. Fluency: replace the bigram LM with a real n-gram on a modern corpus  ★ highest
**Idea.** The measured frontier is French phrase-fluency, and the current scorer
is a stupid-backoff *bigram* on **19th-century novels** (Monte Cristo, Candide).
It under-scores every colloquialism (`c'est`, `il y a`, `qu'est-ce`) — noted in
both `LM_COMPARISON.md` and this session's results. Swap in a **KenLM 4–5-gram**
trained on a **modern, broad** corpus (OpenSubtitles + a Wikipedia/OSCAR slice)
per language.
**Why it's the lever.** This is the *direct* attack on the one thing blocking
phrase↔phrase. It is the 2026 equivalent of the "sharpen" trick: a single
component swap that should move the bottleneck metric the most. Keeps determinism.
**First step.** Build `corpus/` from OpenSubtitles EN+FR; train KenLM; expose the
same `fluency(words)→[0,1]` interface `bigram_lm` already has, so it drops into
the beam (`phonetic_decoder.decode(lm=…)`) and `fragment_weave` unchanged.
**Failure mode.** KenLM needs a binary/build dep; if the env can't have it, fall
back to an in-process trigram with Kneser-Ney. Subtitle text is noisy — clean
speaker tags / SDH markers first.

### B. Search: bidirectional anchored generation instead of free chaining  ★ high
**Idea.** Today `fragment_weave` grows a *random* IPA stream and hopes both
decodes are fluent — most aren't. Instead **anchor on a meaningful content word**
on one side (sampled from mid-frequency nouns/verbs, not function words), decode
the other side, then **alternate** (anchor next on the other language) so both
sides are pinned to real words as the stream grows.
**Why it's the lever.** It changes the prior from "common-word salad that happens
to decode" to "two real phrases that happen to align" — attacking sensicalness at
the *generation* step rather than filtering after (the explicit lesson of
`LM_COMPARISON.md` §"the real finding"). Reuses the existing beam.
**First step.** Add an `anchor=` mode to the grower: pick a seed word with an S/A
v5 entry, fix its IPA span, beam-decode the complement on the other side, score
with arbiter × both-LMs. Compare yield vs the random grower on the same budget.
**Failure mode.** Anchoring can over-constrain and starve the beam; needs a
relaxation knob (allow the anchor span ±1 segment).

### C. Regime 3: build the constrained generation loop with the production LLM  ★ high
**Idea.** The actual goal ("sensical in both at once, neither side given") is the
**propose → render → judge** loop already specified in `REPRESENTATION.md` §3 and
`LLM_RECIPE.md` Job 3 — and never built. The LLM proposes an L1 line *under the
v5 vocabulary constraint* (only words/phrases that have S/A entries or decode
cleanly), our decoder renders top-k L2 candidates, the LLM judges L2 for
sense/grammar, keep Pareto-best (L1 sense, L2 sense, sound), mutate.
**Why it's the lever.** It is the only design that can satisfy *meaning on both
sides*, and it respects invariant #2 (LLM never touches sound). v5's junction
fields are exactly the constraint table it needs.
**First step.** A thin harness: (1) export the v5 S/A entries as a constraint
vocabulary keyed by pivot/junction; (2) one LLM call to propose 20 L1 lines from
that vocabulary on a theme; (3) `phonetic_decoder` renders FR for each; (4) one
LLM call to gloss+rate the FR. Measure how many clear arbiter ≥ 0.55 AND both
LLM-sense ratings ≥ 7.
**Failure mode.** Constraint vocabulary too small → stilted L1. Mitigate by
including B_safe entries and letting the LLM use any function words freely (they
decode trivially).

### D. Meaning: grow the sound+meaning set by synonym-bridging  ★ medium
**Idea.** `LLM_RECIPE.md` Job 2: for `related`-band pairs, ask the LLM for 5 L1
synonyms and 5 L2 synonyms, run the **deterministic matcher** over the 25 cross
combinations, keep any that clear the bar. The LLM proposes meaning-preserving
moves; the pipeline verifies sound.
**Why it's the lever.** It expands the *rare* lexicon (shared-sound-AND-meaning)
without hindsight and without the LLM scoring sound. Upgrades `two~tout`
(related) toward genuine sound+meaning hits.
**First step.** Run Job 2 over the 2,135 `related` rows in `sound-meaning-v1.tsv`;
new entries inherit full QC and the cognate tag.
**Failure mode.** Synonym lists drift in register; gate new entries by the same
zipf/fluency bars as v5.

### E. Multilingual generalization: route through a shared interlingua  ★ medium, strategic
**Idea.** To become genuinely multi-LM, stop hand-writing an EQUIV table per
pair. Define a **shared equivalence-class space** (the matcher already implies
one) and map each language's phoneme inventory **into** it once. Cross-pair cost
then = distance in interlingua space, derived not curated. `learn_costs.py`
already learns costs from certified alignments — generalize it to emit a
language→interlingua map per language.
**Why it's the lever.** Turns O(N²) curation into O(N). It is the precondition
for "any language pair" rather than "EN↔FR plus heroics."
**First step.** Pick a third language with a good lexicon (Spanish: large
WikiPron + clean orthography). Build ES G2P + a fluency LM, *derive* ES↔EN and
ES↔FR floors from panphon + a small labeled set, and re-run the matcher
benchmark on a 50-pair ES↔EN set. If AUC holds without a hand EQUIV table, the
interlingua route is validated.
**Failure mode.** Some pairs need genuinely language-specific rules (tone, vowel
harmony); keep a per-pair override slot over the derived floor.

### F. Validation: Whisper round-trip as a soft re-ranker  ★ low, cheap insurance
**Idea.** On *finished* compositions only, synthesize the L2 side with several
voices, run L1 ASR, check it transcribes back near the L1 side. Soft re-rank,
never load-bearing (`REPRESENTATION.md` §2).
**Why it's the lever.** Catches the cases where symbolic IPA and real acoustics
diverge (espeak G2P errors). Cheap in the production app, which has TTS already.
**Failure mode.** ASR's internal LM hallucinates toward plausible text — use
multiple voices + fuzzy match, treat as a veto-only weak signal.

### G. Substrate polish: the known matcher failure classes  ★ low, finish-the-job
From `DICTIONARY_FEASIBILITY.md` §"known failure modes":
- short diphthong words (`dough/dos`, `low/l'eau`) — weight diphthong-smoothing
  higher for ≤2-segment words;
- nasal vowels (`ant/an`) — also try dropping the nasal consonant, not just split;
- add a frequency/legitimacy prior to break ties among sound-equal candidates;
- stress/prosody is stripped — keep a parallel stressed EN tier (WikiPron has it)
  for line scansion.
These are small, well-scoped, and each removes a documented miss class.

---

## 6. Suggested sequence

1. **A (modern n-gram LM)** — unblocks the measured frontier; everything
   downstream scores better immediately. Do first.
2. **B (anchored bidirectional search)** — with a real LM in the beam, anchoring
   becomes worth it; together A+B is the realistic path to fluent phrase↔phrase.
3. **C (regime-3 LLM loop)** — once A+B make the deterministic core produce
   fluent candidates, the LLM loop has good raw material to steer toward *meaning
   on both sides* (the actual goal).
4. **D (synonym-bridge)** and **G (substrate polish)** in parallel — independent,
   low-risk lexicon/quality gains.
5. **E (interlingua / 3rd language)** — the strategic bet that turns this from an
   EN↔FR system into the multilingual generator. Start it as a spike behind A–C.
6. **F (round-trip)** — add last, as QC on finished output.

The one-line thesis: **sound is solved; the next dollar goes to fluency (A),
then to a generation prior that pins both sides to real words (B), then to an
LLM loop that puts meaning on both sides (C) — and the interlingua (E) is what
makes all of it multilingual instead of bilingual.**
