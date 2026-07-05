# Representation & architecture for dual-language composition

Answers to: "what's the structure afterwards — what representation is most
accurate, allophone strings? — could TTS/Whisper split-results work? — and
can the data be improved?" Written 2026-06-11 against dictionary v5.

## 1. The representation verdict

**Store: orthographic words + broad canonical IPA + equivalence-class pivot.
Do NOT store allophones. Do NOT anchor on audio.**

Three layers, each doing one job:

| layer | example (`said ~ cède`) | job |
|---|---|---|
| orthography | `said` / `cède` | the deliverable — writing, as you said, is where it must land |
| broad IPA segments | `sɛd` / `sɛd` | the matching substrate, from curated lexicons (Lexique/WikiPron) |
| pivot + alignment | `CVC`, `[[s,s,0],[ɛ,ɛ,0],[d,d,0]]` | the composition substrate — what makes entries chainable |

Why **not allophones**: an allophone string ([ɾ] for /t/ in American
*butter*, dark [ɫ], aspirated [tʰ]) is a *realization* — exactly the
accent/idiolect variance you correctly said the system can't depend on.
The right move is what the matcher already does: store **broad phonemes**,
and handle allophony at *match time* as rule variants and equivalence costs
(flapping, schwa elision, h-dropping, liaison). Allophones are code paths,
not data. This is also why the benchmark's symbolic methods beat every
audio method: broad transcription IS the accent-independent abstraction.

Why **one pivot, not two phoneme inventories**: EN /uː/ and FR /u/ are
different phonemes but one match. The equivalence-class space (the EQUIV
table + sharpened features) is the real interlingua — the `pivot` field
("CVC", with `·` for unmatched residue) is its coarse index, and the
`align` field is its exact record.

## 2. The TTS/Whisper idea, honestly assessed

Your instinct ("find audio that ASR splits validly into both languages")
is a real architecture — but as a **validator, never a source**:

- Whisper's decoder has a strong internal language model: it *hallucinates
  toward* plausible text in whatever language it's told. It would "find"
  dual readings driven by its LM, not by the acoustics — unfalsifiable.
- The result depends on the synthetic voice's accent — your own objection,
  and our benchmark confirmed it quantitatively (every audio judge scored
  worse than symbolic, and cross-voice ensembling didn't fix it).
- It's non-deterministic and unscalable per pair (seconds vs microseconds).

The salvageable version, for later: a **round-trip filter** on finished
compositions — synthesize the French side with several FR voices, run
English Whisper, and check it transcribes back to (near) the English side.
Use as a soft re-ranking signal on candidates the symbolic system already
produced, comparing transcripts with the **same featural `combo` matcher**
(not raw string/edit-distance — fuzzy matching is out of the schema). Cheap
to add in the production app (it already has TTS infra); never load-bearing,
and never on the default path.

## 3. The structure afterwards: dictionary as transducer, three search regimes

The dictionary is best understood as a **weighted lexicon transducer**
(Knight & Graehl's framing): each entry is an arc `EN words ↔ FR words`
with a phonetic cost and junction constraints. "Sensical writing in both
languages at once" is then search over paths, and there are exactly three
regimes:

1. **EN fixed → FR free** (done: `soramimi.py`). The English text's phoneme
   stream is decoded into French word sequences. This is the Van Rooten
   direction — for "Humpty Dumpty sat on a wall" the decoder already finds
   *"aiment étain petits étonné"*, the same sound-family as Van Rooten's
   real *"Un petit d'un petit s'étonne aux Halles"*. The French is
   word-legitimate but not yet sensical — that's regime 3's job.
2. **FR fixed → EN free** (done: `phonetic_decoder.py --reverse`, 104
   entries merged). Same machinery, English pronunciation trie.
3. **Both free — the grand goal.** Neither side is given; you want meaning
   on both. This cannot be solved by phonetics alone because "sense" lives
   in a language model. The architecture is a **constrained generation
   loop**, and it is exactly the shape of the reservoir-mining loop already
   in the production app:

   ```
   score(EN, FR) = λe·LM_en(EN) + λf·LM_fr(FR) − φ·phonetic_cost(EN, FR)

   repeat:
     LLM proposes EN line under a VOCABULARY CONSTRAINT
        (words/phrases that have S/A entries or decode cleanly — v5 is
         the constraint table, keyed by pivot and junction fields)
     decoder renders top-k FR candidates        (deterministic, ours)
     LLM judges FR candidates for sense/grammar (cheap rubric call)
     keep Pareto-best (EN sense, FR sense, phonetic score); mutate
   ```

   The deterministic decoder in the middle is what your previous attempts
   were missing: the LLM never has to *invent* the phonetics (it's bad at
   that); it only proposes and judges meaning, which is what it's good at.

## 4. What the junction fields are for

Chaining entries into lines is where naive concatenation fails, so v5
records per entry: `en_onset/en_coda/fr_onset/fr_coda` (segment + V/C
class) and `en_syll/fr_syll`. Composition rules then read directly off
the data:

- **hiatus check**: FR coda V + FR onset V across a join is un-French
  (this was the `when ~ ouais haine` failure class) — require C|V or V|C,
  or insert a liaison arc;
- **liaison arcs**: entry chains may insert /z t n/ exactly where the
  decoder already licenses them (spelling-conditioned);
- **rhythm budget**: EN and FR syllable counts must track each other
  (|en_syll − fr_syll| ≤ 1 per chained unit keeps scansion).

## 5. Data improvements made this round (and what's still missing)

Done now (v5, 10,756 entries):
- machine-readable `align` (segment pairs + costs) on **100%** of entries
  (was: display strings on 29%);
- `pivot` class skeleton for indexing/dedup/constraint lookup;
- junction features and syllable counts both sides;
- `direction` field; 104 reverse (FR→EN-phrase) entries merged;
- earlier this session: Lexique ASCII-/ɡ/ corruption fix (every g-word in
  the database had been indexed missing its /ɡ/).

Known gaps, in priority order for the composition stage:
1. **Stress/prosody is not modeled** — we strip ˈ/ˌ. For single words it
   barely matters; for lines it's the difference between scansion and
   mush. Fix: keep a parallel stressed tier for EN (WikiPron has it) and
   treat FR as phrase-final stress.
2. **No FR language model in the loop yet** — zipf unigrams stand in for
   naturalness. The production app's LLM is the intended replacement.
3. **fr→en coverage is thin** (104) — the EN trie has 65k WikiPron words
   vs 241k Lexique; adding CMUdict-IPA (~130k) would roughly double it.
4. **Syllable-boundary alignment** is approximate (nucleus counts, not
   onsets/codas per syllable); proper syllabification would sharpen the
   rhythm budget.
