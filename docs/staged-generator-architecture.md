# The staged generator — five stages, separate training, no translator

The reframe: this is a **generator, not a translator**. At no stage does
anyone type "she walks in beauty" and receive panphon nonsense back. The
system grows outward from the small verified core — the gold words and
multi-words that have *proven* homophone matches — and tries to replicate
that corpus across the entire dictionary of both languages, each language
expanded on its own, with cross-language truth applied only as a filter.

Each stage trains and runs **separately**, on its own data, with its own
success measure. Most stages already have engines on GitHub
(`claude/phrase-weave-multiword` branch, `research/homophone-bench/`); this
doc maps them and specifies the training for the one stage that needs a GPU.

```
STAGE 1  gold data, kept in place            (tier_ladder.py — done)
STAGE 2  dual ladder: monolingual meaning    (ladder.py, routes.py — mostly done)
STAGE 3  carve training: monolingual GPU     (the new model — this doc §3)
         expansion, EN and FR independently
STAGE 4  cross-language filter + feedback    (matcher/strict_judge/Agent B — done)
STAGE 5  paragraph composition               (bicameral/dual_scale — exists)
```

---

## Stage 1 — the gold stays in place

Nothing is retrained here. The vetted sources are the immutable substrate,
and **`tier_ladder.py` already unifies them**: v5 S/A/B, v7 GOLD/A/B,
STRICT-GOLD (9,803), loop-certified atoms (×count), DUAL-S/A/B — one ranked
dataset (`tier-ladder.tsv`), provenance preserved per column. Plus
`fragments.tsv` (proven EN↔FR sound-blocks) and the phrase bank.

Stage 1's only job going forward: **absorb stage-4 survivors**. New verified
pairs enter as a new provenance column, never overwrite.

## Stage 2 — the dual ladder: add meaning, one language at a time

For every word, in **its own language**: its homonyms (if any), synonyms to
swap to when the word doesn't fit, and paths toward other words. No
cross-language content — this is pure semantic scaffolding, built once,
stored as data.

| need | extant engine |
|---|---|
| typed slots (≈ / = / ~ kept apart, never flattened) | `ladder.py` §1 |
| sense-split (polysemy clustered by embedding) | `ladder.py` §4 |
| paths toward other words (routing, hubs) | `routes.py`, `hoproute.py` |
| synonym chains | `chain_game.py`, MUSE vectors in `weave.py` |

**The one missing piece: within-language homonym sets** (sea/see/C;
vert/verre/vers/ver). Cheap to build — group the EN lexicon and the FR
lexicon each by identical g2p output (`lexicon_g2p.py`), emit
`homonyms-en.tsv` / `homonyms-fr.tsv`. These matter because a homonym is a
*free semantic pivot*: if "sea" doesn't fit the sentence, "see" occupies the
same sound cell and reopens the meaning search. The ladder gains a fourth
bucket: `homonym(≡)` — same language, same sound, different meaning.

Output: one JSON ladder per word — `{word, lang, homonyms[], synonyms[],
senses[], routes[]}` — the context object stages 3 and 5 consume.

## Stage 3 — carve training: replicate the gold corpus across the dictionary

The main GPU stage, and the key discipline: **EN and FR are expanded
independently** — don't worry about the other language at all here. The model
never sees a translation pair. Its task: take the gold subset of words with
proven homophone behavior and generate the *rest of the dictionary's worth*
of candidate material in the same style — conjugations, Zipf-common
inflections, paraphrase variants, multi-word weaves.

### Split what's deterministic from what's learned

**Don't burn GPU learning conjugation tables that exist as data.** Morphology
is a lookup, not a skill:

- French: **Lefff** morphological lexicon (~500k inflected forms, free) —
  every conjugation/agreement of every gold FR word, deterministically.
- English: **UniMorph / LemmInflect** — same for EN.
- Weight every generated variant by **Zipf frequency** (`wordfreq` or the
  existing `zipf_glue.py` machinery) so "walks/walked/walking" outrank
  "walkest". This is the stage-3 expansion's backbone, and it's free.

The **LLM's** job is what lookup can't do — the three generative moves:

1. **Paraphrase-wide** (per language): given a gold unit and its stage-2
   ladder, produce n rewordings that keep the meaning but change the sound
   material. Engine already exists: `paraphrase_search.py` step 1 does
   exactly this via LLM; the training version distills it into a local model
   using the gold-tiered corpus's paraphrases as SFT targets.
2. **Zipf-phrase generation**: produce frequent, natural short phrases
   (checked against books/bigrams — `bigram_lm.py`, PD corpus) that embed the
   expanded vocabulary. `zipf_glue.py` already mined the function-word glue
   these phrases need.
3. **Multi-word weaving**: chain proven sound-blocks into longer units —
   `fragments.py` / `fragment_weave.py` / `phrase_weave.py` are the extant
   engines; the model learns which weaves read as natural language (the part
   the tries can't judge).

### Training recipe (per language, two models or one dual-headed)

- Model: Qwen3-4B LoRA (the `research/qwen-finetune/` kit runs unchanged —
  only the data builder differs). Monolingual generation with the ladder in
  the prompt:

```json
{"messages": [
 {"role": "system", "content": "Expand the unit into natural {lang} variants: inflections in context, paraphrases, short frequent phrases. Stay meaning-true. One per line."},
 {"role": "user", "content": "UNIT: chaude\nLADDER: homonyms=[] synonyms=[chaleureuse, brûlante] senses=[hot/warm]"},
 {"role": "assistant", "content": "chaude\nchaudes\nla nuit chaude\nune chaleur douce\n..."}]}
```

- SFT targets: gold-corpus paraphrase sets + Lefff/UniMorph inflections
  placed into real PD-corpus contexts. Fluency check at build time =
  `bigram_lm.py` + book-sentence n-gram membership (books/bigrams as the
  "does anyone actually say this" test).
- Then the verifier loop (`sample_and_dpo.py`) with a **monolingual** reward:
  `zipf_weight + fluency + semantic_cos(unit, variant)`. No sound term — sound
  is stage 4's business. Keeping the rewards separated is what makes the
  stages separable.

Output: a large candidate store per language —
`expansion-en.tsv` / `expansion-fr.tsv` (unit → variants ×n, Zipf-ranked).

## Stage 4 — the cross-language filter (the temporarily-forgotten truth)

Stages 2–3 are deliberately meaning-only; stage 4 re-imposes the other
language. Every stage-3 variant, EN×FR, is asked one question: **does it
actually work as sound in the other language?** All extant machinery:

- `matcher.py` combo (AUC 0.993) — the verdict.
- `strict_judge.py` geo-ensemble + beats-nearest-rival — the strict tier.
- Agent B (the English/French ear) + Agent C's two-comparison judge
  (`three_agent_v2.py`) — the behavioral check.
- `sandhi_fr.py` — score the *spoken* stream so liaison/elision count.

**Partial credit is mandatory**: even when a whole variant fails, score every
sub-span (the carve machinery already segments); harvest working sub-spans
into `fragments.tsv` and the phrase bank. A 7-word phrase with a perfect
3-word core contributes the core.

**The feedback loop is the point**: survivors → stage 1 (new provenance
column) → stage 3's next SFT round trains on them. Each cycle the gold core
grows, so the next expansion starts from more coverage. This is the same
verifier-flywheel as the DPO kit, but at corpus scale — the system
replicates its gold corpus toward full-dictionary coverage by iterating
expansion→filter, never by translating test sentences.

## Stage 5 — paragraphs

Composition over the enlarged, verified inventory. Extant engines:
`bicameral_paragraph(_v2).py`, `dual_scale_composer.py`,
`recursive_composer.py`, `round_rabbit_bicameral.py`, `bank_composer.py`.
They get better automatically as stages 3–4 grow their alphabet; the stage-5
specific work is seam fluency (the composer LLM from the sentence plan) and
prosody, and it should not start until a stage-4 cycle has actually enlarged
the bank — composing over 813 atoms is the ceiling already measured
(`DUAL_LADDER.md`: both-side fluency ~0.14).

---

## Build order (what's actually new)

1. **Homonym sets** (stage 2's missing bucket): group each lexicon by g2p —
   an afternoon, no GPU.
2. **Deterministic expansion**: Lefff + UniMorph + Zipf weighting over every
   gold unit → the first `expansion-{en,fr}.tsv`, no training yet.
3. **Stage 4 pass over that** free expansion — likely yields thousands of new
   verified pairs before any model is trained (inflected forms of gold words
   are the highest-probability new homophones there are: if `rives` works,
   `rive/rivait` are the first place to look).
4. **Stage 3 SFT** on the enlarged gold (paraphrase + phrase generation), then
   the monolingual DPO loop.
5. Iterate 3→4→1. Watch one number: **verified-pair count by tier** after
   each cycle. If a cycle stops adding STRICT-GOLD pairs, the expansion has
   outrun the filter's mercy and it's time to improve stage 3, not run more.
