# ⭐ THE PIPELINE — canonical map of the homophonic generator

> **READ THIS FIRST.** This is the top-level map of how the system is built
> and trained. It is a **generator, not a translator**: at no stage does
> anyone type in a sentence and receive a phonetic gloss back. The system
> grows outward from a small verified gold core and replicates it across the
> whole dictionary of both languages. Every stage trains separately, on its
> own data, with its own reward. *(2026-07-06)*

Code lives in `research/homophone-bench/` (branch `claude/phrase-weave-multiword`)
and `research/qwen-finetune/` + `docs/` (branch `claude/three-agent-homophone-t51ats`).
Deep dives: `docs/staged-generator-architecture.md`,
`docs/sentence-generation-plan.md`, `docs/qwen-three-agent-schema.md`,
`research/homophone-bench/read me first`.

```
STAGE 1   GOLD, IN PLACE          the verified pairs; immutable; grows only by stage-4 survivors
STAGE 2   DUAL LADDER             monolingual meaning per word: homonyms ≡, synonyms ~, senses, routes
STAGE 3   CARVE TRAINING (GPU)    expand EN and FR *independently*: conjugations, Zipf phrases, paraphrases, weaves
STAGE 4   CROSS-LANGUAGE FILTER   which expansions survive as sound in the other language; partial credit; feed stage 1
STAGE 5   PARAGRAPHS              compose over the enlarged verified bank; prosody + seam fluency
```

---

## Stage 1 — gold, kept in place

| asset | file |
|---|---|
| unified tier ladder (all vetted sources, ranked, provenance kept) | `tier_ladder.py` → `tier-ladder.tsv` |
| strict-gold corpus (9,803) | `strict-gold-training.jsonl` |
| v7 dictionary (16,208) | `dictionary-v7-integrated.json` |
| dual atoms (loop-certified) | `loop-certified-pairs-v7u(-aug).tsv` |
| proven sound-blocks | `fragments.tsv` |
| phrase bank | `phrase-bank(-balanced).tsv` |

Rule: new verified pairs from stage 4 enter as a **new provenance column** —
nothing is ever overwritten.

## Stage 2 — the dual ladder (meaning only, one language at a time)

Per word, in its own language: homonyms, synonyms to swap to when the word
doesn't fit, sense clusters, routes to other words. No cross-language content.

| bucket | engine — **all of this already exists** |
|---|---|
| homonyms ≡ (same sound, same language) | `en-homophone-classes.tsv` (707), `fr-homophone-classes.tsv` (4,583), `fr-homophone-classes-lexique.tsv` (33,660, morphology folded in) |
| typed slots ≈ / = / ~ kept separate | `ladder.py` |
| sense split (polysemy by embedding) | `ladder.py` §4 |
| routes / hubs / paths to other words | `routes.py`, `hoproute.py` |
| synonym chains | `chain_game.py`, MUSE vectors in `weave.py` |

✅ Done (2026-07-12): `build_ladder_json.py` → `ladder-words.jsonl` (93k words,
gitignored, regenerates in ~30s) — homonyms/synonyms/routes/pairs wired from
the extant TSVs; `senses[]` stays empty (`senses_pending`) until the node-vec
cache is regenerated on a GPU box. A homonym is a free semantic pivot: if *sea*
doesn't fit the line, *see* occupies the same sound cell and reopens meaning.

## Stage 3 — carve training (the GPU stage; EN and FR expanded independently)

Never trains on translation pairs. Task: replicate the gold corpus's
homophone-friendly material across the full dictionary of each language.

**Deterministic first (no GPU):** every gold word × full inflection table —
French from **Lefff**, English from **UniMorph/LemmInflect** — weighted by
**Zipf frequency** (`zipf_glue.py` machinery). Conjugation is a lookup, not a
skill; don't spend training on it.

**Learned second (Qwen LoRA, `research/qwen-finetune/`):** the three moves
lookup can't do, each with an extant prototype:

| move | engine |
|---|---|
| paraphrase-wide (meaning-true rewordings) | `paraphrase_search.py` step 1 (distill into local model; gold-tier paraphrases as SFT targets) |
| Zipf-phrase generation (checked against books/bigrams) | `bigram_lm.py`, PD corpus, `zipf_glue.py` glue table |
| multi-word weaving | `fragments.py`, `fragment_weave.py`, `phrase_weave.py` |

Reward is **monolingual only**: zipf + fluency + semantic-cos. Sound is
deliberately absent — that's stage 4. Output: `expansion-{en,fr}.tsv`.

## Stage 4 — the cross-language filter (sound re-enters)

Every stage-3 variant, EN×FR: does it work as sound in the other language?

| check | engine |
|---|---|
| verdict | `matcher.py` combo (AUC 0.993) |
| strict tier | `strict_judge.py` (geo-ensemble + beats-nearest-rival) |
| behavioral ear | Agent B + Agent C two-comparison judge (`three_agent_v2.py`) |
| spoken stream (liaison/elision/h-aspiré) | `research/qwen-finetune/sandhi_fr.py` |
| **sentence-level prosody** | `prosody.py` (stress-weighted, diverged EN/FR contour) + `rhythm_channel.py` (metrical grid) |

**Where sentence-level prosody/weave institute** (the open question, answered):
fragment/phrase **weave is a stage-3 generator**; **prosody is a stage-4
judge channel** — segmental combo gates first, then prosody re-ranks
survivors at phrase/sentence length (rhythm is exactly what segmental scoring
misses on multi-syllable spans); **sandhi sits between them**, transforming
written French into the spoken stream both judges hear.

Partial credit is mandatory: score all sub-spans; harvest working cores into
`fragments.tsv` / the phrase bank even when the whole variant fails.
**Survivors → stage 1 → next stage-3 round.** That loop is the whole engine.

## Stage 5 — paragraphs

`bicameral_paragraph_v2.py`, `dual_scale_composer.py`,
`recursive_composer.py`, `round_rabbit_bicameral.py`, `bank_composer.py` —
all extant. Blocked on inventory, not code: composing over 813 atoms measures
~0.14 both-side fluency (`DUAL_LADDER.md`). Start stage 5 only after a
stage-4 cycle has grown the bank; the stage-5-specific work is seam fluency
(composer LLM, `docs/sentence-generation-plan.md`) and prosody shaping.

---

## Build order

1. Wire the existing homophone-class TSVs into the ladder JSON (stage 2) — no GPU.
2. Deterministic Lefff/UniMorph/Zipf expansion of every gold unit — no GPU.
3. Stage-4 filter over that free expansion — inflections of proven words are
   the highest-probability new homophones there are (if *rives* works, try
   *rive*, *rivait* first). Expect thousands of new pairs before any training.
4. Stage-3 SFT + monolingual DPO on the enlarged gold; iterate 3→4→1.
5. Metric per cycle: **verified-pair count by tier**. When a cycle stops
   adding STRICT-GOLD, improve stage 3 — don't just run more.
