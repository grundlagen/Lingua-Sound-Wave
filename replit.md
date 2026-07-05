# Workspace

> **⚠️ Methodology lives in the 11–12 June Python schema, not here.**
> This file documents the TypeScript pnpm workspace (the `artifacts/`
> explorer + API server). That stack still builds, but the **authoritative**
> homophone methodology is the offline, deterministic Python pipeline in
> `research/homophone-bench/` (`README.md` → "The 11–12 June schema").
> The `phoneme-chain` / `hybrid-phoneme-audio` judges and the LLM-G2P + audio
> machinery described below are the **earlier** approach; the bench
> superseded them with the `combo` featural+n-gram matcher + **learned**
> equivalence-floored costs, the `phonetic_decoder` trie/beam, and re-mining.
>
> **Deliberately NOT used** (do not reintroduce): fuzzy / edit-distance
> matching (Levenshtein, difflib, rapidfuzz); plotting libraries (matplotlib /
> pyplot) — results are TSV + plain text; **epitran** for G2P (use espeak-ng +
> CMUdict + Lexique).

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## Cross-Lingual Homophone Explorer

Artifact `artifacts/homophone-explorer` (web) backed by `artifacts/api-server`.

### Scoring methods

Six phonetic-similarity judges, selectable per request. Audio and symbolic methods are kept as separate first-class systems (not unified) — they have complementary failure modes. **Default is `hybrid-phoneme-audio`** so users see both judges' opinions side-by-side; the headline number is the geometric mean and the per-component sub-scores are surfaced in the UI as chips (see `artifacts/homophone-explorer/src/components/ComponentScores.tsx`).

Audio-based:
1. **mfcc-dtw** (ready) — classical MFCC + DTW on cosine distance. CPU-only, instant.
2. **wav2vec2-mean-cos** (lazy) — wav2vec2-base hidden states mean-pooled, cosine. Coarse baseline. First call downloads ~95MB.
3. **wav2vec2-dtw** (lazy) — frame-level wav2vec2 embeddings DTW-aligned. ~1–4s per pair.

Symbolic:

4. **phoneme-chain** (ready) — *symbolic IPA matcher.* An LLM (gpt-5.4 via the OpenAI integration) converts each phrase to broad IPA along with up to 3 plausible pronunciation variants (fast-speech, schwa reduction, devoicing, dialect substitutions — the "synonym chains"). IPA strings are tokenized (digraph affricates, tie-bar normalization), then aligned via weighted Needleman–Wunsch. Substitution cost combines featural distance (place/manner/voicing for consonants; height/backness/rounding for vowels) with equivalence-class shortcuts (rhotic family, l-vocalization, TH-fronting, sibilant family, voicing pairs, schwa↔reduced vowels, nasal place mismatches) and **offglide-aware gap costs** — deletion of the 2nd half of common diphthongs (ʊ in /oʊ/, ɪ in /eɪ/, glides j/w) costs 0.12 instead of the standard 0.5, so e.g. English /ʃoʊ/ "show" cheaply aligns to Korean /ɕo/ "쇼". Best alignment across all source-variant × target-variant pairs wins. First call per phrase ≈1–2s LLM; cached afterwards (cap 5000, transient failures not cached). Implementation: `artifacts/api-server/src/lib/phoneme.ts`.

Hybrids (combined symbolic + acoustic confirmation, return `componentScores[]` so the UI can show both judges):

5. **hybrid-phoneme-audio** (default, lazy) — DTW + Phoneme combined. Geometric mean of phoneme-chain × wav2vec2-dtw, both individual scores surfaced. Picked as default for transparency: phoneme-chain catches wav2vec2-dtw's TTS-voice-floor false positives; wav2vec2-dtw provides independent acoustic confirmation. First call downloads ~95MB.
6. **hybrid-phoneme-mfcc** (ready) — MFCC-DTW + Phoneme combined. Same idea, no neural download.

`ScoreResult.components?: ComponentScore[]` carries `{id, label, similarity, distance}` per sub-judge. `compare`, `translate`, and `discover` route handlers spread this into their responses; the openapi `ComponentScore` schema is the wire contract.

### Benchmark (8 pairs: 5 unrelated translations as negatives, 3 known cross-lingual homophones as positives)

| Method | mean(neg) | mean(pos) | margin | clean? |
|---|---|---|---|---|
| **phoneme-chain** | **28.8%** | **96.8%** | **+49.9pt** | **YES** |
| hybrid-phoneme-audio | 46.8% | 95.9% | +31.4pt | YES |
| hybrid-phoneme-mfcc | 25.9% | 79.1% | +14.4pt | YES |
| mfcc-dtw | 24.5% | 68.5% | +6.4pt | YES (this run) |
| wav2vec2-dtw | 78.5% | 92.7% | -3.2pt | NO |
| wav2vec2-mean-cos | 84.7% | 65.6% | -49.9pt | NO |

Honest finding: hybrids underperform phoneme-chain alone on this benchmark — neither acoustic component is reliable enough to provide confirmation signal that outweighs the noise it introduces on negatives. Kept available for users who want them but `phoneme-chain` is the recommended judge for cross-lingual sound similarity.

### ScoreInput

`ScoreInput` (in `lib/scoring.ts`) carries optional `text`, `language`, `languageName` alongside the audio fields. Phoneme-based methods require these; audio-only methods ignore them. Call sites in `routes/homophones.ts` (compare, translate, discover) populate them via spread.

### Reservoir (EN↔FR persistent corpus)

A growing tiered corpus of EN↔FR homophone pairs, target size ~2,500. Tables: `homophone_reservoir` (one row per scored pair, with `tier` S/A/B and full `componentScores`) and `mining_jobs` (background mining state, idempotent — single active job at a time).

- **Seeds**: `artifacts/api-server/src/lib/seed-corpus.ts` ships 240 hand-curated EN↔FR seed pairs (proverbs, food, geography, common nouns, idioms). Loaded once into `homophone_reservoir` on first mining run.
- **Mining**: `artifacts/api-server/src/lib/reservoir-mining.ts` runs in-process, scored via `hybrid-phoneme-audio`. Each job pulls a batch of unscored seeds + LLM-suggested expansions, scores both directions (EN→FR and FR→EN), grades into tier S (≥0.92), A (≥0.85), B (≥0.75), drops the rest. Idempotent on `(text_en, text_fr)` — re-runs are cheap.
- **Tier grader**: `artifacts/api-server/src/lib/tier-grader.ts` — pure scoring→tier mapping, used by both the miner and ad-hoc requests.
- **Routes**: `artifacts/api-server/src/routes/reservoir.ts` exposes `GET /api/reservoir` (filter by tier, paginated), `POST /api/reservoir/mining/start`, `GET /api/reservoir/mining/status` (frontend polls every 2s while a job is active).
- **UI**: `artifacts/homophone-explorer/src/pages/Reservoir.tsx` — mining controls, live status, tier-filtered browse. Polling key uses `getGetReservoirMiningStatusQueryKey()` so React Query refetches across remounts.

### Flit Lab

Cross-lingual sound-alike paraphraser. Given input text in EN or FR, produces meaning-preserving renderings in the *other* language that *sound like* the input. Pipeline (`artifacts/api-server/src/lib/flit.ts`):

1. **Input rejog**: LLM generates N semantic-preserving paraphrases of the input (each with a literal gloss).
2. **Target rendering**: For each input paraphrase, LLM proposes M sound-alike candidates in the target language, also with glosses.
3. **Cross-product score**: Every (paraphrase × candidate) pair scored by `hybrid-phoneme-audio`. Top-K by similarity.
4. **Semantic verify**: A second LLM pass checks that each top candidate's gloss is consistent with the input's meaning, setting `semanticOK` + a short note. Sound-alikes that drift in meaning are kept but flagged (not dropped) so users can see why.

Routes: `POST /api/flit/run` ({text, language, inputParaphrases, targetRenderings, topK}). UI: `artifacts/homophone-explorer/src/pages/FlitLab.tsx` with presets and sliders.
