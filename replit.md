# Workspace

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

Four phonetic-similarity judges, selectable per request:

1. **mfcc-dtw** (default, ready) — classical MFCC + dynamic time warping on cosine distance. CPU-only, instant.
2. **wav2vec2-mean-cos** (lazy) — wav2vec2-base hidden states mean-pooled, cosine similarity. Coarse baseline. First call downloads ~95MB.
3. **wav2vec2-dtw** (lazy) — frame-level wav2vec2 embeddings DTW-aligned. Most rigorous neural method. ~1–4s per pair.
4. **phoneme-chain** (ready) — *symbolic IPA matcher, skips audio entirely.* An LLM (gpt-5.4 via the OpenAI integration) converts each phrase to broad IPA along with up to 3 plausible pronunciation variants (fast-speech, schwa reduction, devoicing, dialect substitutions — the "synonym chains"). IPA strings are tokenized (with digraph affricate support and tie-bar normalization), then aligned via weighted Needleman–Wunsch. Substitution cost combines featural distance (place/manner/voicing for consonants; height/backness/rounding for vowels) with equivalence-class shortcuts (rhotic family, l-vocalization, TH-fronting, sibilant family, voicing pairs, schwa↔reduced vowels, nasal place mismatches). Best alignment across all source-variant × target-variant pairs wins. First call per phrase ≈1–2s LLM; cached afterwards (cap 5000, transient failures not cached). Implementation: `artifacts/api-server/src/lib/phoneme.ts`.

On the 8-pair benchmark (5 unrelated translation pairs as negatives, 3 known cross-lingual homophones as positives), `phoneme-chain` is the only method achieving clean separation: mean(neg)=25.4%, mean(pos)=80.9%, spread +55.4pts, min(pos)=67% > max(neg)=38%.

### ScoreInput

`ScoreInput` (in `lib/scoring.ts`) carries optional `text`, `language`, `languageName` alongside the audio fields. Phoneme-based methods require these; audio-only methods ignore them. Call sites in `routes/homophones.ts` (compare, translate, discover) populate them via spread.
