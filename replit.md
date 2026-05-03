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

Six phonetic-similarity judges, selectable per request. Audio and symbolic methods are kept as separate first-class systems (not unified) — they have complementary failure modes.

Audio-based:
1. **mfcc-dtw** (default, ready) — classical MFCC + DTW on cosine distance. CPU-only, instant.
2. **wav2vec2-mean-cos** (lazy) — wav2vec2-base hidden states mean-pooled, cosine. Coarse baseline. First call downloads ~95MB.
3. **wav2vec2-dtw** (lazy) — frame-level wav2vec2 embeddings DTW-aligned. ~1–4s per pair.

Symbolic:

4. **phoneme-chain** (ready) — *symbolic IPA matcher.* An LLM (gpt-5.4 via the OpenAI integration) converts each phrase to broad IPA along with up to 3 plausible pronunciation variants (fast-speech, schwa reduction, devoicing, dialect substitutions — the "synonym chains"). IPA strings are tokenized (digraph affricates, tie-bar normalization), then aligned via weighted Needleman–Wunsch. Substitution cost combines featural distance (place/manner/voicing for consonants; height/backness/rounding for vowels) with equivalence-class shortcuts (rhotic family, l-vocalization, TH-fronting, sibilant family, voicing pairs, schwa↔reduced vowels, nasal place mismatches) and **offglide-aware gap costs** — deletion of the 2nd half of common diphthongs (ʊ in /oʊ/, ɪ in /eɪ/, glides j/w) costs 0.12 instead of the standard 0.5, so e.g. English /ʃoʊ/ "show" cheaply aligns to Korean /ɕo/ "쇼". Best alignment across all source-variant × target-variant pairs wins. First call per phrase ≈1–2s LLM; cached afterwards (cap 5000, transient failures not cached). Implementation: `artifacts/api-server/src/lib/phoneme.ts`.

Hybrids (combined symbolic + acoustic confirmation):

5. **hybrid-phoneme-audio** (lazy) — geometric mean of phoneme-chain × wav2vec2-dtw. Idea: each catches the other's idiosyncratic false positives. In practice the wav2vec2-dtw voice-floor on same-voice TTS inflates negatives more than it confirms positives — see benchmark below.
6. **hybrid-phoneme-mfcc** (ready) — geometric mean of phoneme-chain × mfcc-dtw. Cheaper than the wav2vec2 hybrid; mfcc-dtw has a wider relative spread on TTS but its absolute scores are noisy on positives, so the geometric mean still drags strong positives down.

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
