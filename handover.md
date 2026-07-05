# Lingua-Sound-Wave — Project Handover & Knowledge Base

**Version**: 0.3.0 (Meaning Map — sound+sense convergence layer — June 2026)
**Maintainer**: grundlagen + Claude
**Last Updated**: 2026-06-25

---

> **⚠️ Authoritative methodology: the 11–12 June Python schema.**
> Sections 1–5 below describe the **earlier** TypeScript subsystem
> (`artifacts/api-server`, `artifacts/homophone-explorer`) — LLM G2P,
> wav2vec2/MFCC audio judges, PostgreSQL. That subsystem still exists, but
> it is **not** the current approach. The canonical engine is the **offline,
> deterministic Python pipeline** in `research/homophone-bench/` (see its
> `README.md` → "The 11–12 June schema"): espeak-ng + CMUdict + Lexique G2P,
> the `combo` featural+n-gram matcher with **learned** equivalence-floored
> costs, the `phonetic_decoder` Lexique-trie beam search, `weave`/`explode_web`,
> and **re-mining** as the only growth step. No paid APIs, no audio channel
> on the default path.
>
> **Deliberately NOT used** (do not reintroduce): fuzzy / edit-distance
> string matching (Levenshtein, difflib, rapidfuzz) — matching is featural +
> learned costs, growth is re-mining; **plotting libraries** (matplotlib /
> pyplot) — results are TSV + plain text; **epitran** for G2P — use espeak-ng
> + CMUdict + Lexique.

---

## 1. Project Vision

Lingua-Sound-Wave is a **cutting-edge cross-lingual sound similarity laboratory**. It explores the deep, often mysterious relationship between how words *sound* across languages — homophones, sound symbolism, and meaning-preserving sound-alike paraphrases.

As of v0.3 it also explores how sound and *meaning* converge: the **Meaning Map** lays the
homophone bank (EN↔FR by sound) over a semantic bank (EN↔FR by meaning), stacks synonyms on
top, and whittles the corpus down to the pairs where sound and sense meet — the seed of a
"map of meaning" across languages. See **MEANING_MAP.md** and **NEXT_ROUTINE.md**.

### Core Features (Already Implemented)
- **Cross-Lingual Homophone Explorer**: Compare phrases across languages (EN↔FR benchmarked) using multiple phonetic judges.
- **Six Scoring Methods** (with honest benchmarks):
  - `phoneme-chain` (symbolic IPA + weighted Needleman–Wunsch — currently strongest)
  - `mfcc-dtw`, `wav2vec2-dtw`, `wav2vec2-mean-cos` (acoustic)
  - Hybrids that combine symbolic + acoustic for transparency
- **Homophone Reservoir**: Growing curated corpus (~2,500 target) with tiered mining (S/A/B) using LLM + scoring pipeline. Every row stores `enGloss`/`frGloss` — the latent meaning layer the Meaning Map builds on.
- **Flit Lab**: Sound-alike paraphraser — input in one language → meaning-preserving renderings in another that *sound similar* (with semantic verification).
- **Meaning Map** (v0.3): fuses each reservoir pair's *sound* (`similarity`) and *sense*
  (`semanticSimilarity(enGloss, frGloss)`) into a **resonance** score and a sound×sense
  quadrant; builds a graph of **meaning islands** (synonym/translation clusters) joined by
  phonetic **bridges** and **resonances**.

**Philosophy**: Be honest about what works. The benchmark showed `phoneme-chain` outperforming hybrids on negatives — we surface this transparently instead of hiding it. Users see component scores and can choose judges.

---

## 2. Architecture Highlights

- Full pnpm monorepo with OpenAPI + Orval codegen (same thoughtful foundation as Proto-Lingua-Weaver).
- Sophisticated scoring engine in `artifacts/api-server/src/lib/` (phoneme.ts, scoring.ts, reservoir-mining.ts, flit.ts, tier-grader.ts).
- **Meaning Map** layer in `artifacts/api-server/src/lib/` (semantic.ts, meaning-graph.ts) + `routes/meaning.ts`. The graph engine is pure (no I/O) so it is unit-testable; the semantic judge mirrors `tier-grader.ts` (gpt-5.4, strict JSON, cached).
- LLM integration (gpt-5.4 via OpenAI) for IPA conversion, semantic verification, candidate generation, and gloss-meaning similarity — cached intelligently.
- React frontend with live mining status polling, tier filtering, and Flit Lab UI.
- Persistent PostgreSQL storage for reservoir and mining jobs (idempotent).

**Key Decision**: Symbolic (phoneme) and acoustic (wav2vec2/MFCC) systems are kept as *separate first-class citizens* with hybrids as optional views. The Meaning Map follows the same rule: *sound* and *sense* stay separate first-class scores; **resonance** (their geometric mean) is a view over them, never a replacement.

---

## 3. How to Extend

- Add new scoring judge → implement in `lib/scoring.ts` + expose in OpenAPI + UI chips.
- Expand reservoir → improve `seed-corpus.ts` or mining logic in `reservoir-mining.ts`.
- New language pair → extend `ScoreInput` and LLM prompts; update benchmark.
- Improve Flit Lab → tune LLM prompts or add more semantic verification passes.
- **Extend the Meaning Map** → see NEXT_ROUTINE.md. Highest-leverage next step: replace
  the v1 normalized-gloss-equality SENSE edges with embedding/LLM synonymy so meaning
  islands form on real sense; then persist `semantic`/`resonance` on the reservoir and
  build the frontend Meaning Map page.

Full details in the code comments and the advanced `replit.md`.

---

## 4. Testing & Quality
- Type safety enforced.
- Honest benchmarks included in docs (we show what works and what doesn't).
- Smoke tests via start.sh equivalent.
- Meaning Map endpoints are bounded by design: ≤ `limit` LLM semantic calls per request
  (one per pair, cached, 4 concurrent) — no O(n²) gloss comparisons in v1.

**Golden Rule**: If the numbers lie or the UI hides complexity, we fix it. Transparency > magic.

---

## 5. Roadmap
1. Smarter SENSE edges (embedding/LLM synonymy) so meaning islands are real, not spelling-based.
2. Frontend Meaning Map page (sound×sense scatter + island/bridge force graph) + OpenAPI hooks.
3. Persist `semantic`/`resonance`/`quadrant` during mining so the map doesn't re-call the LLM.
4. Multi-hop meaning chains (alternating PHONE/SENSE pathfinding between distant meanings).
5. More language pairs → islands and bridges that span three+ languages.
6. Real-time audio playback; user-contributed homophones with moderation.
7. Integration with Proto-Lingua-Weaver (shared proto-form data); academic export.

---

**This project is already doing real, novel linguistic work.** Treat the existing code with respect — it represents significant thoughtful engineering.

**Sound responsibly.** 🎵🗣️
