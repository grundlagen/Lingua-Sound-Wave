# Lingua-Sound-Wave — Project Handover & Knowledge Base

**Version**: 0.2.0 (Advanced Linguistic Features Live — May 2026)
**Maintainer**: grundlagen + Grok (xAI)
**Last Updated**: 2026-05-07

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

### Core Features (Already Implemented)
- **Cross-Lingual Homophone Explorer**: Compare phrases across languages (EN↔FR benchmarked) using multiple phonetic judges.
- **Six Scoring Methods** (with honest benchmarks):
  - `phoneme-chain` (symbolic IPA + weighted Needleman–Wunsch — currently strongest)
  - `mfcc-dtw`, `wav2vec2-dtw`, `wav2vec2-mean-cos` (acoustic)
  - Hybrids that combine symbolic + acoustic for transparency
- **Homophone Reservoir**: Growing curated corpus (~2,500 target) with tiered mining (S/A/B) using LLM + scoring pipeline.
- **Flit Lab**: Sound-alike paraphraser — input in one language → meaning-preserving renderings in another that *sound similar* (with semantic verification).

**Philosophy**: Be honest about what works. The benchmark showed `phoneme-chain` outperforming hybrids on negatives — we surface this transparently instead of hiding it. Users see component scores and can choose judges.

---

## 2. Architecture Highlights

- Full pnpm monorepo with OpenAPI + Orval codegen (same thoughtful foundation as Proto-Lingua-Weaver).
- Sophisticated scoring engine in `artifacts/api-server/src/lib/` (phoneme.ts, scoring.ts, reservoir-mining.ts, flit.ts, tier-grader.ts).
- LLM integration (gpt-5.4 via OpenAI) for IPA conversion, semantic verification, and candidate generation — cached intelligently.
- React frontend with live mining status polling, tier filtering, and Flit Lab UI.
- Persistent PostgreSQL storage for reservoir and mining jobs (idempotent).

**Key Decision**: Symbolic (phoneme) and acoustic (wav2vec2/MFCC) systems are kept as *separate first-class citizens* with hybrids as optional views. This avoids the common trap of forcing unification before the data justifies it.

---

## 3. How to Extend

- Add new scoring judge → implement in `lib/scoring.ts` + expose in OpenAPI + UI chips.
- Expand reservoir → improve `seed-corpus.ts` or mining logic in `reservoir-mining.ts`.
- New language pair → extend `ScoreInput` and LLM prompts; update benchmark.
- Improve Flit Lab → tune LLM prompts or add more semantic verification passes.

Full details in the code comments and the advanced `replit.md`.

---

## 4. Testing & Quality
- Type safety enforced.
- Honest benchmarks included in docs (we show what works and what doesn't).
- Smoke tests via start.sh equivalent.

**Golden Rule**: If the numbers lie or the UI hides complexity, we fix it. Transparency > magic.

---

## 5. Roadmap
1. More language pairs + better IPA coverage.
2. Real-time audio playback of scored pairs.
3. User-contributed homophones with moderation.
4. Integration with Proto-Lingua-Weaver (shared proto-form data).
5. Academic export + citation tools.

---

**This project is already doing real, novel linguistic work.** Treat the existing code with respect — it represents significant thoughtful engineering.

**Sound responsibly.** 🎵🗣️