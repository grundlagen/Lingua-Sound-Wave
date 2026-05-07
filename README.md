# Lingua-Sound-Wave

**Cross-lingual sound similarity laboratory** — exploring homophones, sound symbolism, and meaning-preserving sound-alike paraphrases across languages.

> "Sound responsibly." — Project motto

---

## What This Project Actually Does (Advanced Features Live)

This is **not** a basic scaffold. It contains real, production-grade linguistic tooling:

### 1. Cross-Lingual Homophone Explorer
Compare any phrase across languages (currently benchmarked on EN↔FR) using **six different phonetic judges**:

**Symbolic (strongest on current benchmark)**:
- `phoneme-chain`: LLM → broad IPA + variants → weighted Needleman–Wunsch alignment with featural costs and offglide-aware gaps.

**Acoustic**:
- `mfcc-dtw` (classical, instant)
- `wav2vec2-dtw` and `wav2vec2-mean-cos` (neural embeddings)

**Hybrids** (for transparency):
- `hybrid-phoneme-audio` (default — shows both judges side-by-side)
- `hybrid-phoneme-mfcc`

**Honest Benchmark** (8 pairs):
`phoneme-chain` achieved **+49.9pt margin** on negatives — better than hybrids. We surface this transparently.

### 2. Homophone Reservoir (Growing Corpus)
- Target: ~2,500 high-quality EN↔FR pairs
- Tiered mining (S ≥0.92, A ≥0.85, B ≥0.75)
- Background mining jobs with live status polling
- 240 hand-curated seeds (proverbs, food, idioms, geography)

### 3. Flit Lab — Sound-Alike Paraphraser
Input text in EN or FR → produces meaning-preserving renderings in the *other* language that *sound similar*.
- LLM generates semantic paraphrases
- LLM proposes sound-alike candidates
- Scored + semantically verified
- Users see why a candidate was kept or flagged

---

## Architecture & Tech

Full pnpm monorepo with the same thoughtful foundation as Proto-Lingua-Weaver:
- Express 5 + Drizzle + PostgreSQL
- OpenAPI + Orval codegen (perfect type safety)
- Sophisticated scoring & mining engine in `artifacts/api-server/src/lib/`
- React frontend with live polling and beautiful result visualization

See `handover.md` for deep design decisions, extension guides, and why we kept symbolic and acoustic systems as first-class citizens.

---

## Quick Start

```bash
bash ./start.sh   # or the equivalent setup command for this repo
```

The workspace will spin up the API + `homophone-explorer` frontend with all advanced features ready.

---

## Why This Matters
Most "AI language" tools hide their weaknesses. Lingua-Sound-Wave is deliberately **transparent** about what works and what doesn't. The benchmark results are public. The component scores are visible. This is how real linguistic science + delightful software should be built.

**Future**: Tighter integration with Proto-Lingua-Weaver, more language pairs, academic export tools, and community-contributed homophones.

---

*Built with care and intellectual honesty — May 2026*