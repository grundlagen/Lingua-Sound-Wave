# The Meaning Map — Routine & Next Steps

**Lineage**: continues `handover.md` and `replit.md` (the homophonic-hunting routine).
**This routine**: laid the first stone of the *semantic ⨉ homophonic* layer.
**Branch**: `claude/upbeat-thompson-zrx3r8`
**Date**: 2026-06-22

---

## 0. The grand goal (restated, so the next walker knows the bearing)

> Match **semantic** meaning-chains (FR→EN translation pair banks, synonyms) and
> **homophonic** sound-chains (EN→FR sound-alike pair banks), lay them on the
> same map, and follow the connecting chains to **whittle down to the phrases
> that are the same in both worlds at once** — phrases that *sound alike* **and**
> *mean alike* across the language gap. Push toward the limit where every
> language maps cleanly onto every other: a map of meaning you can walk.

The reservoir already hunts sound-alikes. It cannot, on its own, tell you which
sound-alikes also *mean the same thing* — and those rare coincidences (a French
word that both sounds like and means an English one) are the gems. You don't
mine gems by brute force; you find them where two chains cross.

---

## 1. Where the project stood (the previous routine)

The homophonic-hunting machinery (all live, all honest — see `replit.md` §
*Cross-Lingual Homophone Explorer*):

- **phoneme-chain** scorer (`lib/phoneme.ts`) — symbolic IPA + featural
  Needleman–Wunsch + pronunciation-variant chaining. The strongest judge
  (+49.9pt margin). This is the truth source for *sound*.
- **Reservoir** (`lib/reservoir-mining.ts`, `seed-corpus.ts`, `tier-grader.ts`)
  — the **homophone pair bank**: tiered EN↔FR sound-alike rows in Postgres.
- **Flit Lab** (`lib/flit.ts`) — sound-alike paraphraser with a *semantic
  verification* pass. Note: Flit already proves the two-axis idea works at the
  level of a single query (it scores sound, then checks sense). The Meaning Map
  generalizes that one-shot check into a persistent, composable graph.

What was missing: **a semantic pair bank** and **a structure that composes the
two banks into chains.** That is this routine's contribution.

---

## 2. What this routine built

`artifacts/api-server/src/lib/meaning-map.ts` — a pure, dependency-free graph:

- **Nodes** = phrases in a language (`${lang}::${text}`).
- **Two edge kinds** on the same node set:
  - `homophone` — cross-lingual SOUND similarity (weight = phoneme-chain score).
    This is the reservoir, re-expressed as edges.
  - `semantic` — same MEANING: translations across languages **and** synonyms
    within one. This is the new layer.
- **`bestPaths(origin, kind)`** — Dijkstra in −log space; maximizes
  `Π(weights) × hopDecay^(hops−1)`. Walks pure-sound chains or pure-sense chains.
- **`findConvergences()`** — the heart. A **convergence** is a target reachable
  from an origin by *both* a sound-only path and a sense-only path. Two
  independent witnesses that the target sounds like *and* means the origin.
  Ranked by geometric mean of the two path strengths; **gated by the weaker
  world** (a phrase that sounds perfect but means nothing is not a gem).
- **`meaningClusters()`** — union-find over strong semantic edges: the
  "collapse synonyms/translations into one meaning" operation. This is the
  literal "whittle down the same meanings over time" structure.

**Monotonic sharpening** ("whittle down over time"): edges only ever *add* or
*strengthen* (`addEdge` keeps the max weight). New pair-bank rows can create or
reinforce convergences, never erase them. The map gets sharper as both banks
grow — exactly the desired dynamic.

`meaning-map.demo.ts` — runs with **zero infrastructure**:

```sh
cd artifacts/api-server/src/lib
node --experimental-strip-types meaning-map.demo.ts
```

Verified output: it finds `new⇄neuf` and `fresh⇄frais` (gems — sound + sense
meet) by *composing chains*, and correctly **rejects** `more⇄mort` (sounds
alike, but *mort* = "dead" ≠ "more"). That rejection is the whittling working.

---

## 3. The algorithm, in one breath

```
homophone bank  ─┐
                 ├─→  one graph, two edge colours  ─→  walk sound-only chains
synonym + trans ─┘                                     walk sense-only chains
                                                       where they LAND ON THE
                                                       SAME NODE  →  a gem
```

A gem found via a 1-hop sound path and a 1-hop sense path is a classic
homophonic-translation. A gem found via *multi-hop* chains (sound: EN→FR→FR;
sense: EN→EN-synonym→FR) is something the reservoir could never surface
directly — it only exists in the composition. That is the whole point.

---

## 4. The next routine — concrete steps toward the bearing

Ordered by leverage. Each is a clean, self-contained piece.

### 4.1 Build a real **semantic pair bank** (highest leverage)
The map is only as good as its sense edges. Stand up a source of
translation + synonym edges, mirroring how `seed-corpus.ts` + mining feeds the
homophone bank:
- Seed translations from the glosses **already in the reservoir** (`enGloss`,
  `frGloss`) — every reservoir row already carries two meanings; wire them in as
  semantic anchors for free.
- Add an LLM "translate + 3 synonyms" pass (reuse the `openai` integration and
  the `gradePair` caching pattern) to expand each node.
- Optional, stronger: multilingual sentence-embeddings for `semantic` weights
  instead of LLM confidence (cosine in [0,1]) — more calibratable, cheaper at
  scale. Flit's semantic-verify prompt is a ready template.

### 4.2 Persist the graph (Drizzle, alongside `homophone_reservoir`)
Two tables in `lib/db/src/schema/` (follow `homophone-reservoir.ts` exactly):
`meaning_nodes (id, lang, text, gloss)` and
`meaning_edges (from, to, kind, weight, source)` with a unique index on
`(from, to, kind)`. Load into `MeaningGraph` on demand; the class is already
DB-agnostic, so ingestion is a thin adapter.

### 4.3 Mine convergences inside the reservoir loop
In `reservoir-mining.ts`, after a homophone pair is graded, also drop its
glosses in as semantic nodes and run `convergencesFrom()` on the new node.
Persist gems to a `meaning_gems` table with a **witness count** (how many
independent chains found it) — reinforcement over time, the literal "whittle
down" telemetry.

### 4.4 Surface it (API + UI)
- Route: `POST /api/meaning/convergences` ({text, language, opts}) →
  `convergencesFrom`. Add to the OpenAPI spec so Orval regenerates the hooks
  (`pnpm --filter @workspace/api-spec run codegen`).
- UI: a "Meaning Map" page beside Reservoir / Flit Lab — enter a phrase, see the
  gems with **both** chains drawn (sound chain + sense chain), scores, and the
  gate. The two-coloured path drawing IS the product.

### 4.5 Go N-lingual — the actual "all languages to one another"
The graph is already language-generic (`Lang` is just a string; nothing is
EN/FR-specific in `meaning-map.ts`). To pivot through it:
- A meaning cluster (4.x `meaningClusters`) is a language-neutral *interlingua
  node*. Translation edges from many languages into the same cluster turn the
  cluster into a hub. A gem between language X and language Z is then a
  convergence routed sound-wise directly and sense-wise *through the shared
  cluster*. This is the path to "perfection in mapping all languages to one
  another" — every pair bank you add widens the reachable set combinatorially.
- Bridge to **Proto-Lingua-Weaver**: its proto-forms and sound laws are a
  principled *generator* of homophone edges (regular sound correspondences =
  high-confidence sound chains). Roadmap item #4 in `handover.md` (shared
  proto-form data) becomes concrete here.

### 4.6 Honest benchmark (the golden rule)
Build a small labelled set of known cross-lingual homophone-translations
(positives) and sound-only false friends (negatives, like `more/mort`). Report
gem precision/recall at varying `gate` thresholds. Surface it in the docs the
way the 8-pair phonetic benchmark is surfaced. If the gate lets junk through,
say so and fix the gate — don't hide it.

---

## 5. Footsteps & philosophy (for the next mad walker)

- **Keep the two worlds first-class and separate**, then let them *meet* — never
  pre-fuse sound and sense into one mushy score. The whole insight is that they
  are independent witnesses; the gem is where they *coincide*. (Same principle
  the previous routine used keeping symbolic and acoustic judges separate.)
- **The min-gate is sacred.** A high geometric-mean score with a low gate is a
  near-miss, not a gem. Rank by score, *trust* by gate.
- **Purity pays.** `meaning-map.ts` has no imports — it runs under
  `node --experimental-strip-types` with no install. Keep the core walkable
  offline; push DB/LLM/HTTP to the edges. It made this routine verifiable in one
  command; keep that gift for the next one.
- **The map only sharpens.** Every edge you add is a permanent gift to the map.
  Add banks, add languages, add synonyms — and watch the gems multiply where the
  chains cross. That is the whole game.

> Hunt the gems where sound and sense cross. The map is waiting to be walked. 🎵🗺️
