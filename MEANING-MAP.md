# The Meaning Map — Routine Log & Handoff

> *Hunting for a map of meaning that allows great feats: the grand goal of
> homophonic **and** semantic matching, EN ⇄ FR, whittled toward perfection
> over time.*

This file is both a record of what the current routine built and the marching
orders for the next one. Each routine should read it top-to-bottom, do the work
in **§5 Next Routine**, then update **§3 Footsteps** and rewrite **§5** for its
successor. We are walking in our own footsteps on purpose.

---

## 1. The vision (why this exists)

Two pair-banks, fused into one graph:

- a **semantic** pair-bank — FR ⇄ EN by *meaning* (translations), with a
  **synonym layer mapped on top** so meaning can travel further than the
  dictionary;
- a **homophonic** pair-bank — EN ⇄ FR by *sound*.

Meaning edges preserve sense and destroy sound. Sound edges preserve sound and
destroy sense. The prize is where the two channels, walked **independently**,
land on the **same node**: a French phrase that both *means* and *sounds like* an
English one. Those are **convergences** — the "great feats." We then **whittle**
the atlas over rising thresholds until only the gold core of near-perfect
homophonic translations remains. Push that idea to the limit across all language
pairs and you approach *perfection in all language to one another* — a small
goal, I know.

This continues the lineage already in this repo: the homophone **Reservoir**
(the EN↔FR sound pair-bank), **Flit Lab** (sound-alike paraphrasing with
semantic verification), and the `phoneme-chain` scorer. The Meaning Map is the
layer *above* them: it asks not "do these two sound alike?" but "which pairs are
reachable by **both** sound and sense, and how do the chains connect?"

---

## 2. What this routine built

A self-contained, offline, deterministic engine. **No LLM, no DB, no network.**

| File | Role |
|---|---|
| `artifacts/api-server/src/lib/meaning-bank.ts` | Curated seed data: vocabulary tagged with language-neutral `concept`s (the semantic pair-bank), a cross-concept **synonym/sense layer**, and a hand-verified **homophonic floor**. |
| `artifacts/api-server/src/lib/meaning-map.ts` | The engine: a dual-layer graph, a best-first single-layer **closure** walker, **convergence** discovery (sound-path ∧ meaning-path), **siren** detection (sound without sense), auto-**mining** of homophonic edges, and **whittling**. Plus a dependency-free default phonetic scorer. |
| `artifacts/api-server/src/lib/meaning-map-atlas.ts` | Builds the map from the bank, runs the hunt, prints the atlas. Runnable directly. |

### Run it

```sh
# from repo root, no install needed:
node --experimental-strip-types \
  artifacts/api-server/src/lib/meaning-map-atlas.ts

# or, from artifacts/api-server:
pnpm run meaning-map
```

### What it prints today

- **Convergences** — e.g. `en:rich ≈⇔ fr:riche`, `en:table ≈⇔ fr:table`,
  `en:blue ≈⇔ fr:bleu`, `en:rose ≈⇔ fr:rose` (cognate: both paths length 1).
- A **transitive** convergence — `en:raisin ≈⇔ fr:raisin`, where the *sound*
  path is direct but the *meaning* path only connects through a synonym hop:
  `en:raisin → en:grape → fr:raisin` (a raisin is a dried grape). This is the
  connecting-chain feat the map exists to find.
- **Sirens** — `en:pain ≈ fr:pain` ("bread"), `en:main ≈ fr:main` ("hand"),
  `en:sea ≈ fr:si` ("if"): strong sound, no meaning path. Surfaced, never
  silently dropped (same honesty principle as Flit Lab).
- **Whittling** — the atlas shrinking over θ = 0.40 … 0.92 to its gold core.

---

## 3. Footsteps (append-only log)

- **2026-06-17 — routine "map of meaning v0".** Built the dual-layer graph,
  closure walker, convergence/siren/whittle pipeline, default phonetic scorer,
  curated EN↔FR bank (~49 nodes). Established the three output classes:
  cognate convergence, transitive convergence, siren. Proved the transitive
  chain mechanism with `raisin → grape → raisin`. Everything offline & runnable.

*(Next routine: add your entry here.)*

---

## 4. Design decisions worth keeping

1. **Two independent paths, not one mixed chain.** A convergence requires a
   *meaning-only* path and a *sound-only* path to the same node. The
   independence is the evidence. Don't "optimise" this into a single blended
   walk — you'd lose the very thing that distinguishes a real homophonic
   translation from a coincidence in one channel.
2. **`concept` is authoring sugar, not a cheat.** The engine decides
   convergences from *edges*, never by reading `concept`. Concepts only generate
   the seed semantic edges and label ground truth. Keep that wall.
3. **Geometric mean for `combined`.** Mirrors the repo's existing
   `hybrid-phoneme-audio` scoring; a weak channel can't be hidden by a strong
   one.
4. **Sirens are first-class output.** False friends are the most useful thing
   the map finds for a translator. Flag, don't drop.
5. **Offline by default, LLM-upgradable.** `mineHomophonic(scorer, threshold)`
   takes any `(a,b)→[0,1]` scorer. The default is a crude orthography→sound
   heuristic so the demo runs anywhere; production should inject the real
   `phonemeChainScore` from `phoneme.ts`.

---

## 5. Next Routine — pick up here

Do these roughly in order. Each is shippable on its own.

1. **Swap in the real ears.** Wire `mineHomophonic` to the LLM
   `phonemeChainScore` (`artifacts/api-server/src/lib/phoneme.ts`) behind an
   injectable scorer interface, cached. Keep the heuristic as the offline
   fallback. The auto-mined edges should then rival the curated floor — measure
   how many new convergences appear.
2. **Feed the map from the Reservoir.** The `homophone_reservoir` table already
   *is* a mined EN↔FR sound pair-bank. Load its rows as homophonic edges
   (weight = stored similarity, tier S/A/B → confidence). The Meaning Map becomes
   a live view over real mined data instead of a 49-word toy.
3. **A real semantic pair-bank.** Replace hand `concept`s with an actual
   bilingual lexicon + synonym source (e.g. an embedding model's cosine, or a
   wordnet-style resource). Semantic edge weight = semantic similarity, so the
   meaning channel gets graded the way the sound channel already is.
4. **Expose it.** Add `GET /api/meaning-map/convergences` and a
   `MeaningMap.tsx` page in `artifacts/homophone-explorer` — render the graph,
   let users click a node and watch the two chains light up to a convergence.
   Show sirens in red. (Follow the existing route + Orval codegen pattern.)
5. **Whittling that learns.** Persist convergence scores across runs; let
   thresholds rise as evidence accumulates (more confirming chains ⇒ higher
   confidence), so the gold core genuinely sharpens "over time" rather than per
   invocation. This is the long arc toward the grand goal.
6. **Beyond EN↔FR.** The engine is language-agnostic; only the bank and scorer
   are EN/FR-bound. Add a third language and look for *triangle* convergences
   (A sounds like B sounds like C, all meaning the same) — the first real step
   toward "all language to one another."

### Guardrails for the next routine
- Keep the atlas runnable with zero install (`node --experimental-strip-types`).
- Keep `pnpm typecheck` green; the lib files must stay valid in the package
  build, not just under type-stripping.
- Don't break the two-independent-paths invariant (§4.1).
- Commit to branch `claude/upbeat-thompson-0jj6cd`. Update §3 and rewrite §5.

> The map is small today. The method is not. Walk on. 🗺️🔊
