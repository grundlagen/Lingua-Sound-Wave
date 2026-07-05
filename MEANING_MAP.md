# The Meaning Map — fusing sound and sense across languages

**Status**: v1 backend engine live (this routine). Frontend + multilingual expansion are the next routine.
**Lives in**: `artifacts/api-server/src/lib/{semantic,meaning-graph}.ts`, `artifacts/api-server/src/routes/meaning.ts`.

---

## 1. Why this exists

Lingua-Sound-Wave already hunts **homophones**: pairs that *sound* alike across EN↔FR
(the reservoir). It scores how alike two phrases sound, and — crucially — it already
stores a one-line English **gloss** of what each side *means* (`enGloss`, `frGloss` on
every reservoir row).

So every reservoir row secretly carries **two** signals:

- **sound** — `similarity` (do the phrases sound alike?)
- **sense** — the distance between `enGloss` and `frGloss` (do they *mean* the same?)

Homophone hunting used only the first. The Meaning Map adds the second and fuses them.
The grand goal, in the maintainer's words: *a map of meaning* where the two pair-banks —
the **homophone bank** (EN↔FR by sound) and the **semantic bank** (FR↔EN by meaning) —
are laid over one another, synonyms stacked on top, and the connecting chains used to
**whittle down to the phrases where sound and meaning converge** — the points where two
languages genuinely meet, in the ear *and* in the mind.

This is the seed of "perfection in all language to one another": find, and keep finding,
the places where the sound-bridge and the sense-bridge land on the same span.

---

## 2. The model

### 2.1 Resonance — fusing the two scores

For a pair *(E, F)* with phonetic similarity `p ∈ [0,1]` (reservoir) and semantic
similarity `s ∈ [0,1]` (LLM judging `enGloss` vs `frGloss`):

```
resonance(p, s) = sqrt(p · s)
```

A **geometric mean** — a single low factor tanks the score, exactly mirroring the
existing hybrid scorer's "both judges must agree" philosophy. Ranking the reservoir by
resonance *is* the whittle-down: the strongest resonances float to the top.

### 2.2 The sound×sense plane — four quadrants

| quadrant | sound | sense | what it is |
|---|---|---|---|
| **resonant** | high | high | sounds alike **and** means alike — the gems, true homophonic translations |
| **homophone** | high | low | sounds alike, means something else — the classic *Mots d'Heures* effect (the reservoir's bread and butter) |
| **translation** | low | high | means alike, sounds different — ordinary cross-lingual sense; anchors the meaning map |
| **weak** | low | low | neither |

Thresholds default to `phon ≥ 0.75`, `sem ≥ 0.6` and are query-tunable.

### 2.3 The graph — meaning islands and bridges

Nodes are phrases (lang-tagged). Two edge kinds:

- **PHONE edges** — the two sides of a reservoir pair (a sound-link, EN↔FR), weight = phonetic similarity.
- **SENSE edges** — phrases whose glosses denote the same meaning, weight = semantic similarity.
  v1 derives these cheaply: (a) any pair whose two sides already mean the same
  (`semantic ≥ sem`), and (b) any two phrases that share a normalized gloss (the "synonyms
  stacked on top" layer). Deeper LLM-judged synonymy is the next routine.

Over the SENSE edges we run **union-find** to get connected components — **meaning
islands**: clusters of phrases that all mean the same thing, regardless of language. Then:

- a PHONE edge **within** one island is a **resonance** (sound *and* sense agree);
- a PHONE edge **across** two islands is a **bridge** (a sound-link joining two distinct
  meanings — the classic homophonic-translation surprise).

As the reservoir grows, islands merge and the bridges that survive are the genuine
sound↔meaning coincidences. That convergence-over-time is the "whittling down."

---

## 3. The API

Hand-written Express routes (like `reservoir.ts`); no OpenAPI codegen required to run them.

### `GET /meaning/resonance`
Ranks reservoir pairs by resonance and labels each quadrant.
Query: `tier` (S/A/B), `minSim`, `limit` (≤200, default 60), `minResonance`, `phon`, `sem`.
Returns `{ thresholds, counts: {resonant,homophone,translation,weak}, total, pairs: [...] }`
where each pair carries `phonetic`, `semantic`, `resonance`, `quadrant`.

### `GET /meaning/map`
Builds the graph. Query: `tier`, `minSim`, `limit` (≤300, default 80), `phon`, `sem`.
Returns `{ thresholds, islandCount, nodeCount, bridgeCount, resonanceCount, nodes, edges, bridges, resonances }`.

Cost note: each endpoint makes at most `limit` LLM semantic calls (one per pair, cached,
4 concurrent). Bounded by design — no O(n²) gloss comparisons in v1.

---

## 4. Design choices & honesty

- **Reuse, don't re-derive.** The semantic layer reads the glosses the reservoir already
  produces; no new acoustic work, no schema change.
- **Pure core.** `meaning-graph.ts` has no I/O — phonetic scores come from the reservoir,
  semantic scores are injected — so it is trivially unit-testable.
- **Bounded LLM spend.** One cached call per pair, capped by `limit`.
- **No silent unification.** Sound and sense stay separate first-class scores; resonance
  is a *view* over them, never a replacement — same principle as the symbolic/acoustic split.
- **v1 SENSE edges are cheap (normalized-gloss equality).** This will miss true synonyms
  phrased differently ("a small boat" vs "a little vessel"). That gap is named, not hidden,
  and is the headline task of the next routine.

---

## 5. Where this is heading

The map is bilingual (EN↔FR) today because the reservoir is. The architecture is
language-agnostic: nodes are already language-tagged and the resonance math is symmetric.
Add a third language's reservoir and the islands span three languages; the bridges become
multi-hop. That is the path toward the maintainer's "all language to one another."

See **NEXT_ROUTINE.md** for the concrete next steps.
