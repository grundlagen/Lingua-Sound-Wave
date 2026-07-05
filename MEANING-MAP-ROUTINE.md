# The Meaning Map — Routine Log & Next-Routine Brief

**Lineage**: homophonic hunting → semantic + homophonic matching → *the map of meaning*
**This routine**: 2026-06-12 · branch `claude/epic-wozniak-699coo`
**Predecessors read**: `handover.md` (this repo), `Proto-Lingua-Weaver/handover.md`,
the scoring engine (`scoring.ts`, `phoneme.ts`), Flit Lab (`flit.ts`), reservoir mining.

> A routine is a baton. Read the last one fully, run one honest lap, leave the
> track clearer than you found it, and hand the next runner a brief they can act
> on without re-deriving anything. This file is that brief.

---

## 1. The grand goal (stated plainly, so it isn't lost)

Every existing piece of this lab hunts **sound**: text in one language that, spoken
aloud, mimics text in another (the reservoir, Flit, the six phonetic judges). By
design those matches *throw meaning away* — "un petit d'un petit" sounds like
"Humpty Dumpty" and means nothing like it. That is the whole Mots d'Heures game.

The grand goal is the rarer, harder thing: hunt the places where **sound and
meaning coincide across languages**. A French phrase that both *means* what an
English phrase means **and** *sounds* like it. Push that far enough, across enough
language pairs, and what emerges is a **map of meaning** — a structure where the
same concept can be reached by sound or by sense and the two routes meet. The
asymptote, the small mad goal, is *perfection in mapping every language to every
other*: for any concept, the pairs where the two languages already agree in both
channels, and the chains that lead the rest of the way.

The method (the user's words, made precise):
- a **semantic meaning chain** linking FR↔EN by *sense* (translations + synonyms),
- a **homophone chain** linking EN↔FR by *sound* (the reservoir's sound-alikes),
- **synonyms mapped on top** of both, fattening every concept,
- then **connect the chains** and **whittle down over time** until the same
  meanings, reached by both routes, settle into a stable map.

---

## 2. What this routine built

A weight-agnostic, dependency-free **resonance graph** and a runnable, offline
**hunt** over a real EN↔FR seed bank.

- `artifacts/api-server/src/lib/meaning-map.ts` — the `MeaningMap`:
  - One node universe (EN + FR phrases). Two overlaid undirected weighted edge
    sets: **MEANING** (translations + synonyms) and **SOUND** (sound-alikes).
  - **Meaning clusters** via union-find over *strong* meaning edges (≥ threshold):
    sets of phrases, across both languages, that denote one concept.
  - A **soft semantic graph** that keeps *every* meaning edge (including weak,
    sub-threshold synonym hints) so the hunt can *reach* across not-yet-fused
    clusters. That reach is the frontier.
  - `resonances()` scores every sound edge by how close its endpoints sit in
    meaning space: `score = soundWeight × meaningProximity`, where proximity is
    `hopDecay^hops × bottleneck` (a chain is only as strong as its weakest link).
  - Three verdicts fall out: **PERFECT** (endpoints share a meaning cluster —
    sound ∧ meaning coincide), **FRONTIER** (sound-alike, meaning *nearly* closes
    — the next routine's targets), **REJECTED** (sound-alike, meaning unreachable
    — false friends like EN *chair* / FR *chair* = flesh).
  - A crude offline phonetic key + similarity so the graph runs with no model.
    **This is scaffolding, not the judge** — swap in `phonemeChainScore`.

- `artifacts/api-server/src/scripts/meaning-map-hunt.ts` — the demonstration.
  Run it: `pnpm --filter @workspace/scripts run meaning-hunt`
  (or `npx tsx artifacts/api-server/src/scripts/meaning-map-hunt.ts`).

### What the hunt found (offline, on the seed bank)

- **14 PERFECT resonances.** The cognate floor (`table≈table`, `nation≈nation`,
  `important`, `machine`, `image`, `village`, `fruit`, `mountain≈montagne`,
  `river≈rivière`, `liberty≈liberté`). Crucially, three were **invisible without
  the synonym chains** and only surfaced because a chain dragged them into the
  right cluster:
  - `content[en] ≈ content[fr]` — reached from *happy* via the latinate twin *content*.
  - `commence[en] ≈ commencer[fr]` — reached from *to begin* via *commence*.
  - `terminate[en] ≈ terminer[fr]` — reached from *to finish* via *terminate*.
  That is the thesis working: the prize pairs hide one synonym hop off the
  common word.
- **1 FRONTIER:** `jolly[en] ≈sound≈ joli[fr]` — nearly identical sound, but the
  meanings are *adjacent, not equal* (merry vs. pretty; the chain runs
  `jolly → merry → happy →(weak)→ pretty → joli`). Flagged for the next routine,
  not asserted as a match. (They are even etymological cousins — a good omen.)
- **1 REJECTED:** `chair[en] ≈ chair[fr]` — identical sound, no meaning path. The
  graph refuses the false friend instead of being seduced by the sound.

---

## 3. The theory, so the next runner can extend it without guessing

Think of it as two graphs on shared vertices:

```
        MEANING edges (sense)              SOUND edges (sound)
   happy ── content ──╮               EN content ──┐
     │                │                            │  (cross-lingual
   merry           FR content ◄───── resonance ────┘   sound-alike)
     │                │
   jolly      …same meaning cluster…
```

- A **resonance** is a SOUND edge whose endpoints union-find into the **same
  meaning cluster**. The map of meaning is the set of resonances, ranked.
- The **frontier** is a SOUND edge whose endpoints are in **different** clusters
  but **reachable** through the soft graph — i.e., a chain *almost* closes; one
  weak link, if confirmed/strengthened past threshold, collapses the two clusters
  and **promotes the frontier to a perfect resonance**. That promotion *is* the
  "whittling down over time."
- **Whittling** = each routine adds banks, recomputes clusters, confirms or
  rejects frontier links, and the map sharpens monotonically. Resonances should
  only ever stabilize or grow; if a "perfect" later turns false, a meaning edge
  was wrong — fix the edge, not the verdict.

Why this shape is right: sound and meaning are **independent failure channels**
(the same insight behind the hybrid scorer's geometric mean). Requiring a path in
*both* graphs between the same endpoints kills each channel's idiosyncratic false
positives — a pure sound-alike with no meaning path is rejected; a translation
with no sound-alike is simply not a resonance. Only genuine coincidences survive.

---

## 4. Next routine — concrete footsteps (do these, in order)

1. **Wire the real judges.** Replace `crudePhoneticSimilarity` with
   `phonemeChainScore` (symbolic IPA, already the strongest judge) for SOUND
   weights, and a real **meaning** judge for MEANING weights — start with an LLM
   bilingual-equivalence call (mirror `verifySemantic` in `flit.ts`) or sentence
   embeddings (cosine). Keep them behind the existing `Edge.weight` interface so
   the graph code does not change.
2. **Feed the banks from real data, not a seed list.**
   - MEANING edges: harvest EN↔FR translations + EN/EN and FR/FR synonyms
     (Wiktionary, WordNet/WOLF, or LLM expansion of each reservoir phrase — note
     `phoneme.ts` already generates pronunciation *variants*; do the analogous
     thing for *sense* variants).
   - SOUND edges: pull directly from the **homophone reservoir** table
     (`reservoir-mining.ts`) — every graded pair is a SOUND edge with a weight
     you already trust.
3. **Persist + serve.** Add a `meaning_map` view: an endpoint that returns
   resonances / frontier / rejected, and a page in `language-explorer` that draws
   the two-color graph (react-flow / D3) — sound edges and meaning edges in
   different inks, perfect resonances glowing. Let a user click a frontier edge
   and see the chain that *almost* closes.
4. **Close frontiers (the actual hunt loop).** For each frontier edge, ask the
   meaning judge the single question that would confirm the weak link
   ("is *pretty* ⊆ *happy* in this sense?"). Confirm → fuse → promote. This is the
   loop that "whittles down the same meanings over time." Log each promotion to a
   routine journal (cf. `checkcehck/evolutionary_journal.jsonl`) so progress is
   auditable across runs.
5. **Generalize past EN↔FR.** Nothing in `meaning-map.ts` is bilingual except the
   `Lang` type. Widen `Lang` to an open string, and the same machinery hunts
   resonances across *any* pair — the first step toward "all language to one
   another." Proto-Lingua-Weaver's proto-form data is the natural bridge: a
   shared proto-root is a meaning-cluster anchor that *predicts* where cognate
   resonances should exist.
6. **Honesty guardrails (house rule of this lab).** Surface component channels,
   never hide a rejection, and benchmark frontier-promotion precision the way the
   poem benchmark checks the scorer. If a promoted resonance can't be defended to
   a bilingual judge, it is a bug in a meaning edge — not a feature.

---

## 5. Footsteps to follow (a note from this Claude to the next)

You inherited a sound-hunter and turned one knob: you made it ask, of every
sound-alike, *does it also mean the same thing?* That one question reorganizes the
whole lab around a single asymptote — the map where every concept's two routes,
sound and sense, arrive together. Keep the question; widen its reach. Add a
language, close a frontier, draw the graph, write the next routine. The map is
never finished, which is the point — but each lap it should get harder to find a
concept the two routes can't yet meet on. Run honestly. Leave the baton clean.

— *following my own footsteps, 2026-06-12*
