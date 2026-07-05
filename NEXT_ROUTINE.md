# Next Routine — keep hunting the map of meaning

*A note from the previous Claude to the next one. Read MEANING_MAP.md first; it has the
model and the math. This file is the marching order: where I stopped, and the footsteps
to follow. Pick up the chain and keep whittling.*

---

## Where the last routine left it

- Built the **convergence layer** over the homophone reservoir:
  - `lib/semantic.ts` — LLM gloss-similarity judge (the *sense* signal), cached.
  - `lib/meaning-graph.ts` — pure engine: resonance score, sound×sense quadrants,
    meaning-island graph (union-find over SENSE edges), phone bridges vs resonances.
  - `routes/meaning.ts` — `GET /meaning/resonance` (the whittle-down ranking) and
    `GET /meaning/map` (the graph). Registered in `routes/index.ts`.
- **Not done yet** (deliberately, so the increment stays buildable without codegen):
  the frontend page, the OpenAPI spec entry, and richer-than-string-equality SENSE edges.

## Footsteps to follow (in order of leverage)

1. **Verify it runs end to end.** `pnpm run typecheck`. Then with a DB + mined reservoir:
   `curl localhost/api/meaning/resonance?limit=20` and eyeball the quadrants. Confirm
   *resonant* pairs really are sound+sense matches and *homophone* pairs really drift in
   meaning. Write a short honest report into `scripts/` like the poem-benchmark did.

2. **Smarter SENSE edges (the big one).** Today two phrases are "synonyms" only if their
   normalized glosses are byte-identical. That misses "a small boat" ≈ "a little vessel".
   Replace gloss string-equality with embedding similarity (or a batched LLM clustering
   pass) so meaning islands form on actual sense, not spelling. This is the "mapping
   synonyms on top" step the maintainer wants, and it makes the islands real.

3. **Frontend Meaning Map page.** Add a `MeaningMap.tsx` page to `homophone-explorer`:
   a scatter on the sound×sense plane (x = phonetic, y = semantic, the four quadrants
   shaded) plus a force graph of islands & bridges (react-flow / D3). Extend the OpenAPI
   spec with the two endpoints and `pnpm --filter @workspace/api-spec run codegen` to get
   typed hooks — match the existing Reservoir/Flit page patterns.

4. **Persist the sense layer.** Add `semantic real`, `resonance real`, `quadrant text`
   columns to `homophone_reservoir` (or a sibling table) and compute them during mining,
   so the map doesn't re-call the LLM on every request. The miner in `reservoir-mining.ts`
   already has the glosses in hand at grade time — score sense there.

5. **Chains longer than one hop.** v1 finds direct resonances/bridges. Build pathfinding
   over the graph: alternate PHONE/SENSE hops to travel from one meaning to a far one
   ("connecting chains"). Surface the shortest sound↔meaning path between any two phrases.

6. **Toward all languages.** The graph is already language-tagged and the math is
   symmetric. Stand up a second reservoir (EN↔ES or FR↔ES), let islands span three
   languages, and watch bridges go multi-hop. This is the road to "perfection in all
   language to one another" — incrementally, one pair-bank at a time.

## Principles to keep (inherited, do not break)

- **Honesty over magic.** If string-equality islands are crude, say so (this file does).
  Surface component scores; never hide complexity behind one number.
- **Both judges must agree.** Resonance is a geometric mean for a reason — keep it.
- **Pure core, bounded spend.** Keep `meaning-graph.ts` I/O-free and testable. Keep LLM
  calls cached and capped by `limit`. No O(n²) gloss storms.
- **Typecheck always passes.** Ship increments that build. If a step needs codegen you
  can't run, leave it as the next footstep rather than pushing broken drift.

*Go off. Follow the chain. Whittle it down.*
