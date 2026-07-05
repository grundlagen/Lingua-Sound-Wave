# Van Rooten joint search: one stream, source allowed to drift

The goal (corrected): replicate *Mots d'Heures: Gousses, Rames* — **one** French
line that, read aloud, *sounds like* the English and is itself coherent French.
Not a homophonic chain plus a separate translation — a single stream. And the
**English source may drift semantically** ("friend loves the sea" → "my mate
adores the water" → …) until a clean homophonic French stream settles.

`vanrooten.py` builds the synonym-swap / source-drift layer that
`generation_engine.py` flagged but never built, on top of the existing whole-line
carve:

1. **DRIFT** — paraphrase the English by swapping content words for their
   embedding-nearest English synonyms (`friend→buddies/mate`, `sea→ocean/marino`).
2. **CARVE** — for each drifted line, decode its *whole* phoneme stream into French
   words (boundaries fall where French words land), ranked by sound × French
   coherence (`whole_line_carve.carve_line`).
3. **SETTLE** — keep the single best (drift, carve) pair: `joint = sound × frenchness`.

Output is ONE French line + the English drift it sounds like.

## Result (`"friend loves the sea"`, 20 drifts)

```
joint 0.31  sounds like: buddies loves the sea      FR: but slaves des a
joint 0.26  sounds like: buddies loves the marino   FR: but élèves cheminots   (sound 0.54)
joint 0.25  sounds like: buddies amore the sea      FR: but étymologie
```

`but élèves cheminots` ("goal students railwaymen") is a real French word
sequence that sounds like "buddies loves the marino" — the Van Rooten effect,
unpolished.

## Honest state

The **mechanism is right** — one stream, source-drift, joint homophonic+semantic
— but quality is bounded exactly where every prior note predicted:

- **coherence** is a bigram LM, so it scores French *fluency*, not sense; the
  carve lands French-shaped but not yet sensical verse. A real L2 LM/LLM is the
  single upgrade that turns this search into the art.
- **carve pool**: the French unit inventory + filler set limits how cleanly an
  arbitrary stream tiles; multiword sound carves and a bigger unit pool widen it.
- sound and frenchness rarely peak together (0.54 sound vs 0.69 frenchness on
  different candidates) — the joint optimum is shallow with today's components.

So the pipeline is complete and does the correct thing; the remaining work is the
coherence model, not the search. Drop in a French LM that scores sense and the
same drift+carve loop yields Mother-Goose-grade lines.

Run: `python vanrooten.py "friend loves the sea"`
