# Tiered homophonic+semantic translation engine

`translate_engine.py` — type English, get French that aims to SOUND like the
English and MEAN the same, preferring sound fidelity and degrading gracefully.
Built on the full-scope v7 web + the same multilingual embeddings the weave used
(`node-vecs.npy`, all 195,270 nodes encoded by `cache_vecs.py`).

## The tier ladder (sound fidelity first)

| tier | rule | inflation |
|---|---|---|
| **S** | French homophone that already preserves meaning (loop-certified, ≥0.90) | 0 (1 hop) |
| **A** | homophone + synonym chain to the translation *or a synonym of it* (≥1 sound hop) | = chain length |
| **P** | poetic periphrasis: nearest reachable real-French metaphor | = chain length |
| lit | plain translation, no homophony (last resort) | 0 |

## It works on content words

```
"she has a new car"          ->  ... avoir ... nouveautés car
   key   -> clés    (A)   key → qui → keys → clés
   small -> petit   (A)   small → … → petit
   sees  -> voit    (A)   sees → … → voit
   has   -> avoir   (lit)
   she   -> maîtresse (P) she → … → mattress → maîtresse   (metaphor)
   frog  -> fléau   (P)   frog → … → fléau (scourge)        (metaphor)
```

Meaning lands correctly (`key→clés`, `small→petit`, `sees→voit`), and the poetic
tier supplies a metaphor when no literal homophone route exists (`frog→fléau`).

## Two honest findings

**1. The multilingual embedding contaminates monolingual output.** MiniLM puts
`key`, `ključ` (Slovenian), `llave` (Spanish), `chaves` (Portuguese) in one
cluster, so naive routing emits cross-language soup. Corpus frequency alone
(`wordfreq` zipf) does *not* fix it — it admits loanwords and place names
(`chaves`, `winstead`). The reliable gate is **lexical membership**: a French
output word must be an actual French *translation* (MUSE/dict French vocab) +
zipf≥2.5. With that gate the output is genuinely French.

**2. Sound fidelity vs reachability is the core tension.** Requiring only ≥1
sound hop, the chains are mostly *synonym* jumps — meaning lands right but the
rendering doesn't truly *sound* like the English end-to-end. A median route is
~5 hops, so per-word inflation is real. True homophony (mostly-sound chains)
collapses reachability, because the sound-only graph is sparse. This is the knob:

- **max fidelity** → restrict to S-tier / mostly-sound chains → few words translate
  homophonically, output is short and really sounds like the source;
- **max coverage** → allow synonym hops → most words get a French meaning-match,
  but the homophonic illusion weakens and the chain (output) grows.

## Sentence-to-sentence, shortest path, all hops written out

The engine takes a whole sentence, finds the shortest homophonic path per word
over the full embedding graph (path budget up to 14 hops — long is fine), and
writes out **every** intermediate hop (the "couple of pages"):

```
INPUT : "my friend loves the sea"

  [A] friend (11 hops): friend ≈ freine ~ frit ≈ frites ≈ feat ~ achievement
                        ~ success ≈ secs ace ~ ace ~ acy ~ acme ≈ amie
  [A] loves  (14 hops): loves ~ love ≈ laws ~ law ≈ lots ~ plenty ≈ plaine
                        ≈ playing ~ play ≈ plaie ≈ pleasure ~ plaisir
                        ~ apprécier ~ gern ~ aime
  [A] the    (9 hops):  the ~ den ~ das ~ se ~ si ~ fue ~ … ~ là ≈ lie ≈ la
  [A] sea    (8 hops):  sea ≈ scie ~ … ~ mot ≈ more ~ mer

FINAL FRENCH: "ias amie aime la mer"   (= friend loves the sea; 4/5 words land)
5 source words -> 46 total hops (46 written fragments).
```

The meaning endpoints are correct (`amie aime la mer`); the chains in between are
the written homophonic rendering. Two rough spots remain: a function word
(`my→ias`) with no good route, and a few proper-noun/foreign tokens
(`gern`, `fue`) the lexical gate still lets through on *intermediate* hops.

## Known limits / next levers

- **Function words** (`the`, `is`, `a`, `my`) have no good homophone and produce
  noise — they should pass through as grammatical French, handled separately.
- **Mostly-sound routing**: weight sound hops ≫ synonym hops in the search to keep
  the homophonic illusion, accepting lower coverage.
- **Multiword sound carves** (one English word → a French phrase in one sound hop)
  are the highest-value unbuilt piece for keeping fidelity *and* coverage.

Run: `python translate_engine.py "the key is gold"`
(needs `node-vecs.npy`/`node-ids.json` from `cache_vecs.py`, and `graph-v7u.pkl`.)
