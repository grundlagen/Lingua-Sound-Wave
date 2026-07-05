# French-anchoring, unified routing, and ideal hop distance

Toward the homophonic+semantic translation engine: type English → match the
English dictionary → chain-hop to a French word that *means* it → write out the
whole chain (every intermediate hop is an output word). So **chain length = output
inflation**, and the engine wants the shortest possible hops.

## 1. Unified routing — any of the 3 edge types per hop  (`hoproute.py hop`)

Routing now takes whichever edge is shortest at each step — homophone `≈`,
translation `=`, or synonym `~` — no forced alternation. This connects far pairs
the strict-alternation router could not:

```
key  → argile :  en:key = fr:clé ≈ en:clay = fr:argile                    [3 hops]
love → guerre :  love = aimer ≈ emcee ≈ ensi ≈ enter = saisissez ~ chercher
                 ~ obtenir = getting ≈ guerre                              [9 hops]
cat  → fromage:  cat ≈ quitte ≈ cleat ~ cleats = crampons ~ crampe ~ crepe
                 = crêpe = pancake ~ cheesecake ~ cheese = fromages ~ fromage [12]
```

## 2. French-anchoring — symmetric, but mostly redundant  (`french_anchor.py`)

We ran the AUC-0.993 matcher in reverse (French headword → English homophone) to
densify the French sound side: **290 frequent French words anchored ≥0.80, 157 at
S-tier** — `qui~key, si~see, tout~to, nous~new, cette~set, faire~fair, sous~sue`.

But it does **not** shorten hop distance, and the reason is structural:
**homophony is symmetric**. 187 of the 290 pairs (64%) were *already* edges from
the English-anchored side (`qui~key` is the same edge as `key≈qui`). Only 103 are
new (`fils~feel, cher~share, texte~text, celles~sell`), too few to move the metric.

**Takeaway:** anchor *direction* is not the lever for the French side. Symmetric
homophone retrieval rediscovers the same edges. To genuinely densify you need
different evidence (looser threshold, multiword carves, or a bigger lexicon
sweep), not the reverse direction.

## 3. Ideal hop distance — the inflation budget  (`hoproute.py ideal`)

Shortest homophonic chain (sound+synonym edges, ≥1 sound hop) from an English
word to a French word that means it, over 500 sampled words with a real
translation:

| target rule | reachable ≤10 hops | median | ≤3 hops | ≤5 hops |
|---|---|---|---|---|
| **strict** (exact dictionary translation) | 27% | 5 | 6% | 14% |
| **loose** (translation *or a synonym of it*) | **49%** | 5 | 12% | 25% |

**Accepting a synonym of the translation nearly doubles coverage (27%→49%)** — the
biggest single lever. The engine should target a *meaning-equivalent set*, not one
dictionary word.

The zero-inflation ideal is **1 hop**: an English word whose French homophone
*already* preserves meaning. Those are the **loop-certified pairs** — 813 (full)
/ 135 (S-tier). For those words the translation costs no extra written words at
all; they are the engine's gold primitives.

## What this means for the engine

- **Prefer 1-hop loop-certified pairs** as the backbone (no inflation).
- **Target synonym sets, not single words** (27%→49% reachable).
- A median homophonic route is **5 hops = 5 written French words per source word** —
  real inflation, so the engine must *choose* source words/phrasings that have
  short routes (or tolerate partial homophony on the rest).
- French-anchoring by reverse retrieval is a dead end for densification; the live
  levers are synonym-relaxed targets, the loop-certified backbone, and multiword
  sound bridges (one source word → several French words in a single sound hop).
