# Training on the homophonic tradition — what's legal, and our PD gallery

## The copyright facts (so we stay clean)

- **Luis van Rooten, *Mots d'Heures: Gousses, Rames* — 1967, in copyright.** The
  often-cited "1916" is *The Real Mother Goose* (the English nursery-rhyme
  *source*), which is public domain. So the English rhymes are free; Van Rooten's
  **French is not**. We do not reproduce or train on it.
- **Ormonde de Kay, *N'Heures Souris Rames* (1980)** and **John Hulme,
  *Mörder Guss Reims* (1981)** — also in copyright.
- **James Joyce, *Finnegans Wake* (1939)** — dense multilingual *punning*, but not
  systematic homophonic *translation*; and PD only in life+70 countries (since
  2012), not the US (until 2035). Not a usable training source.

**Conclusion:** the homophonic-translation *canon* is in copyright. We cannot
train on it. But the **English nursery-rhyme corpus is centuries-old folk verse**
(public domain), and we generate the French ourselves — so we can build our own
gallery legally.

## Our PD gallery (`corpus-carves.tsv`)

`corpus_bank.py` carves ~60 traditional rhyme lines sound-first with balanced
selection. A complete, original homophonic translation per line:

```
Humpty Dumpty sat on a wall   -> un petit un petit et on éole   (combo 0.52)
to fetch a pail of water      -> fait est chapiteaux or          (0.55 / flu 0.58)
like a diamond in the sky     -> laquais amènent un deux cas      (0.52)
Hickory dickory dock          -> cries critiques                 (0.62)
gently down the stream        -> gênent les en ces trime         (0.62)
London bridge is falling down -> land on bridges pollen âne       (0.58)
come again another day        -> camion un aux deux               (flu 0.65)
```

The French is ours (generated), the English is PD folk verse — nothing
copyrighted is used. This is the Van Rooten *method* applied to legal material.

## How this feeds the engine

- A **standalone gallery** of homophonic translations (the deliverable itself).
- **Content-rich material** for the bank/composer: unlike frequent-bigram phrases
  (function-word heavy), these lines carry subjects (cat, moon, water, bridge), so
  splitting them into units gives the composer something for `--theme` to steer
  toward — the limitation we hit when theme-steering the function-word bank.

The real ceiling is unchanged: a true L2 model would turn these sound-true carves
into *sensical* French verse. Everything upstream (PD source → sound-first carve →
balanced selection → bank → composer) is in place and legal.
