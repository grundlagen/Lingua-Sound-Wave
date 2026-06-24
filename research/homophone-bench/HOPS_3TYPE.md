# Hops are 3-typed: sound + exact-meaning + synonym

Corrected: a hop is not just ≈/=. Full move-set in hops-all.tsv (62,271):
  ≈snd   9,912   sound (homophone, crosses language)
  =trans 7,584   exact translation/cognate (MUSE+cog, in/cross-lingual meaning)
  ~syn  44,775   SYNONYM (graded meaning neighbour, MUSE-pivot)

Writing/translation = a PATH mixing all three. e.g.
  tree ~syn wood =trans bois ≈snd bois...   (synonym to reach a sound-able word)
The synonym hop is what gives reach (Fable's 95%): swap to a near-word that DOES
have a sound bridge. hop-train.jsonl should carry all 3 types.

## Critical (Fable lens)

~syn hops are LOSSY -- each is a small meaning shift, so a path with many synonym
hops DRIFTS (tree->wood->plank->...). Therefore hops-all.tsv tags TYPE + WEIGHT so
the model / shortest-path can:
  - fidelity mode (translation): minimise ~syn, prefer =trans + ≈snd, short paths.
  - wordplay mode (pure homophonic): ~syn freely, length/drift fine.
The 3 types are not interchangeable -- ≈snd is sound-exact (matcher-verified,
never invent), =trans is meaning-exact, ~syn is meaning-approx. An LLM trained on
typed+weighted hops learns WHICH move to spend where: that is the whole control
knob between "faithful homophonic translation" and "free homophonic poetry".

Feed: hops-all.tsv (typed moves) + subway-hubs (shortest) + carve lines
(expansion) -> LLM writes, and the type-mix dials fidelity vs play.
