# June-12 chain-game on v7+MUSE: reach + synonym gap

chain_game = alternation telephone: EN =trans=> FR ≈sounds=> EN =trans=> ...
+ at each step swap any node for a SEMANTIC NEIGHBOR (~sem). That swap is what
gave 99.6% giant-component reach (chain-web-stats).

## Ran on v7+MUSE (=trans real, ≈snd rich, ~sem OFF -- no embeddings here)

Real chains route:
  sea ≈ scie = saw ≈ saut = leap ≈ lippe          (6 hops, q 0.92)
  key = clé ≈ clan = clan ≈ clot
But tree, fish, bush DEAD-END -- because ~sem synonym swap is off.

## Synonym matching (the "extra words for semantic meaning") -- how close

MUSE is EXACT translation, ~1 synonym each: arbre<-tree only; buisson<-bush;
poisson<-fish/pisces. Sparse. So:
  tree  sounds-> fr:tri/tueries (homophone, not meaning)   -- direct fails
  bush  sounds-> fr:bouche (mouth)  -- "bush~arbre" loose, exactly your edge case
  plant (syn of tree) has 2 sound hops; fish->fiches works.
Synonym reach helps in EDGE CASES (a synonym with a sound-neighbor connects where
the head word doesn't) but MUSE gives too few synonyms for general reach.

## Critical (Fable lens) -- the missing best-part

The 95% reach came from ~sem GRADED semantic neighbors (embeddings), not exact
translation. We now have: sound web (rich) + MUSE exact-translation (dense-ish) but
NO graded synonym layer. That is THE gap for "add extra words to create semantic
meaning". Add multilingual embeddings -> ~sem edges -> chain-game reaches the giant
component again on v7, and periphrasis (big bush~arbre-grade) becomes routable, not
just lucky edge cases. Sound: done. Exact meaning: done (MUSE). Graded synonym:
the one missing piece = embeddings.
