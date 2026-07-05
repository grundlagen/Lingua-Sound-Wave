# Applied Fable: synonym layer -> reach restored. Light in tunnel?

## Did Fable's ask: graded ~sem WITHOUT embeddings

Built synonym (~sem) edges from the MUSE pivot: EN words sharing a FR translation
= synonyms (cap groups 2-12, skip hubs). 44,775 ~sem edges. muse-pivot-syn.tsv.

Fed into chain_game (v7+MUSE+syn). Reach RESTORED -- tree/fish/bush now route:
  bush ≈ bouche = mouth ≈ masse = mass ≈ masses ~ messes ≈ mess
  fish ~ peach ≈ patch = patch ≈ putsch
  sea  ≈ scie = saw ≈ saut = skip ...

## Your shortcut point is correct

7 hops drift to nonsense endpoints (sorta, soaked). Meaning drifts ~ chain length.
FIX = shortcut: prefer SHORTEST alternation; if an intermediate matches another
chain's node, cut through it (transitive). The sensical unit is the 2-hop:
  cat = chat ≈ chat     (means cat AND sounds like chat)  <- clean, no drift
Keep chains short; long chains are for reach-proof, not output.

## How far from sensical multilanguage generation -- LIGHT IN TUNNEL: yes

Have now, on v7:
  sound web        rich, AUC 0.993, routes        DONE
  exact meaning    MUSE 5856 edges                DONE
  synonym reach    44775 pivot ~sem, ~giant comp  DONE (Fable's 95%)
  carve engine     whole-line, fillers, fine-grain DONE
Gap to sensical OUTPUT (2 things, both known):
  1. L2 coherence model -- turns routed word-sequences into SENTENCES (bigram too
     weak; real LM/LLM). THE gate.
  2. shortcutting -- keep chains 2-3 hops so meaning stays tight, not drifting.

Verdict: the machine can now FIND, for almost any word, a path that is sound-true
and meaning-true (cat=chat). It cannot yet WRITE fluent sentences from those paths.
That last mile = one L2 model + shortcut search. Light is visible; the tunnel's
remaining length is the coherence model.
