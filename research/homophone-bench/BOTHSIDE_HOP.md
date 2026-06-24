# Both-side hopping + synonym-sub + inflation (demonstrated, honest)

Method (the ask): hop BOTH texts to fit. For each EN word seek a non-cognate FR
homophone (≈snd); if none, hop the EN word to a SYNONYM (~syn) that has one and
sub it (meaning kept); accept length inflation. From hops-all.tsv, own sentence.

  SOURCE:    cold winter wind covers the quiet trees with snow
  HOPPED EN: cold [winter] [wind] covers [the] calm* trees with [snow]   (*=syn-sub)
  FR:        cause cause came tri oued        (FR coh 0.40)

Whole-stream carve on the same (re-cut to fill word gaps):
  "covers the" -> conflit (0.45);  "cold winter wind" / "quiet trees" -> no carve.

## Critical (Fable) -- why thin, and the real fix

1. GAPS: most content words (winter, wind, snow, trees) have no clean FR homophone
   -- and they are CLUSTER-heavy (English clusters; French avoids them), so even
   whole-stream carve fails. This is a CONTENT problem: the source words don't
   carve. Fix = content-selection (content_neighbours.py): pick/sub carve-RICH
   words (crown, night, star) not cluster words.
2. SYNONYM-SUB works but the MUSE-pivot synonym layer is exact-ish, few options.
3. FLUENCY: bigram caps both sides at ~0.4.

I cannot neural-train here (no torch). But the pieces ARE the training program:
  - hop-train.jsonl + hops-all.tsv (typed moves)  = the dataset
  - both-side hop + synonym-sub + inflation        = the generation policy (shown)
  - content_neighbours                              = pick carve-rich source
  - matcher                                         = verify sound (never invent)
An LLM trained on this learns to (a) choose carve-rich content, (b) hop both sides,
(c) inflate with fluent fillers, (d) read as sentences. Demonstrated mechanically;
quality gated on that L2 model + content selection, both staged. No cheat, no
cognates.
