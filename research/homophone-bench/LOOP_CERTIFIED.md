# Loop-certified pairs — corrected (I was wrong before)

I previously split these into "19 meaning-grounded vs 102 drift". WRONG. In
chain_game the meaning-family is {= translation, ~ semantic-neighbor} -- ~ IS a
meaning hop. Verified: all 121 S-tier loops STRICTLY alternate sound(≈) and
meaning(=/~) and use both. So every one is genuinely meaning-round-trip certified.

## What loop-certification actually means (learned)

A loop = leave a word by SOUND, move by MEANING, alternate, and RETURN to the
seed's meaning neighborhood. If meaning round-trips through a sound edge, that
edge is a TRUSTWORTHY dual-translation bridge -- sound preserved (≈) AND meaning
recoverable (the cycle closes). Not "fr means en" literally, but "you can route
sound through this pair and get the meaning home." Self-supervised, no judge.

  succeed ≈ secs ides : sec ≈ secs ~ secs ides ≈ succeed ~ success ≈ secs ace ~ secs
  (strict ≈ ~ ≈ ~ ≈ ~ alternation, returns to secs -> certified ×10)
  pleat ≈ plisse : plead = plaider ≈ pleader ~ plea ≈ pli ~ plisse ≈ pleat

## Numbers (from the uploaded weave)

  full v5 web : 539 loops -> ~410-440 loop-certified pairs
  S-tier web  : 169 loops -> 121 certified  (loop-certified-S.tsv)
S edges cover 61% of seeds but 31% of loops -- looser A-edges let chains bend home.

## This is the gold dual-translation layer

Translate THROUGH loop-certified edges -> sound + meaning-recoverability by
construction. The bootstrap (certify -> promote -> re-weave -> more loops) grows
it deterministically, no LLM. The full set (410-440) is the seed; my earlier
reduction to 19 was the shoddy version. Use loop-certified-S.tsv as the trusted
edge set for the generation engine.
