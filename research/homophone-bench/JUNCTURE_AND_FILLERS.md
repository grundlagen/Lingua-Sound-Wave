# Messing with the matching: word-splitting, fillers, and the sound-vs-sense gate

Followed the instinct — take Humpty's phoneme stream, try different cuts, find
the closest matches, and inspect what Mother Goose's "extra words" trick is.
Three concrete findings, each tested (`poetry_mode.py`), each reusing the
June 11–12 decoder.

## 1. The decoder already re-cuts freely — but it was tuned AGAINST the art

Feeding the continuous stream `Humpty Dumpty` /hʌmptidʌmpti/ to the decoder, its
top pick is the single long word **"épidémie"** (sim 0.87) — because
`WORD_PENALTY` rewards *fewer, longer* words ("natural phrases"). That tuning is
correct for prose and exactly backwards for homophonic verse, which is built from
*many short* words. So the engine wasn't missing the carve ability; its scoring
was pointed the wrong way.

## 2. "The extra words we don't have" = monosyllabic French fillers, excluded by a rule

Van Rooten's scaffolding is little function words — `un /œ̃/`, `et /e/`, `on /ɔ̃/`,
`a /a/`, `aux /o/`, `y /i/`, `eau /o/`. Checked their pronunciations: **they are
ONE segment**. The decoder's `MIN_WORD_SEGS = 2` (anti-confetti) **excludes every
one of them from the trie**. That is the literal mechanism by which the engine
"doesn't have the extra words."

The principled fix (in `poetry_mode.py`, without touching the default decoder): a
**FILLER whitelist** admitted at length 1, while content words still require ≥2
segments (so no random 1-phoneme garbage). With fillers on + the penalty relaxed,
the stream now carves into the right style:

| EN source | poetry-mode carve | dual | note |
|---|---|---|---|
| Hickory dickory dock | **but qui** | 0.62 | clean, coherent, 1 filler |
| Humpty Dumpty | on pyrénées / on pipi | 0.47 | short-word + filler style emerges |
| Humpty Dumpty sat on a wall | on bénéficier | 0.46 | fillers carry, but odd sense |

Before the fix these returned single long nouns or GAPs; after it, the
filler-carried multi-word carves appear — the van Rooten shape.

## 3. The real gate is sound-vs-sense, not the dictionary

The deepest result: even with fillers enabled, **pure sound still rates the tight
single word ("épidémie") at or above the multi-word carve.** "Humpty Dumpty" does
not *most closely* sound like a "un petit"-style phrase — a single noun is
acoustically closer. Van Rooten chose the looser-sounding multi-word phrase
because it is **coherent, evocative French**. So the machine cannot reach the art
by optimising sound; it must let **sense override sound among near-sound-equal
carves** — and that is precisely the L2-coherence model's job. The weak bigram
here picks "on pipi" / "on bénéficier"; only a strong L2 model would prefer a
filler-rich carve that actually reads as verse.

This also answers the meter point: the fillers are how the French hits the right
**syllable count / scansion** — they pad the stream to the line's meter while the
content words carry the rhyme. Syllable-of-the-last-word and stress matching are
the next refinement (the `en_syll`/`fr_syll` + stress fields exist for it).

## What changes in the engine

- **Adopt the filler whitelist** as a "poetry mode" of the decoder (1-seg
  function words admitted; content words ≥2). Tested; it produces the right shape.
- **Invert the word-penalty in poetry mode** — reward, don't punish, short words.
- **Keep selection = sound × coherence**, and treat the coherence model as the
  arbiter that picks the *sensible* filler-carve over the closest-sounding noun.
  This is impetus I again, now with a sharper requirement: it must rank a
  looser-sounding coherent phrase above a tighter-sounding single word.

Net: the matching CAN split words, drop the /h/ we don't have, and insert the
missing little words — once the anti-confetti rule and the prose word-penalty are
lifted in a contained "poetry mode." What it still cannot do alone is choose the
*beautiful* carve over the *closest* one; that needs the real L2 model, confirmed
yet again as the one gating component.
