# The Babel list — every route from English sound to French meaning

_The comprehensive method catalog, from intuitive EN/FR phonology. Hand this to
Codex or any agent: each item is a mineable route or a composer channel. Status
tags: ✅ built · 🔶 partial · ⬜ open._

The high notion: **dual translation** — one phoneme stream, both languages read
it, meaning survives. The tower rebuilt not by one tongue, but by text that IS
two tongues at once. Word scope is solved (9,231 DUAL-S). Sentence scope is 55%
into the Rooten band. The rest of this file is the路map to 90%.

## A. Phonetic correspondence routes (the mouth's own dictionary)

1. ✅ th-stopping/fronting: the≈de, three≈trois/frit, thin≈fine, doth≈dot
2. ✅ h-dropping: hour≈art, hate≈été, who≈où/hou
3. ✅ w↔ou/v: wall≈vol, we≈oui, wine≈ouine, was≈vase, wee≈oui
4. ✅ rhotic swap ɹ↔ʁ (core EQUIV)
5. ✅ nasalization: in≈un, on≈on, and≈en, ban≈banc, men≈main
6. ✅ diphthong smoothing: day≈dé, low≈l'eau, my≈mais, high≈haïe
7. ✅ final-consonant latitude: dog≈dogue, bed≈bette, big≈bigue
8. ✅ silent French morphology: donne=donnes=donnent, petit=petits — FREE grammar
9. 🔶 elision contractions as glue: l', d', j', qu', n', m', t' — vowel-initial
   words get a free consonant: door≈d'or, of≈œuf, the eau≈l'eau.
   ⬜ systematic: mine every (C' + vowel-word) against EN syllables.
10. ⬜ liaison-created consonants: les amis=[lezami] — mine EN words with /z,t,n/
    mid-cluster against FR word-pairs whose liaison produces them (mes amis≈"may
    zamee"). juncture.py scores it; the MINER doesn't propose them yet.
11. 🔶 e-muet elasticity: petite=[pətit]/[ptit] — both variants (bench.variants).
12. ✅ yod tricks: new≈nous, due≈doux; ⬜ EN /ju/ ↔ FR /y/ table (few≈fut, cute≈culte?)
13. ✅ stress divergence tolerance (prosody.py DIVERGED)
14. ⬜ French geminate/gap repair: schwa-insert to break clusters (ELISION_PROPOSAL 10)
15. ⬜ aspirated-h lexical list (h-aspiré blocks liaison — juncture has a stub list;
    extend from Lexique's h-words)

## B. Cross-scope routes (one word ↔ many)

16. ✅ EN word → FR multi-word carve: enough≈un œuf, contrary≈contre air (Haiku),
    any≈haie nient (v7: 1,659 multiword streams)
17. 🔶 EN multi-word window → one FR word: at the≈hâte, in a≈inné, say to≈sitôt.
    ⬜ systematic: slide 2-word EN windows over the sentence, match against the
    whole FR lexicon by IPA (the trie exists — phonetic_decoder inverted).
18. ⬜ portmanteau seams: allow a FR word to absorb the END of one EN word plus
    the START of the next (re-segmentation inside the composer, not just carve).
19. ✅ clitic LEGO: je/te/le/la/ne/se/y/en — dense FR monosyllables carve EN
    streams (poetry_mode fillers).
20. ⬜ FR compound nouns as targets: porte-clés, arc-en-ciel, chauve-souris —
    long fixed sound-strings with single meanings; index their IPA in the trie.

## C. Semantic widening routes (meaning survives, sound gets options)

21. ✅ transitive synonym chains, decay 0.85^hop (round_rabbit revived)
22. ⬜ antonym+negation: small → "pas grand" (sound: pas≈?); mine antonym pairs
    where the FR ANTONYM sounds like the EN word — then negate in composition.
23. 🔶 hypernym/hyponym drift: dog→dogue(breed)✅ via Haiku; ⬜ systematic via
    WordNet/wolf (French WordNet) hierarchies.
24. ⬜ meronymy/metonymy tables: crown→couronne/roi/tête; sea→marée/vague/sel.
    Haiku can mine these with a "name a PART or ASSOCIATE that sounds like" prompt.
25. ✅ metaphor drift channel (sound≥0.6 ∧ cos≥0.25)
26. 🔶 kennings/definition-unfolding: one EN word → FR mini-definition that
    sounds like it (water≈"eau taire" = water-that-silences ✅ via Haiku).
    ⬜ dedicated Haiku prompt: "French two-word kenning sounding like X".
27. ⬜ FR homophone classes as free pivots: vert=verre=vers=ver=vair — group
    Lexique by identical phon; once ANY member matches the sound, CHOOSE the
    member whose meaning fits. (The biggest untapped one: French homophony is
    massive.) Same for EN: their/there, sea/see (final_verse EN side).
28. ⬜ polysemy splitting on the EN side (ladder.py sense_clusters — wire into
    the composer so 'play' picks the right FR sense family).

## D. Register / lexicon expansions

29. 🔶 archaic-poetic French: ores≈or ✅ (v7 GOLD); ⬜ mine full archaic list
    (oncques, moult, céans, icelle) — extra sound inventory, licensed by verse.
30. ⬜ apocope/colloquial: p'tit, m'sieur, v'là — spoken French shortenings =
    more variants (add to fr_realizations).
31. ⬜ verlan & argot: meuf, relou, ouf — new sound shapes with meanings.
32. ⬜ proper-noun latitude: Lille≈little ✅ (used), Caen=[kɑ̃]≈can, Tours≈tours,
    Nice≈niece, Metz≈mess — mine the French gazetteer by IPA.
33. ⬜ interjections as near-free glue: oh/ah/hé/hein/bah/ouf/aïe — meaning-light,
    sound-precise; let the composer insert them where the EN has stray syllables.

## E. Composition & judging upgrades

34. ✅ beam+trigram Viterbi over candidate lists with sound floor (dual_poet)
35. ✅ Haiku grammar-fixer constrained to sound-preserving edits, verified
36. ⬜ rhyme/meter targeting: compose to octosyllabe/alexandrin; rhyme = shared
    final IPA (we have every word's IPA — a rhyme index is one groupby away).
37. ⬜ assonance/alliteration bonus in the beam (Van Rooten lines sing).
38. ⬜ post-pick re-segmentation: after words are chosen, re-carve the JOINED
    IPA stream allowing boundaries to move (juncture + whole_line_carve merged).
39. 🔶 cycle-consistency check: back-carve FR→EN, require the round trip to land
    near the source (cycle_consistency.py exists as signal; wire as gate).
40. ⬜ dual-rail LLM constrained decoding (METHODS_DEEP_DIVE): LLM writes French
    freely; matcher FST masks tokens that break the sound. THE endgame method.
41. ⬜ multi-line coherence: theme vector held across a whole poem (ladder has
    the seed_vec machinery).

## F. Data & model expansions

42. 🔶 Haiku mining at scale: 136 verified bridges from ~180 words at pennies —
    run the whole 9k content vocabulary (~$2), both directions.
43. ⬜ reverse direction FR→EN everywhere (dual_mine ran EN→FR only).
44. ⬜ bigger bilingual dicts: PanLex / Wiktionary translations (MUSE is 113k
    and noisy); Wiktionary also lists FR homophone sets ready-made.
45. ⬜ Lexique phon column: real French phonology WITHOUT espeak — free
    validation + the homophone classes of C27.
46. 🔶 real-audio validation: espeak 0.97 vs real speech ✅; ⬜ scale clips, and
    run the ASR-confusion miner (decode French audio with an ENGLISH recognizer
    — its 'hallucinations' are discovered homophones).
47. ⬜ other languages: espeak has 132 voices — the machinery is language-pair
    generic. Spanish/Italian nasal-poor phonologies are EASIER targets than
    French. The tower has more floors.

## G. Ladder answer (asked): are synonyms known in the ladder?

The graph knows them — hops-all has 44,775 `~syn` edges and muse-pivot-syn
carries the EN-EN / FR-FR synonym lists the composer chains over. The
tier-ladder TSV itself stores only pair provenance (no synonym column).
⬜ add a `syn_cluster` column: group ladder rows whose EN words are synonyms —
then a whole meaning-cluster's sound options are one lookup.

---
*Verified flagship line so far:*
> the dog at the door made me cry → **le dogue hâtent le dors faite mi cris** (0.56/0.66)
> one day we shall cross the sea → **une dé oui châle cross le si** (0.76/0.51)
