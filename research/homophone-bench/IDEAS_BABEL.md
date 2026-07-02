# The Babel list — every route from English sound to French meaning

_The comprehensive method catalog, from intuitive EN/FR phonology. Hand this to
Codex or any agent: each item is a mineable route or a composer channel. Status
tags: ✅ built · 🔶 partial · ⬜ open._

The high notion: **dual translation** — one phoneme stream, both languages read
it, meaning survives. The tower rebuilt not by one tongue, but by text that IS
two tongues at once. Word scope is solved (9,231 DUAL-S). Sentence scope is 55%
into the Rooten band. The rest of this file is the roadmap to 90%.

## A. Phonetic correspondence routes (the mouth's own dictionary)

1. ✅ th-stopping/fronting: the≈de, three≈trois/frit, thin≈fine, doth≈dot
2. ✅ h-dropping: hour≈art, hate≈été, who≈où/hou
3. ✅ w↔ou/v: wall≈vol, we≈oui, wine≈ouine, was≈vase, wee≈oui
4. ✅ rhotic swap ɹ↔ʁ (core EQUIV)
5. ✅ nasalization: in≈un, on≈on, and≈en, ban≈banc, men≈main
6. ✅ diphthong smoothing: day≈dé, low≈l'eau, my≈mais, high≈haïe
7. ✅ final-consonant latitude: dog≈dogue, bed≈bette, big≈bigue
8. ✅ silent French morphology: donne=donnes=donnent, petit=petits — FREE grammar
9. ✅ elision units mined systematically (9 contractions × 4k vowel-words in
   fr-units.tsv) and matched via the window channel.
10. ✅ liaison units (realized consonant) mined into fr-units.tsv AND proposed
    by the composer's window channel.
11. 🔶 e-muet elasticity: petite=[pətit]/[ptit] — both variants (bench.variants).
12. ✅ yod tricks + /ju/→/y/ realization variant (rule_aware.yod_to_y).
13. ✅ stress divergence tolerance (prosody.py DIVERGED)
14. ✅ cluster schwa-insert variant (rule_aware.cluster_schwa: tl/dn/kn/gn/pn/dl).
15. ✅ aspirated-h lexical list extended to 57 frequent aspirates (juncture.py)

## B. Cross-scope routes (one word ↔ many)

16. ✅ EN word → FR multi-word carve: enough≈un œuf, contrary≈contre air (Haiku),
    any≈haie nient (v7: 1,659 multiword streams)
17. ✅ EN multi-word window → one FR word/unit, IN the composer (made me≈m'admis
    0.78, the door≈adorent 0.72; babel_windows + beauty_compose merge step);
    one→many splits and the FR→EN mirror in babel_windows.py.
18. ✅ (approx) 3-word window merges in the composer (span-3 tried before span-2,
    thr 0.72); full sub-word seam re-segmentation folded into E38's juncture credit.
19. ✅ clitic LEGO: je/te/le/la/ne/se/y/en — dense FR monosyllables carve EN
    streams (poetry_mode fillers).
20. ✅ 1,500 FR compounds IPA-indexed into fr-units.tsv (window-matchable).

## C. Semantic widening routes (meaning survives, sound gets options)

21. ✅ transitive synonym chains, decay 0.85^hop (round_rabbit revived)
22. 🔶 antonym Haiku mode mined (llm_bridge --mode antonym); negation-wrapping
    at composition still open (polarity risk needs a meaning check).
23. 🔶 hypernym drift via Haiku ✅; WordNet BLOCKED in this env (nltk corpus
    download corrupted by proxy) — Colab/local task.
24. ✅ metonym Haiku mode mined+verified (+53 rows: garden≈gardien, full≈foule).
25. ✅ metaphor drift channel (sound≥0.6 ∧ cos≥0.25)
26. ✅ kenning Haiku mode mined+verified (+45 rows: great≈gré, fleece≈flèche).
27. ✅ homophone classes BOTH languages (4,582 FR / 706 EN) + composer pivots
    (enclass/frclass channels). Was: ⬜ FR homophone classes as free pivots: vert=verre=vers=ver=vair — group
    Lexique by identical phon; once ANY member matches the sound, CHOOSE the
    member whose meaning fits. (The biggest untapped one: French homophony is
    massive.) Same for EN: their/there, sea/see (final_verse EN side).
28. ⬜ polysemy split — DEFERRED: needs graph-v7u.pkl sense clusters wired to a
    sentence-context vector; medium refactor, no blocker.

## D. Register / lexicon expansions

29. ✅ archaic list (14 forms) in fr-units.tsv, window-matchable.
30. ✅ apocope variant added to fr_realizations (initial-schwa drop).
31. ✅ 15 verlan/argot forms in fr-units.tsv.
32. ✅ 16 place-names IPA-indexed in fr-units.tsv (window-matchable).
33. ✅ 16 interjections in fr-units.tsv, in the window channel.

## E. Composition & judging upgrades

34. 🔶 beam+trigram Viterbi (dual_poet): BENCHED WORSE than greedy joint-max
    (7/30 vs 55%) -- the trigram buys grammar with sound even above a 0.45
    floor. Verdict: greedy beauty_compose stays composer of record; salvage
    = conjugation families + Haiku fixer bolted onto GREEDY picks; raise the
    floor to 0.55 and use trigram only to break ties within equal-sound sets.
35. ✅ Haiku grammar-fixer constrained to sound-preserving edits, verified
36. ✅ rhyme-index.tsv + rhyme_pick.py: 119 rime families with ladder-grade
    French enders — couplets can end on the same sound in both languages.
37. ⬜ assonance bonus — DEFERRED (one scoring term in final_verse; low risk).
38. ✅ juncture upper-envelope credited in the composer's final verify
    (max(combo, best_juncture_score)).
39. 🔶 cycle-consistency check: back-carve FR→EN, require the round trip to land
    near the source (cycle_consistency.py exists as signal; wire as gate).
40. ⬜ constrained decoding — BLOCKED in this env (needs local llama.cpp/vllm
    logit-bias hook; no GPU). The endgame; Colab/Hetzner task.
41. ⬜ multi-line coherence: theme vector held across a whole poem (ladder has
    the seed_vec machinery).

## F. Data & model expansions

42. 🔶 Haiku mining at scale: 136 verified bridges from ~180 words at pennies —
    run the whole 9k content vocabulary (~$2), both directions.
43. ✅ reverse FR→EN index (dual-pairs-fr2en.tsv, 102,898; scoring symmetric).
44. 🔶 Lexique homophone sets now in (45 ✅ covers the FR side); PanLex/Wiktionary
    translations still open (large downloads).
45. ✅ real Lexique383 fetched: 33,659 authoritative homophone classes
    (fr-homophone-classes-lexique.tsv, merged into the composer); espeak
    validated a SECOND way (85% agreement on Lexique-homophone pairs).
46. 🔶 real-audio validation: espeak 0.97 vs real speech ✅; ⬜ scale clips, and
    run the ASR-confusion miner (decode French audio with an ENGLISH recognizer
    — its 'hallucinations' are discovered homophones).
47. 🔶 EN↔ES teaser running (babel_es.py on MUSE en-es) — same machinery, new
    floor of the tower; full pipeline port still open.

## G. Ladder answer (asked): are synonyms known in the ladder?

The graph knows them — hops-all has 44,775 `~syn` edges and muse-pivot-syn
carries the EN-EN / FR-FR synonym lists the composer chains over. The
tier-ladder TSV itself stores only pair provenance (no synonym column).
✅ `syn_cluster` column added (75,769 clusters; 2,852 with ≥5 ladder rows).

---
## Bench log (honest)

| composer state | Rooten band |
|---|---|
| word-aligned literal | 0% |
| + glue/chains/haiku (greedy) | 55% (22/40) |
| + class-pivots & 234 bridges, same weights | 45% (18/40) — priors displaced better picks |
| + REAL-cosine re-rank of top-K + window-merge | 48% (19/40) |
| + window gate rm≥0.35 (bypass 0.85) | **50% (20/40)** — trend right; remaining gap to 55%: enclass tail beyond top-10 uncalibrated. Clean fix stays: per-channel logistic on strict-gold |

Windows demo (B17/A9 live): sat at≈s'hâte · at the door≈s'adorent · door of the≈d'orage · the door≈d'ores.

*Verified flagship line so far:*
> the dog at the door made me cry → **le dogue hâtent le dors faite mi cris** (0.56/0.66)
> one day we shall cross the sea → **une dé oui châle cross le si** (0.76/0.51)
