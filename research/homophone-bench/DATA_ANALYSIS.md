# Deep data analysis — homophone-bench

_How the datasets relate, what the schwa-elision core actually does, and the ladder graph that links pairs into dual-reading lines._

## A. Dictionary lineage (v2 → v7)

| version | rows | tier column | top tiers |
|---|---|---|---|
| v2 | 8319 | col0 | B:5570  A:2331  S:418 |
| v3 | 9684 | col0 | B:6604  A:2341  S:739 |
| v4 | 10652 | col0 | B:6604  A:2745  S:1303 |
| v5 | 11788 | col0 | B_reservoir:5703  A:3224  S:1960  B_safe:901 |
| v6 | 2351 | col0 | B:1764  A:461  S:126 |
| v7-remined | 9912 | col5 | A:5071  GOLD:4802  B:39 |

v7-GOLD pairs: 4802;  already in v5: 2614 (54%);  in v6: 831 (17%).  The GOLD set is mostly a re-scoring of long-lived pairs, not new mining.

**Tier-history answer (your "S was higher before" memory is right):** the top
tier was **S** in v2–v6 and was a *different, stricter* thing — LLM
coherence-on-both-sides (≥ 2/3) **and** sound-similarity ≥ 0.85
(`tier-grader.ts`). Its count grew v2→v5 (418 → 739 → 1303 → **1960**) then the
v6 rebuild kept only 126. v7 renamed the top tier **GOLD** and redefined it as
`prosody ≥ 0.70 ∧ semantic-cosine ≥ 0.45` — bigger (4802) but a *softer,
symbolic* bar with no coherence check. So v7-GOLD is larger in count yet lower in
bar than the old S. Restoring S = re-add the LLM coherence gate on top of the
strict sound bar (see RUBRIC.md).

## B. The schwa-elision / cheap-gap core — which rules carry the corpus

Aligned 2614 GOLD pairs (those with IPA). Share of pairs whose alignment RELIES on each rule:

| rule | pairs using it | share |
|---|---|---|
| schwa elision (ə/ɚ/ɐ/ɜ gap) | 125 | 5% |
| h-dropping (h gap) | 0 | 0% |
| offglide drop (ʊ/ɪ/j/w gap) | 261 | 10% |
| EQUIV-class substitution | 1456 | 56% |
| exact-segment only (no rule) | 842 | 32% |

Total alignment operations: 19753  (exact 8074, other subs 2293, gaps 1312).

Most-deleted segments (the elision workhorses):
  s×143  ɪ×142  t×141  ɹ×137  n×99  ʊ×88  d×86  ə×78  z×50  ɐ×48

Most-used EQUIV substitution classes (the cross-language merges):
  i~ɪ×276  a~æ×135  ɛ~ɪ×117  d~t×104  ə~ɛ×103  e~i×94  e~ɛ×82  e~ɐ×74  a~ɑ×73  s~z×60  ɒ~ɔ×56  ɔ~ə×56

> Reading: the schwa/elision core is not decoration — it is load-bearing. A large share of GOLD homophony only exists because reduced vowels, /h/ and offglides are allowed to vanish cheaply, and lax/tense vowels merge across the languages.

## C. The ladders — the graph that links pairs into dual-reading lines

Pairs are not isolated: a *hop* connects one carving to another that shares sound or sense, a *chain* is a path of hops, a *loop* is a chain that returns to its seed (both rails reconcile), and *loop-certified* pairs are the atoms that survive in ≥1 loop — the alphabet `ladder.py` composes with.

**chain-web-v7u**: 12170 src→dst routes.  hop-length: 2-hop:619  3-hop:1079  4-hop:518  5-hop:1767  6-hop:8187
**transfer-ranked-v7u**: 12170 ranked transfers, mean sound 0.90, mean cos 0.43.
**chain-loops v7u**: 1109 closed loops, hop-lengths {3: 168, 4: 562, 5: 300, 6: 79}.
**chain-loops v7u-aug**: 1222 closed loops, hop-lengths {3: 190, 4: 639, 5: 320, 6: 73}.
**loop-certified v7u**: 813 dual atoms; 619 certified in ≥2 loops (max 12× certifications).
**loop-certified v7u-aug**: 896 dual atoms; 697 certified in ≥2 loops (max 12× certifications).

**hops-all**: 62271 edges, by type: ~syn:44775  ≈snd:9912  =trans:5856  =cog:1728

## D. Reusable sub-word fragments (the carving vocabulary)

2640 EN-chunk → FR-chunk fragments mined from carvings. Top reused:

| count | en chunk | fr chunk |
|---|---|---|
| 372 | `st` | `st` |
| 244 | `ɹi` | `ɹi` |
| 227 | `ɛk` | `ɛk` |
| 199 | `ɛs` | `ɛs` |
| 191 | `ks` | `ks` |
| 184 | `ɛn` | `ɛn` |
| 183 | `tɹ` | `tɹ` |
| 170 | `ɹɛ` | `ɹɛ` |
| 167 | `li` | `li` |
| 148 | `sɛ` | `sɛ` |
| 141 | `pɹ` | `pɹ` |
| 137 | `si` | `si` |
| 125 | `ɛks` | `ɛks` |
| 104 | `ɛst` | `ɛst` |
| 104 | `ti` | `ti` |

## E. learned-costs overlay (what the S-tier + loop alignments tightened)

Mined from S-tier + loop-certified alignments: 89 substitution costs and 20 gap costs were lowered below the hand table (validated AUC 0.989 → 0.994).

cheapest learned substitutions (strongest cross-language merges):
  i~ɪ=0.15  ə~ɛ=0.16  ɛ~ɪ=0.16  e~ɐ=0.16  d~t=0.18  a~æ=0.18  e~ɛ=0.19  ɔ~ə=0.19  a~ɑ=0.19  e~i=0.19

learned gap costs: s=0.15  t=0.15  d=0.17  ɹ=0.20  z=0.20  ɐ=0.21  ɪ=0.24  ə=0.25  ʒ=0.25  n=0.25  f=0.25  j=0.26  ð=0.29  w=0.30  θ=0.33  e=0.36  ʊ=0.38  ŋ=0.42  i=0.42  l=0.42

> The old S-tier did not vanish — its alignments are baked into the live matcher's EQUIV/CHEAP_GAP. The schwa/elision model you like is partly LEARNED from that higher tier.

