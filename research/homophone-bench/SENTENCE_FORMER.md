# sentence_former — grammar-aware carve assembly (structure + neighbour effects)

The decoder emits sound-true word sequences with no idea of French syntax
("décennies code" — two nouns in apposition). This layer adds explicit French
sentence structure and, critically, models what each word choice does to its
NEIGHBOURS — the thing that makes French sentence formation non-local.

## The four passes (`sentence_former.py`)

1. **TAG** — POS + gender + number per word from **Lexique 3.83** (`pylexique`,
   PyPI-shipped, offline), with a closed-class function-word table taking
   precedence (Lexique is ambiguous exactly on the scaffold words: la = ART/PRO/NOM).
2. **SYNTAX** — score the tag chain with a hand-encoded French POS-transition
   grammar (no corpus needed): DET→NOM 1.0, clitic→VER 1.0, NOM→NOM apposition
   0.25, no DET/PREP sentence-finally (0.05), post-nominal adjectives default
   with a `PRENOMINAL` exception set (petit/grand/bon...). Geometric mean.
3. **REPAIR** — the neighbour effects, applied as rewrites with notes:
   - det–noun agreement: `un série → une série`, `le haine → la haine`
   - mandatory elision: `le ami → l'ami`, `de un → d'un`
   - prep-article fusion: `de le → du`, `à les → aux`
   - plural spreading: `les âne → les ânes` (only if the plural form is a real
     Lexique word)
   These CASCADE: `la accord → le accord → l'accord` (agreement feeds elision).
   Unit-verified on 10 constructed cases, including the no-op on already-correct
   `la mer est froide`.
4. **RESCORE** — repairs change the phoneme stream at exactly the boundaries the
   citation-form judge forgets, so sound is re-scored through `juncture.py`
   realizations. Per the one law, sound remains judged by the matcher/juncture
   combo — the grammar only proposes.

`joint = sound^1.0 × syntax^0.55 × fluency^0.35 × coverage`, with gates
sound ≥ 0.80 and coverage ≥ 0.80 (partial stubs must not outrank full carves
on syntax alone).

## Measured effect (demo run, honest)

- Ranking shifts toward sentence-shaped French: for "the sun is gone",
  **décennies cogne** (NOM VER, syn 0.73) now outranks **décennies code**
  (NOM NOM, syn 0.53) — same sound tier (0.90/0.90).
- "come to me now" → **con dominants / con dominant** (NOM ADJ, syn 0.77).
- Repairs rarely fire on current decoder output in the wild — the trie
  proposes few determiner+noun sequences — but the repair engine is verified
  and fires on constructed input. **The next lever is proposal-side**: make
  the decoder/poetry-mode offer determiner-led segmentations so the agreement/
  elision machinery has material to work on.

## Also in this commit: `sentence_selfimprove.py`

Bounded expert-iteration loop over the extant `corpus-phrases-en.tsv` DB:
compose (juncture on) → certify best-of-best into `certified-phrase-pairs.tsv`
→ densify the trie with whole-phrase fragments → re-measure the SAME eval set.
Honest result over 3 rounds × 16 phrases: gold bank 24 → 37 pairs; yield flat
at 62.5% — fragment re-injection alone does not lift yield on a small
function-word set (fragments don't overlap the other phrases' streams). The
lift, as the handoff already argued, must come from the L2/coherence side —
now also from the syntax score above.
