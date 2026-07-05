# The corrected objective, and the gold corpus

This supersedes impetus IV of `CENTRAL_PROBLEM.md`. Decision by the project owner.

## The objective, stated correctly

The point of the project is **the homophone aspect**: ONE phoneme stream that
reads as **sensical text in both languages at once**. The constraint *is* the art.

- **Retracted (was IV): "optimize meaning at the discourse level / shared theme."**
  This defeats the purpose. Anyone can write two thematically-related poems in two
  languages; that throws away the homophonic identity that makes this special. We
  do **not** loosen to "locally free, globally on-theme."
- **Kept (I, II, III):** an L2-coherence model; a passage-level held-out eval; and
  co-composition (open both sides + choose content the languages are in tune on).

The achievable, valuable target per item is therefore:

> `sound(L1_text) ≈ sound(L2_text)`  (tight, line-level — one stream)
> **AND** `L1_text` is coherent in L1
> **AND** `L2_text` is coherent in L2.

No "they mean the same" term — the gold homophonic poems prove that term is the
wrong one (they don't translate meaning; van Rooten's French is its own absurd
sense). What must be accurate for both is **independent coherence under one
sound**, not equivalence.

## Why this is the right line (not theme-matching)

`dataset.py` already carries the canonical case: *Humpty Dumpty sat on a wall* ⇄
*Un petit d'un petit s'étonne aux Halles*. The French means "a little one of a
little one marvels at Les Halles" — fully coherent French, **unrelated in meaning**
to the English, **identical in sound**. The magic is the dual-reading of one
stream. Theme-matching would discard exactly this.

## The gold corpus (the existence proof the machine must reach)

Human-crafted homophonic verse is where high sound AND high L2 coherence
**demonstrably coexist**. That makes it both the gold standard for the eval and
the training target. Ladder:

1. **Now — Mother Goose homophones.** Luis d'Antin van Rooten, *Mots d'Heures:
   Gousses, Rames* (1967): English nursery rhymes rendered as coherent French
   that sounds the same. Then Ormonde de Kay, *N'Heures Souris Rames* (1980), and
   John Hulme's German *Mörder Guss Reims*. These are the EN↔FR / EN↔DE gold.
   `gold-dual-readings.tsv` is seeded with the verified Humpty Dumpty line; add
   the rest of the verse here (one row each: source rhyme + homophonic rendering
   + provenance).
2. **Later — classical Arabic.** A far older, richer sound-constraint tradition:
   **jinās / tajnīs** (paronomasia — identical sound, divergent meaning),
   **tawriya** (double entendre), the virtuoso constraint pieces of al-Harīrī's
   *Maqāmāt*, and the **mulammaʿ** (macaronic) dual-language verse of al-Andalus.
   This both deepens the gold and pushes the system past EN/FR into a tradition
   built entirely on sound-carrying-two-senses.

## How the eval implements it

`dual_reading_eval.py` scores each gold item on the three axes above and — the
key measurement — holds **sound fixed** by comparing the human L2 line against a
**shuffle of its own words** (same vocabulary, same sound material, broken
syntax). The coherence **margin** between them is precisely the strength of the
L2-coherence model (impetus I): the bigram LM currently gives only ~+0.15 on the
gold, which is why a real L2 LM/LLM is the gating component. When a coherence
model ranks human craft over sound-matched salad on this gold by a wide margin,
it is good enough to drive both-sides-free generation (impetus III).

Sequence unchanged otherwise: **II (this eval, grow the gold) → I (the L2
coherence model, scored on it) → III (open both sides + content selection)**.
