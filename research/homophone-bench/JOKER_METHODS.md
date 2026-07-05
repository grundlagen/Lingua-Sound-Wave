# JOKER CLEF — reusable methods for our homophonic engine

The CLEF JOKER track (CC BY 4.0 corpus) is the open EN↔FR wordplay/pun-translation
benchmark. From the 2024/2025 overviews and the multi-agent method paper
(arXiv 2507.06506), the approaches that map onto what we're building:

## Methods worth reusing

1. **Combined phonetic-semantic embeddings** — score a candidate by BOTH phonetic
   similarity and semantic similarity, jointly. We already have both halves:
   prosody/combo (phonetic) + MiniLM multilingual embeddings (semantic). The
   JOKER lesson is to *combine* them in the reward (we do sound × validity now;
   add the semantic-cosine term Nemotron also recommended).
2. **Generator–discriminator multi-agent loop** — a generator proposes, a
   discriminator scores and sends it back to regenerate. Our `bank_composer` +
   `fr_coherence` (LLM judge) is exactly this; the JOKER framing says make the
   feedback explicit (regenerate the low-scored ones), which our best-of-N
   self-learning already does.
3. **Contrastive dataset** — train on good-vs-bad wordplay pairs so the model/
   reward learns the boundary. We can mint this for free: each English phrase's
   high-reward carve (positive) vs a low-reward / random French (negative) →
   feeds `experiment.py`'s separation metric and a future reward model.
4. **Creativity over literal duplication** — preserve the *effect* (sound + a
   coherent meaning), not the exact words. Matches our "both rails unanchored,
   source may drift" stance.
5. **Evaluation** — human humor/wordplay preservation + phonetic similarity +
   semantic adequacy + bilingual fluency. Mirror as: combo (phonetic), embedding
   cosine (semantic), LLM judge (fluency) — held out on JOKER's EN-FR pairs.

## Action items it suggests

- Add the **semantic-cosine** term to `reward.py` (we have the embeddings).
- Use the **JOKER EN-FR corpus** (CC BY 4.0, register on CLEF) as a held-out
  eval set for the judge — the one real external benchmark for this task.
- Build a **contrastive pos/neg set** from our reward to harden discrimination
  (the `experiment.py` finding said the scorer needs it).

## As methods (judging unchanged)

The semantic-cosine meaning signal is provided as `semantic_cosine.py` — a METHOD
for inspection, deliberately NOT wired into `reward.py`/`prosody.py`/
`fr_coherence.py`. It scores cosine(EN source, FR output): la mer 0.95, le chat
0.95 (translations) vs un voile d'or 0.37, un petit un petit 0.49 (sound-alike,
different meaning). Promote into the judge only by a deliberate, reviewed edit.
