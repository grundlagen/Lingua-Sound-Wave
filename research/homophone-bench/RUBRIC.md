# Judging rubric — homophonic EN↔FR translation

How a candidate pair (English source → French carving) is scored, and how the
judge avoids the two failure modes we hit: (a) one metric grading by its own
methodology (the "0.993 on everything" artifact), and (b) a citation-form judge
forgetting connected-speech rules and so UNDER-rating true homophones.

The pair has two independent readings; both must hold for GOLD.

---

## Channel 1 — SOUND (does the French, read aloud, sound like the English?)

Scored by an **ensemble of orthogonal methods**, never a single metric:

| method | what it catches | file |
|---|---|---|
| `ngram_dice` (rule-aware) | exact phoneme-bigram overlap, order-aware (precision) | `rule_aware.py` |
| `feat_nw_sharp` / learned | articulatory alignment + gold-learned segment similarity (recall) | `bench.py`, `self_improve.py` |
| `prosody` | stress/rhythm: EN stress-timed & falling vs FR syllable-timed & phrase-final | `prosody.py` |
| `drive_equiv` | your Drive phoneme-equivalence classes, SequenceMatcher | `drive_phoneme_map.py` |

**Rule-aware, not citation-form.** Every method scores the MAX over the
*connected-speech realizations* of both sides — flapping, l-vocalization,
h-dropping, th-fronting, schwa-elision, French e-muet, liaison, nasal split. A
legal realization can only RAISE a true homophone, never invent one. (`rule_aware.py`)

**Combining the channel:**
- **Lenient / calibration:** z-normalize each method, average (`hard_judge.py`).
  No single methodology dominates; LLM arbitrates highest-disagreement cases.
- **Strict / promotion:** GEOMETRIC MEAN (AND-logic) — *all* methods must agree;
  one sceptic drags it down (`strict_judge.py`).

**The honest number is the strict GOLD-RATE, not AUC.** AUC only asks "is the
positive ranked above the negative?". Strict gold-rate asks the real question:
does the true homophone clear an absolute bar **and** beat its *nearest
confusable rival* by a margin?

| score | sound grade |
|---|---|
| geo-ensemble ≥ 0.60 **and** beats nearest rival by ≥ 0.10 | **STRICT-GOLD** sound |
| geo-ensemble ≥ 0.60 | gold sound (citation) |
| 0.45–0.60 | loose / needs rule-aware lift to qualify |
| < 0.45 | reject |

Negatives for any benchmark must be **adversarial**: each English word's
negative is the French word that *sounds most like it but means something else*
(argmax `combo` over the pool). Random/length-matched decoys inflate AUC to ~0.99
and must not be used to claim accuracy.

### LLM arbitration (optional, costs money — currently 402/exhausted)
DeepSeek/OpenRouter on the highest-disagreement or borderline pairs only.
Harsh rubric: 90–100 indistinguishable; 70–89 same word, accent only; 40–69
noticeably off; 0–39 not the word. Reserve ≥70 for genuine matches. Used to
calibrate the symbolic bar, **not** as the sole judge.

---

## Channel 2 — MEANING (does the French also read as coherent, related French?)

`semantic_cosine` = cosine of multilingual MiniLM embeddings
(`paraphrase-multilingual-MiniLM-L12-v2`). This is the dual-reading gate (the
Van Rooten property: sounds English, reads French). It is a **standalone method
and a gate — never wired into the sound judge.**

| cosine | meaning grade |
|---|---|
| ≥ 0.45 | admissible dual reading |
| < 0.45 | sound-only homophone (lower tier) |

---

## GOLD definition

```
GOLD          = sound (geo-ensemble ≥ 0.60)            ∧ meaning (cosine ≥ 0.45)
STRICT-GOLD   = sound beats nearest rival by ≥ 0.10    ∧ meaning ≥ 0.45
```

Note: the v7 `GOLD` tier was defined by `prosody ≥ 0.70 ∧ meaning ≥ 0.45`, which
is **softer** than the strict bar — at the strict gold-rate only ~56% of v7-GOLD
pairs clearly beat their nearest rival. Treat **STRICT-GOLD as the frozen eval
set** and the strict gold-rate as the metric the self-learning loop must move.

---

## Corpus principle

The corpus of creation need **not** be word-to-word. The winning sound methods
score *phoneme streams*, not lemma equality, so a whole-line carving that no
single French word matches is admissible — judged by the same ensemble over the
line's phoneme stream. Word-pairs are the seed, not the ceiling.

---

## Self-improvement metric

`self_improve.py` runs expert iteration: learn a phoneme-similarity model from
gold alignments → re-score candidates rule-aware → promote STRICT-GOLD → relearn.
Success = the **frozen-eval strict gold-rate rises then plateaus** across
iterations. Optimise that curve, not AUC.
