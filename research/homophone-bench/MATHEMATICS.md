# The mathematics of dual translation

_What this system is, stated precisely — and which branches of mathematics
transfer, with the concrete upgrade each one buys. Ranked practical-first._

## 0. The objects

- IPA segments form an alphabet Σ; phoneme strings live in the free monoid Σ*.
- G2P maps g_E : E → Σ*, g_F : F → Σ* (validated: espeak ≈ real speech 0.97).
- Sound similarity s = 1 − d/len, where d is a **weighted edit distance** with
  learned substitution costs. If the segment cost matrix is a metric and gaps
  constant, d is a metric on Σ* — so (Σ*, d) is a metric space and
  "homophone" = "inside a small ball".
- Meaning lives twice: as an embedding μ : text → S^{d−1} (cosine), and as a
  **relation** R ⊆ E × F (translation/back-translation edges).

## 1. Two partitions and their intersection (the set theory, made exact)

g_F induces an equivalence on F: x ~ y iff g_F(x) = g_F(y). The equivalence
classes are the homophone classes (vert=verre=vers=ver=vair) — 33,659 of them
from Lexique. Same on E (sea=see, their=there).

The translation relation R induces a **Galois connection** between ℘(E) and
℘(F): A ↦ A′ = {f : ∀e∈A, eRf} and dually. Its closed pairs form the
**concept lattice** (Formal Concept Analysis) — our synonym clusters are its
coarse shadow. The DUAL lexicon is the intersection

    D_θ = R ∩ S_θ,   S_θ = {(e,f) : s(g_E(e), g_F(f)) ≥ θ},

and the **tier ladder is the filtration** {D_θ}_θ: nested sets as θ tightens.

**Transfer (cheap, real):** rank pairs by *persistence* — the largest θ at
which the pair survives. Persistence, not a single score, is the robust tier:
it is exactly why strict-gold (survive θ=0.6 AND beat the nearest rival)
outperformed raw AUC. Formalizing tiers as a filtration makes every future
threshold choice principled.

## 2. Composition = two covers over one selection (and why greedy is safe)

A rendering of stream τ ∈ Σ* selects units u₁…u_k whose concatenated sound
segments **exact-cover** τ (an ordered partition — a path in a lattice DAG),
while their meaning sets **set-cover** the content universe U.

The coverage function C(S) = |⋃_{u∈S} M(u) ∩ U| is **monotone submodular**.
Therefore greedy selection carries the Nemhauser–Wolsey–Fisher guarantee:
C(greedy) ≥ (1 − 1/e)·OPT. `set_dual.py`'s greedy is not a heuristic hack —
it is the canonical approximation with the classical bound. Budgeted version
(maximize coverage subject to a sound-cost budget) is submodular knapsack:
use the cost-ratio greedy + lazy evaluation (CELF) when scaling to paragraphs.

## 3. The decoder is a WFST composition (formal language theory)

The whole pipeline is one algebraic object:

    BEST-PATH( T_g2p ∘ T_rules ∘ T_lex ∘ T_LM )

- T_rules: the connected-speech rules (elision, liaison, flapping, th-stop)
  are a **string rewriting system** on Σ*. Matching modulo rules = the word
  problem in the quotient monoid Σ*/≈.
- **Transfer (big speed + correctness):** run Knuth–Bendix completion on the
  rule set → confluent, terminating system → every string has a canonical
  normal form. Then match normal forms ONCE instead of max-over-16×12
  variants. Same math, 100× cheaper, and no missed variant combinations.
- Constrained decoding (E40) is precisely **automaton intersection**: the
  LLM's next-token automaton ∩ the sound transducer's viable-continuation
  language. The GPU logit-bias recipe is this intersection computed lazily.

## 4. Meaning migration = optimal transport

Paragraph-level slack is a transport problem: supply = meaning mass on the
English content words; demand = sound capacity of each French span; cost
c(e, f) = 1 − fit. The best paragraph assignment is min-cost transport —
Hungarian for hard assignment, **Sinkhorn** for the soft/differentiable form.

**Transfer (direct quality win):** replace single-vector cosine with **Word
Mover's Distance** (OT over MiniLM word vectors) as the paragraph meaning
score. Cosine of one pooled vector under-measures long text; WMD measures
exactly "did every meaning atom land somewhere", which is our cover metric's
continuous twin. One function swap in the judge.

## 5. Why duals must exist (information theory + probabilistic method)

Natural language is ~50–75% redundant (Shannon). A phoneme stream carries H
bits; an English parse and a French parse each need only a fraction — the
**redundancy budget is what the art spends**. A dual text is one codeword
decodable under two decoders: a broadcast-channel object. Existence is a
counting statement: paraphrase space grows exponentially in sentence length
(synonyms × word order × register per clause) while the sound constraint is
LOCAL (per span). In Lovász-Local-Lemma shape: if every span has ≥ d
sound-viable meaning-true options and constraints only couple adjacent spans,
a satisfying assignment exists. This is the precise form of the claim
"we are search-limited, not existence-limited" — and it says exactly what to
grow: **options per span** (lexicons, paraphrases), not global cleverness.

## 6. The judge is a log-linear model (statistics — the named open task)

The channels (dual, ladder, glue, chains, bridges, windows...) are features;
the composer's weights are a log-linear model. Tuning them on strict-gold is
**minimum error rate training (MERT)** — the standard machinery of classical
MT, or equivalently one logistic regression. The 50%→55%+ calibration gap is
a solved-problem-shape: ~30 lines with sklearn, data already labelled.

## 7. Loops and cycles (graph/homology, for the record)

The typed graph (≈ sound, = translation, ~ synonym) has a cycle space
(H₁ over GF(2)). Loop-certified pairs are edges lying on type-alternating
cycles — certification is membership in the cycle space with a parity
constraint. Systematic harvest = sparse linear algebra over GF(2) instead of
BFS sampling; would turn 896 certified atoms into the complete set.

## 8. Priority of transfer (effort × payoff)

1. **MERT/logistic channel calibration** (§6) — hours, closes a measured gap.
2. **WMD/Sinkhorn paragraph meaning** (§4) — one judge function, truer metric.
3. **Knuth–Bendix canonical forms** (§3) — big speedup + correctness of the
   rule system; unlocks paragraph-scale search.
4. **Submodular budgeted cover in set_dual** (§2) — guarantees at scale.
5. **Persistence-ranked tiers** (§1) — principled thresholds everywhere.
6. **Automaton-intersection decoding** (§3) — the GPU endgame, already in
   RUN_ON_GPU.md.
7. **GF(2) cycle harvest** (§7) — completes loop certification.

The one-sentence summary: *dual translation is a submodular cover over the
intersection of a metric relation and a Galois relation, decoded as a WFST
best-path, with existence guaranteed by the redundancy budget — every piece
has a classical theory, and none of the classical theories say no.*
