# The central problem: accuracy on BOTH sides — prior-art audit + what matters

Before proposing more, I audited the repo for what already exists, then used the
*measured* numbers already committed here to state the real bottleneck precisely.
Short version: **sound accuracy is solved; meaning accuracy under the sound
constraint is the wall, it is already quantified (~0.41→0.66 cosine, with output
that is sound-right but semantically incoherent), and the thing blocking progress
is not another generator — it is (a) the wrong objective and (b) a missing
passage-level evaluator + an L2-coherence model.**

---

## 1. Prior-art check: what I was about to propose already exists (mostly)

| I proposed | Already in repo? | Reality |
|---|---|---|
| cycle-consistency cost learner | **partly** | `learn_costs.py` already learns substitution/gap costs from certified alignments (Ristad–Yianilos count curve) and validates on the frozen benchmark. Its certified set already used a **meaning-preserving round trip** (loop-certification). My phonetic round-trip (`cycle_consistency.py`) is genuinely *different* (sound fidelity, not meaning return) — but it improves **sound**, which is already solved, so it is **not** the priority. |
| sound+meaning grading | **yes** | `sound_meaning.py` grades every usable pair by multilingual-embedding cosine into identical/close/related/unrelated bands. Word/short-pair semantic accuracy already exists as a column. |
| "homophonic AND meaning" production | **yes** | `dual_translate.py` decodes EN→FR homophones then re-ranks by embedding similarity to the literal translation (regime 1). |
| open both sides to beat the ceiling | **partly** | `paraphrase_translate.py` already opens the **English** side (synonym search) and *measured the ceiling*. |
| phrase/passage-level held-out eval | **no** | Does not exist. `bench.py`/`dataset.py` are 105 **word/short-pair** classification items with a word-level hash split. **This is the real gap.** |
| regime 3 (both sides free, LLM loop) | **no** (designed only) | `REPRESENTATION.md` §3 + `LLM_RECIPE.md` specify it; unbuilt. |

So the word-level and sound-level toolkit is largely built. The two missing
things are a **passage-level evaluator** and the **both-free generation loop** —
and, underneath both, a model that can judge **L2 coherence**.

## 2. The biggest problem, stated from the repo's own measurements

Homophonic translation imposes three constraints at once:

1. `sound(L1_text) ≈ sound(L2_text)` — phonetic fidelity,
2. `L2_text` parses as fluent target words,
3. `L2_text` *means* something close to `L1_text`.

The repo has already measured what happens when you demand all three:

- `dual_translate` ("the sea is cold" → **"laissées colt"**): sound 0.90, **sem
  0.49**, and the French is *semantic nonsense* ("dropped colt").
- `paraphrase_translate`, even after opening the English side: baseline sem ~0.41
  → best **~0.66** ("the sea is chill" → "dessus gel" = "above frost"), gains
  +0.25/+0.29, and *one sentence gained +0.00*. At 0.66 the French is still
  evocative fragments, **not coherent prose**.

**That is the wall.** The three constraints **over-determine** the output: the set
of phoneme streams that are simultaneously sound-faithful AND parse as fluent L2
is already tiny; demanding they also *mean* the L1 nearly empties it. You get
sound ≈ 0.9 and meaning ≈ 0.5-of-nonsense. Sound is solved; **meaning-under-sound
is the problem, and it is a problem of over-constraint, not of a weak scorer.**

## 3. The reframe that matters most: coherence, not equivalence

The art form already tells us the resolution. Van Rooten's *Mots d'Heures*
(in `dataset.py` as the loose gold) does **not** preserve meaning — "Humpty
Dumpty" ⇄ "Un petit d'un petit" means something *different and surreal*. The
French is **grammatical and evocative**, not a translation. The pleasure is the
double-reading, not semantic fidelity.

So "accurate for both" should be redefined, because the strong version (same
meaning both sides) is provably over-determined:

> **Wrong target:** `L2 means the same as L1` (collapses to ~0.5-nonsense).
> **Right target:** `sound ≈` **AND** `L1 independently coherent` **AND** `L2
> independently coherent` **AND** (optionally) `L1, L2 share THEME/MOOD`.

"laissées colt" fails not because it isn't "the sea is cold" but because it
**means nothing**. The achievable, valuable accuracy is **bilateral coherence +
global theme**, measured at the **passage** level (where local lines are free but
the whole reads as *about the sea, and melancholy* in both languages). This is
both honest to the craft and escapable from the 0.49 ceiling.

## 4. Therefore, what is actually important (the impetus)

In priority order — note none of these is "a better homophone generator."

### I. An L2-coherence model is the single most important missing component.
The same gap shows up everywhere: the production demo's bottleneck (FR phrase
fluency), the dual-translate nonsense ("laissées colt"), the bigram LM's blindness
("set could" looked fine). **There is no model here that distinguishes a coherent
French image from word-salad.** A word-bigram cannot; mean-zipf cannot. This is
the thing to build/borrow: a real L2 language model (KenLM 5-gram at minimum, an
actual LLM ideally) used as a **coherence judge**, not a fluency nudge. Every
downstream goal is gated on it.

### II. A passage-level, held-out sound+meaning evaluation set.
Right now "accurate for both" is **unfalsifiable** above the word level — there is
no eval that scores (sound fidelity, L1 coherence, L2 coherence, theme-match) on
whole passages. My own two experiments (rhythm, dense-graph walks) *both* dead-
ended on exactly this: the word benchmark is saturated and can't see phrase-level
gains. **Build this first**, because nothing above the word can be optimized or
even claimed without it. ~50 passages: source line, a human-grade L2 coherence
0–5, a theme tag, and the sound score — frozen, never tuned on.

### III. Co-composition, not translation (open both sides + choose the content).
The escape from over-constraint is *degrees of freedom*. `paraphrase_translate`
opened one side and gained +0.25; opening **both** (regime 3) is the real move —
neither side anchored to a given meaning, the system **searches for a meeting
point** where both happen to be coherent. And crucially, **let the writer choose
content the languages are phonetically in tune on** (the density-map idea): you
don't translate a fixed sentence, you co-generate the message under a shared sound
skeleton. This is where an LLM belongs — proposing coherent lines in *both*
languages while the deterministic decoder owns sound (the iron rule).

### IV. Optimize meaning at the discourse level, not the word.
Give the search **local freedom, global constraint**: score theme/mood on the
whole passage embedding, not line-by-line cosine. A poem can be "accurate for
both" as *two coherent poems on one theme* even when no single line is a literal
translation — and that target is reachable where line-level equivalence is not.

## 5. Honest correction to my earlier priority

My last writeup said the highest-confidence build was the cycle-consistency cost
learner. After this audit I'd **demote it**: it sharpens **sound**, which is
already at AUC 0.993 and ~0.9 on outputs, so it polishes the part that isn't
broken. The real ladder is **II → I → III → IV**: build the passage evaluator so
you can *see* meaning accuracy, add the L2-coherence model so you can *score* it,
open both sides + pick content so you can *reach* it, and judge theme at the
passage level so the target is the achievable one. The central problem isn't
making the French sound right — it already does. It's making the French *mean
something*, on its own terms, while it does.
