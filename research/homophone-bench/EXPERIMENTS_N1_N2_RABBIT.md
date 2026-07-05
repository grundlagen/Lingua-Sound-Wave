# Giving the ideas a go: three measured experiments (with self-critique)

Ran 2026-06-22, pure-Python env (panphon/numpy/wordfreq + espeak-ng). These take
three proposals from the planning docs and *test* them, scored honestly — the
project's own discipline (labels independent of scorers; report the negative).
Two of the three partly contradict what the planning docs claimed. Good.

Scripts: `rhythm_channel.py`, `cycle_consistency.py`, `rabbit_walk.py`.

---

## N1 — Cycle-consistency as a label-free signal: **confirmed, strong**

**Claim tested.** A true homophone round-trips: `en → fr → en'` should land back
near the start phonetically, and that round-trip fidelity — computed with **no
labels and no trusted scorer** — should agree with the hand-validated AUC-0.993
matcher. If so, it's a valid self-supervised training target (back-translation
for sound).

**Method.** 89 common English words → decode to French (FR trie) → re-G2P that
French → decode back to English (EN trie) → re-G2P. `cycle = nw_sim(ipa_en,
ipa_en')` is the label-free signal; `combo(en, fr)` is the trusted scorer, used
*only* to validate.

**Result.**

| metric | value |
|---|---|
| Pearson(cycle, combo) | **+0.701** |
| AUC: cycle ranks trusted-good (combo≥0.55) over trusted-weak | **0.872** |

Top round-trips are textbook (`she~chie`, `two~tout`, `said~cède`, all cycle
1.00); the broken ones self-flag (`with~oued~"announce"` cycle 0.76). **A
quantity that never sees a label or the matcher recovers the matcher's verdict at
0.872 AUC.** N1's premise holds.

**Self-critique (what a careful reviewer must add).**
- Cycle **saturates at 1.00** for many common short words — coarse at the top; it
  separates good-from-broken better than good-from-great.
- It can be **fooled by a consistently-biased decoder**: if both directions share
  the same systematic error, a degenerate cycle still closes. The signal is only
  as honest as the decoder's *independence* across directions — worth a guard
  (e.g., require the FR midpoint to also clear a fluency floor).
- It measures **sound** fidelity, not word identity (`two→tout→to`), which is
  correct for our purpose but means it can't be the *only* objective — pair it
  with the fluency/meaning terms.

**Verdict.** Build it. The right next step is `learn_costs.py` upgraded from
counting certified alignments to **minimizing mean cycle-distortion** over the
lexicon — a label-free objective for the equivalence/gap costs (and later the
WFST arc weights). This is the principled successor to the "promotion can't
compound" honest negative: growth from self-supervision, not re-labeling.

---

## N2 — A prosody/rhythm channel: **mechanically sound, but the benchmark can't show it**

**Claim tested.** Every channel scores segments; none scores rhythm. A metrical-
grid channel (syllable nuclei + stress prominence from espeak) should help on
multi-syllable **phrases** even if neutral on monosyllables.

**Method.** Extract a per-syllable prominence sequence (2/1/0) from espeak stress
marks; score = ½·syllable-count-agreement + ½·prominence-pattern-alignment.
Compare AUC of combo, rhythm-only, and blends on the 105-pair set.

**Result (after a bug fix — see below).**

| scorer | AUC | AUC_hard | AUC_phrase |
|---|---:|---:|---:|
| combo (baseline) | **0.994** | 0.994 | **1.000** |
| rhythm only | 0.722 | 0.700 | 0.597 |
| combo+rhythm avg | 0.951 | 0.945 | 0.902 |
| combo, rhythm-gate | 0.990 | 0.989 | 0.995 |

**The bug a phonetician catches immediately.** First pass scored rhythm-only at
0.60 because espeak emits diphthongs as two vowel symbols and the grid counted
`/eɪ/` as **two syllables** (`may day → [2,0,2,0]`). Merging adjacent vowels into
one nucleus fixed it (`may day → [2,2]`, rhythm 0.50→0.75) and lifted rhythm-only
to 0.722. *That* number is the real finding: **pure prosody, with zero segmental
information, separates homophones at 0.72 AUC** — rhythm genuinely carries the
signal the doc claimed.

**Why it still doesn't help the matcher.** combo is **saturated** on this set
(phrase AUC already 1.000); there is no headroom, and adding a noisier channel
only dilutes. The benchmark is monosyllable-heavy (only 22% of positives are
multi-word) and segmentally separable — the *wrong instrument* to prove a phrase-
rhythm idea, exactly as `DICTIONARY_FEASIBILITY.md` warned the AUC oversold an
easy task.

**Verdict.** Don't add rhythm to the matcher. Its place is **generation-time
reranking of long phrase↔phrase candidates**, where combo is *not* saturated
(the production demo's actual bottleneck). The channel is validated; the
benchmark just can't see it. Needs a harder, phrase-heavy eval set to score
properly — a concrete data TODO.

---

## Round Rabbit Fix 1 — best-product vs min-hop: **correct, but I overclaimed its impact**

**Claim tested (my own, from DEPS_RABBIT_AND_NOVEL.md).** `round_rabbit.py` keeps
the fewest-**hop** path and only ranks edge quality afterward, so best-**product**
Dijkstra (the WFST shortest-path) should "largely dissolve" the whole-word-
dominates-shallow problem.

**Method.** Rebuild a homophone sound-graph from dictionary-v5 directly (7,937
nodes, 8,236 edges), run both min-hop BFS and best-product Dijkstra from 200
well-connected seeds at 3 hops, and count where best-product strictly wins on the
same reachable node.

**Result.** Best-product beats min-hop on **only 0.3%** of reachable pairs (46 of
13,949), mean gain 0.079. Where it differs the win is real and clean
(`ct → cite`: 0.45 → 0.78, *same* hop count — BFS just took a weaker equal-length
route), but it is **rare**.

**Honest correction to my own roadmap.** The fix is *right* but **low-impact on
the current graph**, and the reason is graph topology: 8,236 edges over 7,937
nodes is **nearly a tree** — most nodes have a single path, so min-hop and best-
product trivially coincide. Best-product only matters where **multiple routes**
exist, i.e. in a **dense** graph. That points the lever elsewhere: the payoff
comes from **graph density** (the fragment-tunnel edges + the re-mining growth the
"honest negative" prescribed), and *then* best-product harvests it. The original
`round_rabbit` ran on the richer typed `mapping-web.json` (15,681 nodes; sound +
fragment + meaning + loanword + surface edges) where multiplicity is higher; my
sound-only reconstruction under-built density and so under-shows the fix — but
even there, density is the gating variable, not the path rule.

**Verdict.** Keep Fix 1 (it's free and exactly correct as a WFST shortest-path),
but **sequence it after density**: grow edges first (fragments + re-mining), then
best-product + typed-alternation + temperature (Fixes 2–4) pay off. I had the
priority backwards.

---

## What the three say together

- **Self-supervision is the real unlock** (N1): a label-free objective tracks the
  trusted scorer at 0.872 AUC — that is how this scales past hand labels and past
  the re-labeling dead end.
- **The benchmark is saturated** (N2): new *accuracy* ideas can't be judged on it;
  the project needs a harder, phrase-heavy, generation-oriented eval before
  claiming any matcher improvement.
- **Topology gates the graph methods** (Rabbit): path-quality rules only matter
  once the graph is dense; density (fragments + re-mining) is upstream of the
  walk policy.

Net: the highest-confidence build is **N1's cycle-consistency cost learner**; the
highest-value *missing tool* is a **phrase-level evaluation set** that can finally
score rhythm (N2) and dense-graph walks (Rabbit) on the regime that matters.
