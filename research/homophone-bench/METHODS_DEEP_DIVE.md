# Methods deep-dive: better engineering for avenues A–F

Companion to `ROADMAP_MULTILINGUAL.md`. The roadmap said *what* to build; this
says *how to build each one well*, thinking as a phonetician + TTS/ASR engineer +
ML coder rather than a list-maker. Each section states the limit of the naive
version, the stronger method, why it's stronger (grounded, not buzzwords), the
tooling **and a pure-Python fallback** (this env has only panphon/numpy/wordfreq;
heavy libs are pip-only), and the failure mode.

Two reframes unify almost everything below, so they lead:

> **Reframe 1 — the whole search is one WFST cascade.** Lexicon, pronunciation,
> phonetic equivalence, and *both* language models are finite-state objects.
> Composing them and taking k-shortest paths replaces every hand-rolled beam with
> an *exact, jointly-optimal* search. This is literally how Knight & Graehl did
> transliteration; the project gestures at them but still hand-codes the beam.
>
> **Reframe 2 — the interlingua is feature space, corrected by typology.** The
> universal pivot already exists: panphon's distinctive features. The per-pair
> EQUIV table is just a *correction* where a specific language *neutralizes* a
> contrast. Those corrections are derivable from phoneme-inventory typology
> (PHOIBLE) and loanword-adaptation rules — O(N) language descriptions, not
> O(N²) hand tables. **Cross-lingual homophony ≈ loanword phonology run in
> reverse**, and that literature is a free rule source.

Whisper/ASR is treated only as a *generator/miner* ("make more"), never a
validator, per instruction.

---

## A. Fluency LM — from one bigram to a calibrated two-tier scorer

**Limit of the naive plan.** "Swap KenLM 5-gram on subtitles" fixes corpus and
order but keeps three deeper problems: (1) the beam needs a *cheap* score but the
*best* fluency signal is expensive; (2) a word LM is blind to syntax on the sparse,
weird sequences homophone decoding emits ("but savent"); (3) `fluency()` maps
logprob→[0,1] with hand constants, so the beam weight isn't a real probability.

**Better method — three moves:**

1. **Two-tier scoring (first-pass/second-pass), the standard ASR/MT pattern.**
   Cheap **KenLM modified-Kneser-Ney** *inside* the beam for breadth and
   determinism; **neural masked-LM pseudo-perplexity** (Salazar et al. 2020) to
   *rescore the top-k* finished candidates only. You pay the transformer cost on
   ~50 survivors, not on every beam state. This is exactly first-pass n-gram +
   second-pass neural rescoring from production speech systems, transplanted.

2. **Factored / POS-aware LM in the beam** (Bilmes & Kirchhoff 2003). Score
   `P(word | prev_word, prev_POS)` or a tag-sequence LM alongside the word LM.
   Homophone decoding's failures are usually *syntactic* (a verb where a
   determiner belongs); a tag LM catches that even when the specific word bigram
   was never seen. Cheap, deterministic, and complementary to the word LM exactly
   like the two-channel matcher.

3. **Calibrate fluency to an actual probability.** Fit a logistic regression
   `P(fluent) = σ(a·logprob + b)` on a held-out set of *real phrases vs. shuffled
   phrases* (free labels: any corpus line is positive, its word-permutation is
   negative). Now the beam's `lm_weight` multiplies a calibrated probability, not
   a hand-tuned ramp — and the same fitter works per language, which matters for
   going multilingual.

   Plus: **character/morph n-gram backoff** for OOV. Lexique has 241k inflected
   forms; rare conjugations get zero word-bigram support and currently fall back
   to unigram wordfreq. A 4-gram *character* LM gives graded plausibility to
   unseen but well-formed French forms (morphology without a morphological
   analyzer).

**Tooling.** KenLM (`pip install kenlm` + the binary) on OPUS OpenSubtitles +
a Wikipedia/news slice per language; `transformers` distil-multilingual-BERT for
pseudo-PPL. **Fallback (this env):** a pure-NumPy interpolated modified-KN
trigram (a few hundred lines) recovers most of KenLM's quality; skip the neural
rescorer until torch is available — the factored LM + calibration alone should
move the frontier.

**Failure mode.** Subtitle text is dirty (SDH tags, OCR errors, code-switching);
clean before training. Pseudo-PPL is slow — keep it strictly top-k.

---

## B. Search — from a left-to-right beam to a WFST cascade with k-shortest paths

**Limit of the naive plan.** Even "anchored bidirectional" is a *heuristic*. The
real defect of the current decoder is that it beams left-to-right over **one**
phoneme stream and therefore **cannot jointly optimize sound + L1-fluency +
L2-fluency** — it decodes one side, then the other, then scores. Anchoring patches
the prior; it doesn't fix the objective.

**Better method — compose the transducers and search them exactly (Pynini).**
Build, once:

```
H  = L1 lexicon acceptor               (words → their IPA, weighted by LM_L1)
P1 = L1 pronunciation / variant FST     (schwa-drop, diphthong-smooth, liaison)
E  = phonetic-equivalence FST           (the EQUIV table as weighted arcs)
P2 = L2 pronunciation FST  (inverted)
G  = L2 lexicon acceptor    (IPA → words, weighted by LM_L2)

PAIRS = H ∘ P1 ∘ E ∘ P2⁻¹ ∘ G
best  = shortestpath(PAIRS, n=k)        # k best homophone pairs, jointly optimal
```

Intersecting with `LM_L1` on the input tape and `LM_L2` on the output tape makes
**both fluencies hard parts of the single shortest-path objective**, not a
post-filter. k-shortest gives controllable *diversity* for free; A* with an LM
heuristic gives speed; a **stochastic path sampler** (sample arcs ∝ weight) gives
*creative* output instead of only the argmax — the knob the project actually
wants for writing.

**Why it's strictly better.** (a) Optimality: shortest-path over the composition
is provably the best path under the combined weight; the hand beam is a greedy
approximation. (b) The equivalence table, variants, and liaison are *already*
finite-state in spirit — encoding them as real FST arcs removes the bespoke
`_sub`/`_variants` code and makes them composable and inspectable. (c) It is the
Knight-Graehl architecture the repo cites but stops short of. (d) Bidirectional
"anchoring" falls out as just constraining the input or output tape to contain a
chosen word.

**Tooling.** Pynini (Gorman 2016; OpenFst-backed, Python, *built for exactly this*
G2P/transliteration grammar work). **Fallback (this env):** a hand-coded
**bidirectional/bigram-Viterbi over the trie** captures ~80% of the win without
OpenFst — keep the current beam but (i) score both LMs in the *same* state tuple
(already started this session) and (ii) add the anchored mode. Treat full Pynini
as the upgrade once installable.

**Failure mode.** Naive composition blows up in states; needs on-the-fly
composition + pruning (Pynini supports both). Weight semiring choice (tropical for
shortest-path vs log for sampling) must match the goal.

---

## C. The regime-3 LLM loop — from "propose/judge" to WFST-constrained decoding

**Limit of the naive plan.** Asking an LLM to "propose lines from this vocabulary"
*and obey a phonetic constraint* fails twice: it drifts off the vocabulary, and it
cannot feel whether an L1 line even *has* a fluent L2 homophone until after it's
written. Proposal and constraint are separated, so most proposals are dead.

**Better method — fuse them: the WFST from B *constrains the LLM's decoding*.**
At each LLM generation step, run forward-backward over `PAIRS` (Reframe 1) to find
which next-tokens keep at least one complete homophone path alive, and **bias the
LLM's logits** toward exactly those. The LLM supplies fluency + meaning; the WFST
supplies the *hard* sound constraint, token by token. The model literally cannot
emit an L1 continuation with no viable L2 sound-partner.

This is **grammar-constrained / logit-biased decoding** (the same machinery as
JSON-mode / GBNF / `outlines` / `llguidance`), with the "grammar" being the
phonetic transducer instead of a syntax. It is the genuinely state-of-the-art
move and it *enforces* invariant #2 (the LLM never scores sound — the FST does;
the LLM only chooses among sound-legal tokens by meaning/fluency).

**Two more refinements:**
- **Free L2 fluency from the same model.** A multilingual LLM scores French
  natively, so the L2 reading's own sequence-logprob *is* the LM_L2 signal —
  reserve explicit API calls for the meaning gloss only (the anti-hallucination
  `fr_gloss` trick from `LLM_RECIPE.md`).
- **Best-first, not "mutate."** Frame propose→score→keep as best-first search /
  light MCTS with `score = λ1·LM_L1 + λ2·LM_L2 − φ·cost + μ·meaning` as the value
  function. Cleaner and more sample-efficient than ad-hoc mutation.

**Tooling.** A local instruction model via `llama.cpp`/`vllm` with a logit-bias
hook, or hosted models with a `logit_bias` parameter (coarser, per-token). The
forward-backward over `PAIRS` reuses B's FST. **Fallback (this env):** *re-ranking*
constrained decoding — let the LLM propose freely, then **filter+rescore** each
proposal through the deterministic decoder + arbiter (the loop already sketched).
Weaker than logit-bias but needs no local-model plumbing; it's the bridge until
B's FST and a biasable model are in place.

**Failure mode.** Per-step FST forward-backward is the compute hot spot — cache
the FST's reachability and bias only the top-N LLM logits. Constraint too tight →
stilted L1; loosen by admitting B_safe vocabulary and free function words.

---

## D. Sound+meaning — from synonym-bridge to a learned joint embedding index

**Limit of the naive plan.** LLM synonym-bridging is good but *ad hoc and per-pair*
— it can only upgrade pairs you already hand it. It doesn't *mine* the space.

**Better method — make sound+meaning a searchable joint space.**

1. **Two indexes, intersect them.** A **phonetic-neighbor index** (LSH/FAISS over
   either phoneme-bigram bit-vectors or panphon feature sequences) and a
   **semantic index** (multilingual sentence embeddings — LaBSE / multilingual-E5,
   which already co-embed EN+FR). A sound+meaning hit is any item lying in *both*
   the phonetic ball and the semantic ball around a query. This mines the **whole
   lexicon** with no per-pair LLM call — the LLM is then spent only on the audit
   queue where the two indexes disagree (the dual-judge pattern again).

2. **Learn the metric.** Train a small projection so distance =
   `α·sound + β·meaning`, contrastively, on the **1,811 already-confirmed pairs**
   as positives. This is **PWESuite-style phonetic word embeddings fused with a
   multilingual semantic encoder**: nearest-neighbor in the learned space *directly*
   returns new sound+meaning pairs ranked by exactly the blend the project wants.
   The graded bands in `sound_meaning.py` become a trained retrieval model instead
   of a post-hoc filter.

**Why it's better.** It converts a manual upgrade step into *retrieval*, scales to
any lexicon size, and — crucially — generalizes to every language pair the moment
their words share the multilingual embedding space (they do).

**Tooling.** `sentence-transformers` (LaBSE), `faiss`. **Fallback (this env):**
brute-force cosine in NumPy over the 5,596 usable pairs is trivially fast; the
phonetic index can be exact Dice over the existing bigram sets. Even without a
learned metric, the *intersect-two-rankings* mining works today in pure Python.

**Failure mode.** MiniLM-grade encoders are noisy on 1-syllable words
(`awed~odes`) — keep the LLM gloss as the tiebreaker exactly where embeddings are
unreliable (short items), which is precisely `LLM_RECIPE.md`'s point.

---

## E. Multilingual interlingua — from hand EQUIV tables to typology-derived floors

**Limit of the naive plan.** "Map each language into a shared space and learn the
floors from labeled alignments" still needs labeled data per pair and a curated
table. With 132 espeak voices already available, hand-curation is the bottleneck,
not phonemization.

**Better method — derive the equivalence floor from phonological typology.**

1. **Feature space is the interlingua (Reframe 2).** Don't pivot through a
   *language*; pivot through panphon features. Triangulating through, say, English
   adds English's neutralizations as noise. Feature space is neutral and already
   universal — it's *why* the matcher ports across espeak voices unchanged.

2. **Generate the per-pair floor from inventory gaps (PHOIBLE).** A floor exists
   precisely where a language *lacks a contrast*: French has no /h/ → /h/ is
   free-deletable against French; Japanese merges /l/–/r/ → that floor ≈ 0 for any
   Japanese pair; Spanish lacks /ʒ/–/dʒ/ distinctions in many dialects → floor
   them. **PHOIBLE** (phoneme inventories for 3000+ languages) lets you *compute*
   these from each language's inventory, automatically — O(N) inventories, not
   O(N²) tables, and grounded in real typology rather than a curator's ear.

3. **Loanword-adaptation rules as a free transducer (the deep insight).**
   *Cross-lingual homophone substitution is loanword phonology in reverse.* When
   Japanese borrows "strike" → /sutoraiku/, the epenthesis + /r/-substitution +
   final-vowel rules ARE the legal sound-correspondences for EN↔JA homophony. This
   literature is documented per language and converts directly into weighted FST
   arcs for `E` in the cascade (B). It gives principled, language-specific
   correspondences — including epenthesis and deletion — that a flat feature
   distance misses.

**Why it's better.** It makes "add language N" a matter of *describing N once*
(inventory + a handful of adaptation rules), not labeling N×(existing) pairs. That
is the actual difference between a bilingual demo and a multilingual generator.

**Tooling.** PHOIBLE CSV (static data), `panphon`/`epitran` for G2P across many
scripts. **Fallback (this env):** panphon is already here; bootstrap the first
extra language (Spanish: clean orthography, big WikiPron) with a *derived* floor
(inventory-gap rules from a small hand list) and confirm the matcher holds AUC on
a 50-pair ES↔EN set *without* a bespoke EQUIV table. If it holds, the typology
route is validated cheaply.

**Failure mode.** Suprasegmentals (tone, vowel harmony, pitch accent) aren't in a
segmental feature table — keep a per-language override slot for those, and flag
tonal-language pairs as needing extra modeling.

---

## F. ASR/TTS as a *generator*, not a validator ("make more")

Per instruction, no Whisper-as-judge. But speech tech is a strong **source**:

1. **Mine homophones from real-speech ASR confusion.** RESULTS.md killed
   Allosaurus *on synthetic speech*. The fix is real human audio: force-decode L1
   recordings (Common Voice, multilingual read-speech) against an L2 lexicon/LM;
   high-confidence L2 outputs that back-align to the L1 audio are homophone pairs
   discovered from acoustics, free of espeak G2P error. ASR's language-model
   hallucination — a *bug* for validation — is here a *feature*: it surfaces the
   sound-legal L2 strings.

2. **Forced alignment → acoustically-grounded fragments.** Montreal Forced
   Aligner / CTC alignment on real bilingual speech yields phone-level timestamps;
   mining the aligned phone sequences enriches the **fragment index** (`fragments.tsv`)
   with blocks proven in real audio, not only espeak-derived ones.

3. **Neural multi-voice TTS for variant coverage (the literal "make more").**
   espeak gives one robotic pronunciation; modern neural TTS (VITS/XTTS, many
   accents) generates pronunciation *variants* — the pun that works in a Marseille
   vs. Parisian accent. Feeding those variants into `_variants()` widens
   equivalence coverage and stress-tests robustness across accents.

4. **Self-supervised discrete speech units as an acoustic interlingua.**
   HuBERT/wav2vec2 discrete units are language-agnostic by construction
   (textless-NLP line). They offer a second, *acoustic* pivot to cross-check the
   symbolic feature pivot — and to reach languages with poor G2P/orthography.

**Tooling.** Common Voice (data), MFA, a neural TTS, fairseq/HF SSL models — all
heavyweight. **Fallback (this env):** none of the above installs cleanly here;
treat F as a data-pipeline track for a richer environment. The cheap slice usable
today is **espeak multi-voice variant generation** (132 voices already present) to
expand `_variants()` coverage — a real, zero-dependency "make more" win.

---

## What this changes about the sequence

The deep-dive collapses several roadmap items into **two foundational builds** that
the rest hang off:

1. **The WFST cascade (B)** — once `E` is an FST, it carries the typology-derived
   floors (E), is what the LLM is biased against (C), and is the exact search the
   anchored heuristic only approximated. Build this and A's LM simply becomes the
   tape weights.
2. **The calibrated two-tier LM (A)** — the in-beam/in-tape fluency for *every*
   language, and the thing that finally moves the measured phrase↔phrase frontier.

D (joint embedding mining) and F (espeak multi-voice variants) are independent,
pure-Python-today wins to run in parallel. The neural pieces (pseudo-PPL rescorer,
logit-biased LLM, LaBSE, SSL units) are upgrades that slot into these two
foundations once the environment carries torch/pynini/faiss.

One-line thesis: **make the search a transducer and the fluency a calibrated LM;
then the LLM is something you *constrain with the transducer* rather than trust,
the interlingua is *feature space corrected by typology* rather than hand tables,
and speech models are how you *mine more*, never how you judge.**
