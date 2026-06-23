# Unrestricted dependencies, Round Rabbit, and novel directions

Third planning doc (companion to `ROADMAP_MULTILINGUAL.md` and
`METHODS_DEEP_DIVE.md`). Assumes the same machine but **no install/network
restriction**. Three parts as asked: (1) where to actually get the dependencies
and data, curated by someone who builds TTS/G2P systems; (2) a real read of the
**Round Rabbit** hopping generator with concrete fixes; (3) novel ideas of my own
after ingesting the whole project. Still planning, not production.

---

## Part 1 — Where to get the extra dependencies (expert sourcing)

The point isn't "pip install everything." It's knowing which **two or three
assets change the ceiling** versus which are commodity. Tiered by leverage.

### Tier 0 — the two that make it *planet-scale* multilingual
These are the strategic unlocks; everything in `ROADMAP`'s avenue E assumes them.

- **Meta MMS (Massively Multilingual Speech)** — TTS *and* ASR for **1,100+
  languages**, plus a 4,000-language LID. One model family gives you the
  pronunciation + speech-mining substrate for nearly every language on Earth.
  This is what turns "EN↔FR plus heroics" into "any pair." HF:
  `facebook/mms-tts-*`, `facebook/mms-1b-all`.
- **PanLex** — a translation graph across **~5,700 languages** (free DB dump).
  This is the *meaning* layer at the same scale as MMS is the *sound* layer.
  MUSE (current) covers a handful of pairs; PanLex covers the long tail and lets
  the sound+meaning lexicon (the rare, valuable one) exist for under-resourced
  languages. Pair it with **WOLD** (World Loanword Database) for the
  loanword-adaptation rules that avenue E wants as FST arcs.

### Tier 1 — pronunciation (G2P), the real bottleneck per new language
- **CharsiuG2P** — ByT5 neural G2P, **100 languages**, one model. The modern
  replacement for espeak's per-language hand rules; outputs IPA directly.
- **Epitran** (Mortensen/CMU) — rule-based G2P for ~100 languages, **panphon-
  native** (feeds the matcher's feature vectors with zero glue code). Best for
  clean-orthography languages (ES, IT, TR, …).
- **WikiPron** — 1.7M+ pronunciations scraped from Wiktionary across **165
  languages**; plus **wiktextract** to pull the full Wiktionary dump yourself
  (pronunciations, translations, senses — three layers at once).
- Keep **espeak-ng** (here, 132 voices) as the universal fallback, **Lexique**
  (FR) and **CMUdict / WikiPron-EN** as the high-quality anchors.

### Tier 2 — language models (avenue A fluency)
- **KenLM** (Heafield) — the n-gram engine; modified Kneser-Ney, fast binaries.
- Corpora, in order of usefulness for *n-gram fluency*: **Leipzig Corpora
  Collection** (clean, sentence-tokenized, per-language, purpose-built for exactly
  this — start here), **OPUS OpenSubtitles** (colloquial register the literary
  corpus lacks), **CC-100 / OSCAR / mC4** (web scale when you need coverage).
- Neural rescorer (the second-pass): **distil-mBERT** or **XLM-R** for
  pseudo-perplexity; a small **multilingual GPT** if you want generative fluency.

### Tier 3 — meaning embeddings (avenues C/D)
- **LaBSE** (109 languages, translation-aligned — the default), **multilingual-
  E5**, **LASER3** (200 languages incl. low-resource). All via
  `sentence-transformers`. These already co-embed every language, so the
  sound+meaning index generalizes for free.
- Phonetic embeddings: **PWESuite** (phonetic word embeddings + benchmark) to
  fuse with the above into the joint α·sound+β·meaning space.

### Tier 4 — the transducer + speech tooling
- **Pynini + OpenFst + Thrax** (Gorman/Google) — the WFST cascade of
  `METHODS_DEEP_DIVE` avenue B. Pip-installable; this is the single most
  architecture-changing library to add.
- **Montreal Forced Aligner** / **WhisperX** / **ctc-segmentation** — phone-level
  alignment to mine acoustically-grounded fragments.
- **Common Voice, FLEURS (102 langs), VoxPopuli, MLS** — real human speech for
  the ASR-confusion miner (the fix for the Allosaurus-on-TTS collapse).
- **PHOIBLE** (3,000+ inventories, static CSV) — the typology that *generates*
  the equivalence floors.
- TTS for accent variants: **Coqui XTTS / Piper** alongside MMS-TTS.

### The honest install note
`torch`/`transformers`/`pynini`/`faiss`/`kenlm` are all pip-installable but heavy;
PanLex/PHOIBLE/WikiPron/Leipzig are data downloads, not packages. The right move
in an unrestricted env is a **`make data` step** that fetches PHOIBLE + Leipzig +
WikiPron + PanLex once into `data/`, and a `requirements-heavy.txt` that's
separate from the current pure-Python core — so the AUC-0.993 matcher keeps
running anywhere while the multilingual stack is opt-in.

---

## Part 2 — Round Rabbit, the hopping generator

**What it is.** `round_rabbit.py` builds a *semantic→sound lattice*: collapse
meaning edges into semantic components (union-find), then from each component
**BFS-walk the sound graph outward**, recording how many homophonic hops away
each node sits, and attach every v5 string at each node. With `--with-fragments`
it tunnels through subword chunks (`en:word → sound:en:chunk → [fragment] →
sound:fr:chunk → fr:word`), so fragment-built routes become first-class paths
(reachable nodes 85→360 at 3 hops). The output isn't a poem — it's the **lattice
a generator hops through**: "start at this meaning, here's everything 0/1/2
homophonic hops away, with all bilingual strings attached."

It's a genuinely good idea — a *homophonic semantic field*. But the
implementation leaves real value on the table:

**Fix 1 — walk the best-*product* path, not the fewest-*hops* path.**
`bfs_component` keeps `best[nxt]` by minimum hop count and only *afterwards*
folds `edge_mean` into the rank. So it can commit to a weak 2-hop route over a
strong 3-hop one, then dock it with `−0.08·hops`. Replace the BFS with **Dijkstra
in the tropical (or log) semiring** over `−log(edge_score)`: the path cost becomes
the *product of edge confidences*, which is the right quantity and is exactly the
WFST shortest-path of `METHODS_DEEP_DIVE` avenue B. The honest "whole-word
dominates at shallow hops" problem largely dissolves, because a deep fragment
tunnel of strong edges can now out-score a shallow whole-word hop of weak ones —
which is the whole point of the fragment patch.

**Fix 2 — make the hops *typed and alternating*.** Right now meaning collapses
into a component and then the walk is pure sound. The more expressive object
alternates edge types: `sound → meaning → sound`. A walk constrained to use
**≥1 sound and ≥1 meaning hop** produces the actual target — "means something
near the seed, *sounds* like something else" — and the typed web already carries
`sound/meaning/fragment/loanword/surface` edge labels to do it. This is the
`chain_translate` alternation idea promoted into the lattice.

**Fix 3 — give the rabbit a temperature.** The lattice is built by argmax BFS and
ranked deterministically. For *writing*, you want diversity: sample the next hop
**∝ edge_weight^(1/T)** (a biased random walk, node2vec-style but over typed
edges). T→0 recovers today's behavior; T>0 gives many different fluent poems from
one seed meaning. This is the cheap creativity knob the generator lacks.

**Fix 4 — close the rabbit's loop (it's named for this).** The chain-web already
found **539 loops** — walks that return to the seed's own semantic neighborhood.
Those are the self-reinforcing pun families: a phrase whose homophonic image
*also means something near where it started*. Rank and surface **closed loops**
first; they are the densest gold and exactly what "Round" should mean.

Reframed: Round Rabbit is a **generative thesaurus across sound and sense** —
query a meaning, get a cloud of ways to say nearby meanings that sound like other
things. With Fixes 1–4 it becomes the front-end the writing tool should sit on.

---

## Part 3 — Novel directions (mine, after ingesting everything)

Ordered by how much I'd bet on them. These are *new* angles, not A–F restated.

### N1. Cycle-consistency as a free training signal ★ the one I'd build first
Translate EN→FR homophonically, then take that FR and translate it FR→EN
homophonically. A *true* homophone round-trips: the phoneme stream you recover
should match the one you started from. So **reconstruction error is a label-free
loss** — back-translation / CycleGAN logic applied to *sound*. Today
`learn_costs.py` tightens equivalence costs by *counting* certified alignments;
replace that with a **differentiable cycle loss**: adjust the EQUIV/gap costs (and
later the WFST arc weights) to minimize round-trip phonetic distortion over the
whole lexicon. No human labels, it scales to every language pair MMS/PanLex
reach, and it directly optimizes the thing we care about (sound preserved both
ways). This is the principled successor to the "honest negative" — growth from
*self-supervision*, not re-labeling.

### N2. Prosody/rhythm as a matching channel ★ candidate for the next "sharpen"
Every method here scores *segments*; **none scores rhythm**. But phrase homophony
is carried by stress and syllable timing — "a name"/"an aim", "ice cream"/"I
scream", and the whole Van Rooten *Mots d'Heures* effect, are **metrical** puns,
not segmental ones. Add a channel that aligns the two phrases as **metrical grids**
(stressed/unstressed beats, syllable weight, and — grounded by TTS — actual
durations and an F0 contour). Score rhythm-match alongside ngram+feat. Given the
"sharpen" trick was *one* added dimension worth +0.05 AUC, I'd wager rhythm is the
most underexploited remaining axis, especially for *phrase*↔*phrase* where the
project is currently stuck. WikiPron carries EN stress; FR is phrase-final; TTS
gives the rest.

### N3. The oronym / juncture engine — resegmentation as first-class search
The richest homophones are **oronyms**: identical phoneme streams cut into words
*differently* in each language ("un petit d'un petit" ⇄ "Humpty Dumpty"). The
system half-has this (fragments + the decoder), but it isn't the explicit object.
Build a **parallel lattice parse**: fix one phoneme stream, enumerate *all*
segmentations into BOTH languages simultaneously (the WFST does this exactly), and
rank by joint fluency. This makes juncture/liaison — already in the data as
`onset/coda` fields and liaison arcs — the *driver* of generation, not a
post-hoc fixup. It's the structural heart of the art form.

### N4. Phonosemantic steering — let sound symbolism break ties
When several FR candidates are sound-equal, prefer the one whose **phonaesthetics
fit the meaning**: /gl-/ for light (glow/gleam/glisten), bouba/kiki roundness,
nasals for softness. A small learned phonosemantic prior (trainable from
sound-symbolism datasets and from the embedding space itself) adds a "feels right"
nudge. Subtle, cheap, and uniquely on-theme for a system whose whole premise is
that sound and sense rhyme.

### N5. The interactive homophonic writing IDE ★ the highest-value *product* shape
The iron lesson of this project is that **the machine owns sound and the human
owns sense**. So stop batch-generating finished poems and instead build a
**co-writing surface**: the author types the EN line; the system live-renders the
FR phonetic shadow + its gloss, highlights where the FR breaks fluency (red), and
offers sound-legal next-word completions (the WFST forward-pass = "homophonic
autocomplete"). The deterministic decoder makes it real-time; the human supplies
the judgment the LLM can't be trusted with. This is where the art actually lives,
and it's a far better fit than "press button, get poem."

### N6. A homophone-density map (where the languages are "in tune")
Some semantic fields and some language pairs are homophonically rich, others
barren. Precompute, per pair, a **density map over meaning space**: which topics
have the most cross-lingual sound-cover. A writer (or the generator) then chooses
a theme the two languages *rhyme on*. It's a novel analytic deliverable, it guides
N5's suggestions, and it's computed straight off the existing giant component +
the embedding space.

### N7. Self-supervised embedding of the giant component
The chain-web is one 193k-node component. Run **typed node2vec** over it to learn
a continuous space where homophonic neighbors are close. Three payoffs: O(1) kNN
homophone retrieval across the whole lexicon (replaces blocking), a *learned*
stand-in for the hand EQUIV table (the graph already encodes the corrections), and
embeddings that feed N1's cycle loss and N4's phonosemantics. The graph is already
built; this is reading structure already paid for.

---

### If I had to pick the through-line
**N1 (cycle-consistency) gives label-free learning, N2 (rhythm) gives the missing
accuracy axis, N3 (oronyms) gives the right generative object, and N5 (the IDE)
gives the right product** — and all four ride the same WFST cascade + multilingual
G2P/LM/embedding stack that PanLex + MMS + Pynini + Leipzig make possible. The
rabbit, fixed (best-product, typed, temperature, loop-closing), is the lattice the
IDE hops through.
