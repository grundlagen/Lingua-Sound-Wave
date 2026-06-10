# Are the results real, and can we build the dictionary?

Follow-up to RESULTS.md, answering the right question: the 0.993 AUC looked
suspicious, so I stopped trusting the summary and looked at the actual scored
pairs and ran the dictionary-building task directly. Run 2026-06-10.

## TL;DR

- **The judge is not dodgy, but the AUC oversold it.** Classifying curated
  pairs is a far easier task than building a dictionary. On the real task —
  searching a whole lexicon — the matcher genuinely surfaces legitimate
  word-to-word homophones at the top, but a fixed score threshold is
  useless (≈82 candidates clear 0.45 per query, almost all noise).
- **A real dictionary is feasible.** From the 561 most common English words,
  blocking+ranking against 3,950 French words yields **27 gold (S) + 53
  strong (A) = 80 legitimate entries — ~14 per 100 English words.**
  Extrapolated, a few hundred to ~1,000 strong EN↔FR entries is realistic,
  and it multiplies across language pairs.
- **You were right about "un œuf / enough".** Restricting candidates to real
  single words structurally excludes forced phrase puns, so the dictionary
  fills with `set/cette`, `west/ouest`, `rest/reste` (both real words), not
  `un œuf/enough` (a grammatical-but-forced concatenation). Phrase puns are a
  separate, lower tier that needs a human.
- **"Shared sound AND meaning" is a much smaller lexicon than "shared
  sound".** Only **3 of 80** strong entries are also cognates
  (`group/groupe`, `form/forme`, `rest/reste`). Sound-alikes are abundant;
  sound-AND-meaning matches are rare (~4%).

## Is the scoring legitimate? Looking at the actual pairs

The classification negatives that scored highest are **not judge failures —
they are genuine near-misses my own labels got wrong**:

| pair | combo | what it actually is |
|---|---:|---|
| fire / feu | 0.554 | shared /f/ onset + both short — a real (mild) sound overlap |
| garden / jardin | 0.507 | **cognates**; /ʒaʁdɛ̃/ genuinely echoes /gɑːrdən/ |
| chair / chaise | 0.484 | **cognates**, shared /ʃ…/ onset + vowel |
| moon / lune | 0.496 | shared /n/ coda + high rounded vowel |
| night / nuit | 0.469 | **cognates**, /n…i/ skeleton |

So the headline AUC is slightly inflated two ways: (1) most translation
negatives (dog/chien) sound nothing alike and are trivially separated,
padding the score; (2) the handful of hard negatives that *do* score high are
cognates I mislabeled as negatives — the matcher was right, my labels were
wrong. **Lesson: the matcher is sound; the benchmark measured an easy task.**
Only one positive was missed — `dough/dos` (0.341), a real failure on very
short diphthong words (see below).

## The real test: retrieval, not classification

`retrieval.py` ranks the whole French lexicon for each English query. `combo`
puts the documented ideal at **rank 0 for 8/20 queries**, with most of the
rest in the top 5 or simply absent from the frequency list. But the decisive
finding is the **noise floor**: an average of **82 French words clear the
0.45 classification threshold per query**. A dictionary built by thresholding
would be ~98% garbage. Dictionaries need **top-k retrieval with a high bar**,
not a classifier threshold.

## Building it at scale

`build_dictionary.py` is the right architecture and the feasibility proof:

1. phonemize real lexicons once (frequency lists ⇒ every candidate is a
   legitimate lexical unit — this is the "no un œuf" guarantee);
2. **block** with cheap phoneme-bigram Dice (fast set ops) to ~25 candidates;
3. **rank** the shortlist with full `combo`;
4. keep the best per word, bucketed by quality.

Result from 561 EN × 3,950 FR (`dictionary-sample.json`, 358 entries):

| tier | bar | count | character |
|---|---|---:|---|
| **S** | ≥ 0.90 | 27 | gold: `to/tout`, `for/fort`, `see/si`, `set/cette`, `west/ouest`, `rest/reste`, `true/trou`, `less/laisse`, `men/mène`, `form/forme`, `group/groupe` |
| **A** | 0.78–0.90 | 53 | strong: `shows/choses`, `like/lac`, `guys/gaz`, `no/nos`, `by/bas` |
| **B** | 0.62–0.78 | 278 | loose/punning — needs a human (this is where `un œuf/enough` lives) |

**Yield: 14.3 S+A entries per 100 English words.** That is a genuinely usable
dictionary, and it scales: bigger lexicons + more language pairs (the matcher
is language-agnostic via espeak voices) multiply it. The S-tier entries are
exactly the "full laurels" cases — both sides real, common, near-identical in
sound.

## The three lexicons are different sizes

Your phrase "shared meaning and sound" names a specific, rare thing. The data
separates three populations cleanly:

1. **Shared sound** (homophones): abundant. ~14 strong / 100 words.
   `set/cette`, `two/tout` — sound alike, unrelated meaning. This is the big
   dictionary.
2. **Shared sound AND meaning** (cognate-homophones): rare — **3 of 80**
   strong entries (`group/groupe`, `form/forme`, `rest/reste`). A real but
   small lexicon; most cognates have *drifted* in sound (nation/nation sound
   different) and most homophones are meaning-unrelated coincidences.
3. **Forced phrase puns** (`un œuf/enough`): excluded by construction here;
   they need a curated phrase list and human judgement, and they never earn
   S-tier because one side isn't a single legitimate word.

If the goal is a lexicon of **shared sound** for wordplay/learning, it's
clearly feasible at scale. If the goal is **shared sound + meaning** (true
cognate-homophones, the linguistically "glorious" set), expect it to be
small — dozens per language pair, not hundreds — and worth curating by hand
from the S-tier candidates this pipeline surfaces.

## Known failure modes (for the next iteration)

- **Short diphthong words**: `dough/dos`, `low/l'eau` — espeak emits EN
  diphthongs (/oʊ/) that don't align to FR monophthongs (/o/) on 2-segment
  words where one mismatch dominates. The diphthong-smoothing variant helps
  but should be weighted higher for short words.
- **Nasal vowels**: `ant/an`, `pan/paon` — FR nasal vs EN VN handling still
  loses real matches. The nasal-split variant needs to also try dropping the
  nasal consonant.
- **No frequency/legitimacy prior in the score**: the matcher ranks `more`
  → `morts`/`mort`/`mots`/`mot` all ≈0.72; picking the best *dictionary*
  entry among sound-equals needs a frequency tiebreak (added as the candidate
  ordering, not yet in the score).

## Bottom line

The method is real and the dictionary is buildable — but as a **top-k
retrieval problem with a high score bar and real-word candidates**, not as
the classifier the AUC measured. Expect a large shared-sound dictionary and a
small shared-sound-and-meaning one. `dictionary-sample.json` is a real first
slice; the S-tier is ready to use as-is.
