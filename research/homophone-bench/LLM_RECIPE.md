# Getting an LLM to do sound+meaning matching (and why the freehand way fails)

## Verdict on the codex branch (`codex/v5-composition-mapping-web`)

Not dodgy in execution — dodgy in premise. Its lattice/poet code is tidy,
but its semantic signal is **binary** (MUSE translation edge or nothing).
Consequence: every "high-semantic" line it produces is built purely from
cognates we had already separated (`west air rose ~ ouest airs roses`,
`breeze proves rose ~ brise prouve roses`). It rediscovers the easy subset
and cannot find what you actually asked for — pairs that sound the same and
mean something *close*. Its `dream~rime` gets an arbitrary 0.475 from a
hack, not a measurement. The lattice data structure (semantic component →
homophonic radius) is worth keeping as a query pattern; the scoring isn't.
It also rewrote mapping-web.json into an incompatible format — don't merge
as-is.

## The measured answer (this branch, `sound_meaning.py`)

Graded cross-lingual meaning = multilingual embeddings (one vector space
for EN and FR, cosine = meaning closeness, phrases embed natively so
combinations are covered). Over the 5,596 usable sound pairs:

| band | sem cosine | count | meaning |
|---|---|---:|---|
| identical | ≥ 0.80 | 694 | cognate-grade (prove~prouve) |
| **close** | 0.55–0.80 | **1,117** | **the prize: pled~plaide, teepee~tipi, to~tout** |
| related | 0.35–0.55 | 2,135 | domain echo — theming material |
| unrelated | < 0.35 | 1,650 | pure puns |

So: 1,811 pairs sound-same + mean-same-or-close. Caveat: MiniLM embeddings
are noisy on 1-syllable words (awed~odes scores high on form, not sense) —
which is precisely where the LLM earns its keep.

## The recipe: three jobs for the LLM, zero of them phonetic

The iron rule (proven across this whole project): **the LLM never invents
or scores phonetics.** Sound is the deterministic pipeline's job. LLMs
hallucinate IPA and overrate spelling similarity — that's why freehand
GPT attempts look dodgy. Give the LLM only meaning-work:

**Job 1 — judge the top slice.** Batch the `identical`+`close` bands
(1,811 rows, ~40/call) with this shape:

    For each EN~FR pair, rate semantic relatedness 0-10 and name the
    relation. JSON per row:
    {"en":..., "fr":..., "fr_gloss": "<plain-English meaning of the
    French>", "relatedness": 0-10, "relation": "same|synonym|domain|
    morphological|none", "note": "<=10 words"}
    Do not consider spelling or sound similarity AT ALL — meaning only.

  The `fr_gloss` field is the anti-hallucination trick: forcing the model
  to translate the French first makes the relatedness honest. Disagreement
  between embedding band and LLM rating = audit queue, exactly like the
  dual-judge pattern in the matcher.

**Job 2 — synonym-bridge proposals (the combinations you asked about).**
For pairs in `related`, ask: "give 5 EN synonyms of <en> and 5 FR synonyms
of <fr>" — then run the *deterministic matcher* over the 25 cross
combinations. The LLM proposes the meaning-preserving moves; the pipeline
verifies the sound. This is how `two~tout` (related) can be upgraded: tout
→ synonym "tous"... → matcher rescores. New entries inherit full QC.

**Job 3 — line steering.** Feed `generate.py` lots filtered to
identical+close bands only, have the LLM pick/order units into a line that
parses in both languages, then re-verify the concatenated line with the
matcher. The LLM arranges meaning; the pipeline owns sound, always.

Where to run it: the production app's existing LLM plumbing
(`lib/integrations-openai-ai-server`) fits Job 1 as a mining task today.
