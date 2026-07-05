# Achieving the homophonic-poem dataset: scholarship, copyright, and a tested strategy

What the gold corpus should be, where it can come from, and what `build_poem_dataset.py`
measured when we tried to auto-generate it. Reuses extant tools (soramimi /
generate / fragment_weave); reinvents nothing.

## The scholarship (what the field has established)

- **Alexandra Lukes, "Crypto-Back-Translation in Van Rooten's Homophonic Nursery
  Rhymes," *Translation and Literature* 29.3 (2020), 427–47** (the link you sent;
  paywalled). Its key idea — **crypto-back-translation** — is structurally useful
  to us: van Rooten presents the French as "original" poems with a mock-scholarly
  English apparatus (notes/glosses) that secretly *back-translate* to the hidden
  Mother Goose source. So the natural gold record is a **triple**, not a pair:
  `(English source rhyme, French homophonic rendering, the gloss/apparatus)`.
- **Louis & Celia Zukofsky, *Catullus* (1969)** — the best-known literary
  homophonic translation: English tracking the *sound, rhythm, and syntax* of
  Catullus's Latin. Scholarship (z-site.net; OpenEdition "Estranging the Classic")
  stresses it is **not** mere sonic transcription but a procedure "between eye,
  ear and lexical sense" — i.e. exactly the dual-reading-coherence target, with a
  **public-domain Latin source**.
- The technique is otherwise under-studied (the 2016 Paris "Sound / Writing"
  conference is a rare focused venue). Background: Wikipedia "Homophonic
  translation"; languagehat.com's survey.

Implication: there are **two** ready gold streams whose *source* side is free —
Mother Goose (EN, public domain) and Catullus (Latin, public domain) — even
though the famous *renderings* are not.

## The copyright reality (decisive for "find the books")

*Mots d'Heures: Gousses, Rames* (van Rooten 1967), *N'Heures Souris Rames* (de
Kay 1980), and the Zukofsky *Catullus* (1969) are **all in copyright**. We must
**not** ingest, scrape, or redistribute their text. What is usable:

- the **source** texts (Mother Goose; Catullus) — public domain, free to use;
- a **handful of lines** quoted in scholarship/encyclopaedias under fair use — the
  one van Rooten line already in `dataset.py`/`gold-dual-readings.tsv` is exactly
  this, and the gold stays at that scale on purpose;
- anything the project **generates itself**.

So the books are the **target to match, never a corpus to ingest.**

## What we tested (`build_poem_dataset.py`)

The obvious idea — "auto-translate public-domain rhymes into French homophones
with our decoder" — was run and measured. Result, honestly:

| | yield | machine mean dual-reading | human gold |
|---|---|---|---|
| fixed-source rendering | **3/12 chunks (25%)** | **0.23** | **0.43** |

A whole line over-runs the decoder gate; even phrase chunks mostly fail, and the
few that pass are weak (`Miss Muffet → "but must"` snd 0.46 / L2coh 0.77;
`hickory dickory → "critiqué"` snd 0.59 / L2coh 0.29). **Why:** fixing the exact
source phonemes *and* demanding fluent French is the **over-constraint wall** from
`CENTRAL_PROBLEM.md`, now at the line level. Van Rooten didn't greedily decode a
fixed stream — he searched a vast space with human judgement and *chose* which
renderings to keep.

## The corrected strategy (what to actually do)

The dataset is **not** "translate the rhymes." It is two complementary tracks:

1. **A tiny fair-use GOLD (calibration).** Attested human lines — the van Rooten
   line we have, plus any *individually* quoted in scholarship — stored as the
   crypto-back-translation **triple** (source, rendering, gloss). Role: anchor and
   calibrate the scorer (`dual_reading_eval.py`), never bulk training. Stays small
   by both law and design.

2. **CONTENT-SELECTED generation at scale (the real dataset).** Do **not** force
   every source line. Let the system mine where the languages *already align*:
   - `fragment_weave` grows shared streams and finds **new** dual-readings (not
     tied to a given source) — the high-yield direction;
   - `soramimi`/the decoder render only the source fragments that **clear the
     gate**, and we keep those — content selection, not forced translation.
   This is exactly impetus III ("choose what to say where the languages are in
   tune"), and it is copyright-clean because we generate it from public-domain
   sources and our own lexica.

3. **Second language pair for free: Catullus → English.** The Latin source is
   public domain; espeak has a Latin-ish path; this extends the gold beyond EN↔FR
   without any copyright exposure, and tests the multilingual claim.

## Why this doesn't reinvent the June 11–12 work

That window already built the pieces this strategy reuses: `generate.py`
(pattern-lot line generation across granularities), `soramimi.py` (the Van Rooten
EN→FR direction), `web.py`/`mapping_web` (the typed homophonic-semantic web),
`sound_meaning.py` (the meaning grade). The only **new** artifacts are the
evaluator (`dual_reading_eval.py`), the gold file, and this acquisition test —
the scoring/measurement layer those generators never had. Generation: extant.
Evaluation + dataset method: new.

## Next concrete steps (no new generator)

- grow `gold-dual-readings.tsv` only with individually-quoted, attributed lines
  (fair use), recorded as triples;
- run `fragment_weave --lm` as the scalable producer, score every output with
  `dual_reading_eval`, keep dual ≥ the gold's 0.43 as silver data;
- stand up the Catullus→EN source side to validate the second pair;
- the gating component remains impetus I: a real **L2-coherence model**, whose
  margin on the gold (bigram only +0.15 now) is the number to move.
