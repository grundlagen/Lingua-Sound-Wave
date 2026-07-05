# Open-source material, the papers, and testing against dictionary v5

Two asks: (1) gather only legitimately-open material and test it against v5
("word-for-word the best — you can't top it"); (2) mine the cited papers for
anything useful. Copyright kept clean throughout: the famous *renderings* are in
copyright and are **not** reproduced; only public-domain sources and documented
linguistic facts are used.

## Open / moral sources actually used

- **The Real Mother Goose (1916), Project Gutenberg #10607** — public domain.
  Used as the English *source* corpus (2,072 distinct words) for phrase-level
  coverage. (English Mother Goose is the free side; van Rooten's French is not.)
- **`dataset.py` positives** — dictionary/etymology-documented EN↔FR homophones
  (shoe/chou, key/qui, mayday/m'aider…). Linguistic facts, not copyrighted.
- **Catullus (Latin)** — public domain, noted as a second copyright-clean source
  pair (the Zukofsky *rendering* is in copyright; the Latin is not).

The famous books (*Mots d'Heures*, *N'Heures Souris Rames*, Zukofsky *Catullus*)
remain **targets to match, never corpora to ingest**.

## Testing against v5 (`test_against_v5.py`)

| measurement | result |
|---|---|
| coverage of documented homophone words | **33/38 (86%)** |
| v5's stored FR ≥ the documented homophone (matcher) | **27/33 (81%)** |
| documented homophone **beats** v5's stored choice | **6/33 (19%)** |
| Mother Goose source words with a v5 word-for-word entry | 1143/2072 (55%) |
| documented **phrase** homophones v5 holds as ready units | 1/10 |

**You're substantially right: v5 is at or near the word-for-word ceiling.** 86%
coverage, and where it has an entry it matches or beats the documented homophone
4 times out of 5, almost all S-tier at score 1.0 (shoe→choux, key→qui, sue→sous,
do→doux…).

**But it is not literally untoppable — and that's a concrete, free win.** In ~19%
of cases the documented homophone is better than v5's stored pick, because v5
selected by its build-time score, not a final matcher re-rank:

- `wee` → v5 has **ouïe** (0.62) where the obvious **oui** scores **1.00**;
- `dough` → v5 **do** (0.40) vs documented **dos** (0.41);
- a handful more in the same vein.

So the one place you *can* top v5 word-for-word is a **matcher re-rank of the
per-headword FR choice**: recompute combo over v5's own candidate FRs and swap in
the argmax. Small, safe, and it removes exactly these mis-selections — no new
data, just re-ranking the table v5 already has.

**The real gap is not the table.** v5 holds only 1 of 10 documented *phrase*
homophones as a unit, and covers 55% of rhyme words individually. Word-for-word
is **necessary but not sufficient**: turning those gold words into whole *lines*
that stay coherent in both languages is the dual-reading problem (CENTRAL_PROBLEM
/ OBJECTIVE_AND_GOLD), not a dictionary problem. v5 is the parts bin; the open
work is assembly.

## The papers — what's actually useful

- **Ryan Fraser, "Evading Frames: D'Antin van Rooten's Homophonic Mother Goose"
  (TTR 25.1, 2012).** Argues the homophonic rendering **"evades anchorage in any
  specific contextual frame"** — the French escapes the English's semantic frame
  rather than preserving it. This is **scholarly backing for retracting impetus
  IV**: the art is *frame-evasion*, so the target is each side independently
  coherent, NOT shared meaning. Confirms the corrected objective from theory.
- **Catford, "phonological translation"** (via **Brian Mossop, "Singing in
  Unknown Languages," JoSTrans 20, 2013).** Catford's term is the **proper
  theoretical name** for this whole system: translate the *phonological* level,
  replacing lexis/grammar. Mossop adds a practical apparatus — phonetic
  transliteration **plus multiple semantic glosses** written into the score —
  which supports storing each gold item as a *crypto-back-translation triple*
  (source, rendering, gloss) and keeping **several** candidate glosses.
- **Hannah Sarvasy, "Warblish: Verbal Mimicry of Birdsong" (J. Ethnobiology 36.4,
  2016).** Distinguishes **warblish** (mimic a sound stream with *existing words*)
  from **onomatopoeia** (coin *new* words). Our hard constraint — output must be
  real lexicon words — makes the system a **"warblish engine"**: it maps one
  language's uninterpreted sound stream onto real words of another. A clean
  framing and a reminder the phenomenon is cross-culturally attested.
- **Ryan Fraser, "Underground Games: Surface Translation and the Grotesque" (TTR,
  2018)** and **Voltolini, "Puns for Contextualists" (2012).** Surface translation
  theory + **"multi-stable" puns that need context to disambiguate** — which is
  exactly the dual-reading: one stream, two stable readings selected by which
  language frame the listener holds. Useful vocabulary for describing the artifact.
- The remaining listed papers (Cohen on French in *Villette*; Spence on
  chronograms; Spilka; Gibson on *Skippy*; Slessor & Voyer, "Algorithmic Mimesis")
  are tangential here, though "Algorithmic Mimesis" (translation as creation vs
  carrying-across) echoes our proposer/arbiter split.

### Net
The scholarship **confirms the objective** (Fraser's frame-evasion = no
meaning-equivalence term), **names the method** (Catford's phonological
translation; warblish), and **suggests the gold record shape** (Mossop's
transliteration + multiple glosses → the triple). The v5 test confirms the table
is near-ceiling, finds a small free re-rank win, and re-locates the open frontier
at line-level composition — consistent with every prior finding this session.
