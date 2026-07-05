# Training data & corpus survey (homophonic / wordplay FR↔EN)

## The uploaded *Mots* PDF — checked, treated as copyrighted

Asked to verify a modern 12-author *Mots d'Heures*-style edition. On inspection
the file carries a **© copyright symbol** and **no open-license marker** (no
Creative Commons, CC0, or public-domain dedication anywhere in it). "Publicly
downloadable" ≠ public domain; under Berne a modern anthology is copyrighted on
creation unless explicitly released. So it is **not ingested, trained on, or
reproduced**. If an explicit CC/CC0 notice for that edition surfaces, reassess.

## Largest genuinely-open FR↔EN wordplay corpus: CLEF JOKER (CC BY 4.0)

The **JOKER Corpus** (Ermakova, Bosser, Jatowt, Miller; SIGIR 2023) is the
real, citable, openly-licensed resource — **CC BY 4.0**, English↔French parallel
**pun / wordplay** data with professional translations and pun-location
annotation, grown across the CLEF JOKER tracks 2022–2025 (detection, location,
interpretation, and **pun translation** — Task: ~1,405 EN wordplay instances with
~5,838 FR translations, plus larger detection sets). Access via the JOKER project
/ HAL `hal-04299292` (CLEF registration for the full splits).

Caveat: JOKER is *pun* translation (preserve the joke), not strictly
*sound-preserving* (Van Rooten) translation — so it is best for the **semantic /
pun** side and for evaluation, complementing our sound-first carves. With
attribution it is usable for training.

Nothing else open surfaced: general MT corpora (Europarl, CroissantAligned,
OPUS) carry no homophonic signal; the homophonic-translation *canon*
(Van Rooten 1967, de Kay 1980, Hulme 1981) is in copyright.

## Our consolidated training set: `train-homophonic.jsonl` (4,984 ex.)

Built by `build_training_set.py` from **our own generated data + PD English**
only (no copyrighted homophonic text):

| task | n | example |
|---|---|---|
| `word_carve` | 2,999 | "…sounds the same in French: to" → `toux` |
| `dual_atom` (sound ∧ meaning) | 896 | "…sounds like and means: sex" → `sectes` |
| `phrase_carve` | 1,089 | "…: Humpty Dumpty" → `un petit un petit` |

Each row carries `combo` / `coverage` / `fluency` so a trainer can filter to the
high-quality tail. This is instruction-shaped (prompt/completion) for fine-tuning.

## What to train (the standing recommendation)

The recurring ceiling is the **L2 model** — the bigram scores adjacency, not
sense. The dataset above is exactly what a small French-capable LLM would be
fine-tuned on to learn the **restructuring** (English phoneme stream → fluent
French re-cut) with our combo/fluency signals as reward. JOKER (CC-BY) can be
folded in for the semantic/pun channel and for held-out evaluation. Training
itself needs a GPU; this repo stages the data and the reward signals.

Sources: JOKER corpus (SIGIR 2023), HAL `hal-04299292`, CLEF JOKER tracks.
