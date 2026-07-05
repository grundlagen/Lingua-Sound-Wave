# Phrase bank — a database of sensical homophone matches to compose from

`phrase_bank.py` carves a large set of real English phrases (the most frequent
English bigrams + public-domain Mother Goose lines) into French **sound-first**
(the lever that nearly doubled combo), keeping each strong homophone with good
coverage. Each row is a reusable unit: an English phrase that, spoken, sounds like
a real French phrase.

Output `phrase-bank.tsv`: `en_phrase · fr_phrase · combo · coverage · fluency`.

## Settings (from `carve_quality.py`)

Sound-first decoding: beam 600, `lm_weight 0.10` (chase sound, not fluency),
`top_n 80` candidates re-ranked by the matcher; keep `combo ≥ 0.45`,
`coverage ≥ 0.78`. These were the measured-best knobs (mean combo 0.30→0.54).

## Sample rows

```
0.55  Humpty Dumpty          -> un petit un petit       (the Van Rooten line)
0.58  little Jack Horner     -> lord licorne
0.61  Hickory dickory dock   -> écrit critiques
0.53  Jack be nimble         -> déclin un bile
0.52  Jack and Jill          -> chicanes gin
```

## Van Rooten calibration

We do **not** reproduce the copyrighted book. We use only the single
universally-quoted line — *Humpty Dumpty → "un petit d'un petit"* — to confirm our
independent carve lands in the same homophonic neighbourhood: our
`Humpty Dumpty → "un petit un petit"` scores combo 0.55 and is one filler word off
the canonical phrasing. That is the calibration anchor: the engine reaches
Van Rooten-class matches on its own.

## On fluency (the known weak link) — and the balanced bank

`carve_quality` showed the bigram fluency weight was *hurting* the homophone when
used inside the search — so we carve sound-first and treat fluency as a re-rank,
not a search constraint. But v1 *selected* the best-combo carve and ignored which
candidates were fluent. The **balanced** bank (`phrase-bank-balanced.tsv`) instead
selects `argmax(combo × (fluency+0.3))` among strong-combo candidates:

| bank | combo med | fluency med | fluent (≥0.5) |
|---|---|---|---|
| `phrase-bank.tsv` (best-combo) | 0.59 | 0.29 | 142 (14%) |
| **`phrase-bank-balanced.tsv`** | 0.52 | **0.56** | **664 (64%)** |

A small combo cost nearly **doubles** fluency and 4.7× the count of fluent rows.
Balanced top rows are usable bilingual pairs: `said to → sept ou` (0.83/0.77),
`to tell → t elle`, `order to → hors du`, `but to → but tout`, `it for → est fort`.

Remaining levers (unchanged): a real L2 model in place of the bigram, and a
French-anchored bank (start from real French phrases so fluency is guaranteed).

Run: `python phrase_bank.py [n_phrases]`
