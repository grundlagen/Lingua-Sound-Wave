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

## On fluency (the known weak link)

`carve_quality` showed the bigram fluency weight was *hurting* the homophone when
used inside the search — so we carve sound-first and treat fluency as a stored
re-rank signal, not a search constraint. The bank keeps the `fluency` column so a
composer can prefer fluent rows. The deeper fix is unchanged: a real L2 model in
place of the bigram. A complementary data move (French-anchored entries: start
from real French phrases so fluency is guaranteed) is the natural next expansion.

Run: `python phrase_bank.py [n_phrases]`
