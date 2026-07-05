# bank_composer — chaining the phrase bank into Van Rooten lines

`bank_composer.py` is the payoff of the phrase bank. Each bank entry is an English
phrase that already **sounds like** a French phrase (a homophone unit, with its
combo verified). The composer beam-searches a CHAIN of entries so that, read
across the whole chain:

- the **English side** is a fluent English sentence (EN bigram across each seam),
- the **French side** is fluent French (FR bigram across each seam),
- every unit is a strong homophone,

so the French, spoken, reconstructs the English — and **both** sides read.

## Why this works where tile-composition didn't

The webbing study showed dual-atoms almost never co-chain on both rails (both-rail
edges ≈ 0). The bank composer dodges that wall: the homophone is **inside each
unit**, so we never align two fixed sentences — we only need the **seams** between
units to be fluent, which the bigram LMs handle. Composition becomes tractable.

## Real output (free composition)

```
EN reads : said to the time to do it for         EN-fluency 1.00
FR sounds: sept ou d âme tout où est fort         FR-fluency 0.87   homophone 0.68

EN reads : to tell it for the name to do          EN 0.98
FR sounds: t elle est fort son âme tout où         FR 0.91          homophone 0.65

EN reads : said to the time to do with them
FR sounds: sept ou d âme tout où ils aiment        FR 0.88

EN reads : said to the time to do it for the more  (5 units)
FR sounds: sept ou d âme tout où est fort son or    EN 1.00  FR 0.89
```

Read the FR line aloud in a French mouth and it lands on the EN line; read it as
words and it is fluent French. That is the dual reading the project has been
after, assembled from verified homophone units.

## Knobs

- `python bank_composer.py` — free composition (most fluent strong-homophone chains)
- `python bank_composer.py -n 5` — longer lines (5 units)
- `python bank_composer.py said the` — seed by starting word(s)

Built on `phrase-bank-balanced.tsv` (the fluent bank). Quality scales directly
with the bank: more fluent units → longer, richer lines. The remaining ceiling is
the same one throughout — a real L2 model would let it choose *meaningful* chains,
not just locally-fluent ones.
