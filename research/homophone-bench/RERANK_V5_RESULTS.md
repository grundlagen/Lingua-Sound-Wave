# An alternative word-for-word table: matcher re-rank of v5

`rerank_v5.py` builds `dictionary-v5-reranked.tsv` — v5's word-for-word picks,
but choosing each English headword's French by the matcher instead of by v5's
saturated build-time score. **No new data**: it only re-ranks candidates v5
already contains.

## The fix

v5 selects the "best" FR per headword by its build-time `score`, which saturates
at 1.0 — so among ties it picks arbitrarily and sometimes lands on the rarer or
worse-sounding word. `wee` carried both `ouïe` and `oui` at 1.0; v5 shipped
`ouïe`. Re-rank key:

    (round(combo, 3) desc, zipf_frequency(fr) desc)

best sound first; among sound-ties, the more natural (frequent) French word.
`combo` is computed from v5's **own stored IPAs** (deterministic, no espeak), the
same two channels as `matcher.homophone_score`.

## Result

| metric | value |
|---|---|
| headwords re-ranked | 7,286 |
| selection changed | **1,360 (18%)** |
| mean combo (v5 → reranked) | 0.6992 → **0.7090** (+0.0097) |
| documented homophones fixed (of 7 spot-checked) | 5 (wee, sea, knee, shoe, bell) |

Representative swaps (v5 → reranked):

| en | v5 pick (combo) | reranked (combo) | why better |
|---|---|---|---|
| wee / we | ouïe (0.40) | **oui** (1.00) | exact homophone, far more frequent |
| talk | toc (0.48) | **toque** (1.00) | exact, common |
| seen / scene | seine (0.71) | **seines** (1.00) | better sound match |
| sees / seas | six (0.73) | **sise** (1.00) | better sound match |
| partly | bat lis (0.54) | **party** (0.82) | one natural word beats two-word salad |
| accept | acceptes (—) | **accepte** | correct inflection |

The 18% that change split into two honest kinds: **true upgrades** (we→oui,
talk→toque — combo genuinely higher) and **tie-breaks toward naturalness** (a
single frequent word replacing equal-combo multiword salad). A handful of
low-combo headwords (cartoon→clowns 0.68) only pick the least-bad option — the
re-rank never lowers combo, but it can't manufacture a good match where v5 had
none.

## What it is and isn't

- **Is:** an improved *default* word-for-word table — the best single FR per
  headword by sound-then-naturalness. A drop-in alternative to v5's `best` pick.
- **Isn't:** a replacement for v5's full candidate set. The other FRs per
  headword remain valuable for *composition* (a line may need `sis` not `si` to
  fit a junction). The reranked file is the headword default; the v5 candidate
  lists stay the parts bin.
- **Doesn't mutate** `dictionary-v5.json`. It's an additive, regenerable view.

## Honest scope

This confirms and bounds the earlier finding: v5 is near-ceiling, and the only
free word-for-word win is exactly this selection fix — worth +0.01 mean combo and
~1,360 cleaner entries, but it does not move the real frontier (composing these
gold words into coherent *lines*). It tops v5 precisely where the test said it
could, and no further.
