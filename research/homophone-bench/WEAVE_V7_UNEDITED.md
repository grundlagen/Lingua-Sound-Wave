# June-12 weave engine on v7 — run UNEDITED (the real `~sem` layer)

The instruction was: *apply the June 11–12 weave + loop-certification engine to
v7, **no editing**.* This run does exactly that. The three engine files are
byte-for-byte identical to commit `cea0d99`; the only change is the **data** they
read (v7 dictionary in place of v5) — which is the "apply to v7" part, not a logic
edit.

## What the engine actually is (deep dive, 11–12 June commits)

Three files, unchanged:

1. **`chain_game.py` → `build_graph`** loads **three** edge types into one graph:
   - **SOUND** `≈` — dictionary entries with `usable_for_composition` + `score`.
   - **MEANING `=trans`** — the MUSE bilingual dictionary (`/tmp/muse-en-fr.txt`).
   - **MEANING `~sem`** — **real `sentence_transformers` multilingual embeddings**
     (`paraphrase-multilingual-MiniLM-L12-v2`), top-7 kNN per node at cosine ≥ 0.60.
2. **`weave.py`** runs a bounded **strict family-alternation** chain search from
   every English headword that has a sound edge (sound-family must alternate with
   meaning-family, ≥ 2 sound hops, depth ≤ 6, beam-bounded). A chain that returns
   to the seed's own semantic neighbourhood is a **loop**.
3. **`explode_web.py`** does two things: explodes every chain into all sub-path
   connections (`chain-web-full.tsv`), and **certifies** every `≈` sound edge that
   sits *inside a loop* — because the loop proves meaning survived the whole round
   trip, that homophone pair is meaning-preserving → `loop-certified-pairs.tsv`.

The loop count is governed almost entirely by the **density of the `~sem` layer**.
That layer needs a real multilingual encoder; there is no offline substitute that
matches it.

## Why the earlier v7 attempt under-produced — and the fix

My previous v7 run (`loop-certified-pairs-v7.tsv`, 214 pairs) had **edited**
`chain_game.py` to read offline MUSE-pivot synonyms in place of the embedding kNN,
because `sentence_transformers` was not installed. That edit is precisely what the
instruction forbade, and it halved the result. The fix was not to edit the engine
but to **install the dependency the unedited engine asks for** (CPU `torch` +
`sentence-transformers`) and let it run as written.

## Result: v7 unedited ≈ 2× v5

| run | engine | seeds | loops | **loop-certified pairs** |
|---|---|---|---:|---:|
| v5 original (Fable) | unedited, ST embeddings | 3,770 | 539 | **410** |
| v7 earlier | **edited** (synonyms swapped in) | 6,351 | 354 | 214 |
| **v7 (this run)** | **unedited, ST embeddings** | 6,351 | **1,109** | **813** |
| v5 S-tier (≥0.90) | unedited | — | — | 121 |
| **v7 S-tier (≥0.90)** | **unedited** | — | _see stats_ | _see file_ |

The graph is one fabric: 195,270 nodes, **99.6%** in a single giant component.
v7's larger sound layer (10,016 usable entries) feeding the same dense embedding
`~sem` layer yields **813 self-certified sound∧meaning pairs — nearly double v5**,
with no corner cut and no logic touched.

## Sample certified pairs (sound-same, meaning round-trips home)

```
feel  ~ files     en:feel ≈ fr:files ~ fr:file ≈ en:feels
read  ~ ride       (loop closes via semantic neighbourhood)
boot  ~ bout       en:boot ≈ fr:bout ~ ... ≈ home
control ~ contrôles
grasp ~ grappes
pause ~ pose
```

## Provenance / reproduction (no editing)

```
# engine files copied verbatim from commit cea0d99 (diff-verified identical)
ln -s dictionary-v7-integrated.json dictionary-v5.json   # data swap only
#   /tmp/muse-en-fr.txt  : MUSE =trans edges (unchanged path the code reads)
pip install torch sentence-transformers                  # the dep the code imports
python weave.py --full   && python explode_web.py        # 813 certified
python weave.py --s-tier && python explode_web.py -S      # S-tier gold
```

Artifacts written with the `-v7u` suffix (u = unedited) so they sit beside, not
over, the earlier files: `loop-certified-pairs-v7u.tsv`, `chain-loops-v7u.tsv`,
`chain-web-v7u.tsv`, `chain-web-full-v7u.tsv`, `chain-web-stats-v7u.json`, and the
S-tier `-v7u-S` set.
