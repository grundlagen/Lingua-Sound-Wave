# Full v7 (efficient) + the allophone finding

## v7, done without doing retrieval twice

You were right: v5 *is* the retrieval output (and a richer one — multiword,
pair-bank, overrides), so re-running retrieval just reproduces it. `build_v7_decode.py`
therefore **reuses v5 as the retrieval leg** and runs **only the single-phoneme
decoder**, keeping what it adds:

```
v7 = v5 (11,788)  +  4,420 decoder additions  ->  16,208 entries
additions: 150 S / 696 A / 3,574 B   gold 148   filler 309
```

So v7 strictly exceeds v5 (16,208 > 11,788) with no redundant work — the decoder
recovered S/A homophones outside v5's retrieval reach plus filler/line carves.
`dictionary-v7-integrated.json` is the consumable; feed it to Round Rabbit and the
carve engine.

## The allophone layer: already covered by EQUIV (honest)

`allophone_layer.py` is the separate, borrow-from-existing allophone enrichment you
asked for: it expands both sides through `matcher._variants` plus the classic
English allophones (flapping /t,d/→[ɾ], coda l-darkening /l/→[ɫ]) and scores the
max combo over realizations.

**Result: 0/9 gain.** Flapping and darkening did not improve any test pair —
because the matcher's `EQUIV` table **already** encodes those equivalences: t~d at
0.20 (voicing), the rhotic map (ɾ/ʁ/r → ɹ), and l~w. So "butter's [ɾ] sounding like
/d/" is already captured by t~d; the explicit allophone is redundant.

**What this means:** the system is already allophone-aware at the level that
creates homophony — the broad-phoneme matcher + EQUIV + `_variants` subsume the
common allophonic processes. A *separate* allophone layer only adds value for
distinctions NOT in EQUIV (e.g., language-specific realizations like the Spanish
β/ð/ɣ spirantization, or tone/pitch), which is where to point it for new language
pairs. For EN↔FR, the allophony is handled; keeping the layer as an optional
re-ranker is fine but it is not a lever here.

This is consistent with `REPRESENTATION.md`: allophones live as match-time rules,
and the ones that matter are already in the table — confirming the design rather
than extending it.
