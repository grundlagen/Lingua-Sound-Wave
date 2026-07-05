# Dictionary Archive — Version History

These are older dictionary versions preserved for reference. Only two dictionaries
are canonical:

- **`dictionary-v5.json`** — 10,846+ entries, 33-column schema. The active dictionary
  used by the composition pipeline.
- **`dictionary-v7-integrated.json`** — 16,208 entries. v5 (11,788) ∪ 4,420 decoder
  additions. Not yet wired to the composition pipeline.

## Version Progression

| Version | Builder | Entries | Key innovation |
|---|---|---|---|
| v2 | `refine_dictionary.py` | 8,319 | Curated Lexique/WikiPron G2P, multi-word FR, cognate tagging |
| v3 | `refine_dictionary.py` | 9,684 | Iterated refine |
| v4 | `refine_dictionary.py` | 10,652 | Iterated refine |
| v5 | `enrich.py` + `finalize.py` | 10,846+ | Composition-ready schema: alignment, syllables, gap ratios, junction features |
| v5-reranked | `rerank_v5.py` | 7,286 | Combo+zipf tiebreak re-rank (analysis only) |
| v6 | `build_v6.py` | 2,351 | Single-phoneme decoder + fillers + arbiter ranking |
| v6-integrated | (precursor) | — | v5 ∪ v6 (predecessor to v7-integrated) |
| v7 | `build_v7.py` | 2,069 | v5 retrieval (allophone-aware) ∪ v6 decoder |
| v7-remined | `remine_dict.py` | 9,912 | Stress-weighted prosody + semantic-cosine re-score (companion, not replacement) |

## How to Rebuild

```bash
# v5 pipeline (the canonical dictionary)
python build_dictionary.py              # block+rank from lexicons
python enrich.py                        # add alignments, pivots, syllables
python finalize.py                      # add composition gates

# v7 upgrade (adds decoder entries)
python build_v7_decode.py               # v5 + single-phoneme decoder additions
```
