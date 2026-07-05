# Multi-language framework — the button

`multilang.py` is the single entry point. Give it a language pair; it builds
both pronunciation lexicons, mines the homophone dictionary, and (with MUSE)
the meaning edges — the whole loop, generic.

```
python multilang.py en es                       # build + mine, en->es
python multilang.py en es "the sea is cold"     # + translate (when wired)
```

Per-pair artifacts land in `pairs/<src>-<tgt>/`.

## Why it generalizes

The matcher core (`matcher.py`) was always language-agnostic: panphon
articulatory features + sharpened NW + n-gram Dice score *any* two IPA
strings. The only EN-specific things were the data-loading paths and the
hand-tuned EN↔FR equivalence table — and that table is a *floor* under the
panphon baseline, so a new pair runs at panphon quality immediately and
specializes later via `learn_costs.py` (per-pair learned overlay).

A new pair needs only three facts, all in the `PAIRS` registry:
1. **espeak voices** (G2P) — `--voices` lists what's installed
2. **wordfreq codes** (lexicon source) — top-N frequency words
3. **a MUSE bilingual dict** (meaning edges) — fbaipublicfiles en-XX

## Status per requested pair (proven / ready / blocked)

| pair | G2P | wordlist | MUSE | status |
|---|---|---|---|---|
| **en–fr** | ✓ | ✓ | ✓ | reference, fully built (10.6k dict, woven web) |
| **en–es** | ✓ | ✓ | ✓ (112k) | **PROVEN** — 783 pairs mined, 23 S-tier (`see~si`, `team~tim`, `mean~min`) |
| **en–ga** (Irish) | ✓ espeak `ga` | ✗ wordfreq has no Irish | en-ga exists | **needs a wordlist** — drop an Irish frequency/word list at `tgt_wordlist`; everything else runs |
| **en–he** (Hebrew) | ✓ but needs native script | ✓ wordfreq `he` | en-he exists | **ready w/ caveats** — RTL; espeak must receive Hebrew-script words (works on the wordfreq list, which is native script) |
| **ar–he** (Arabic↔Hebrew) | ✓ both | ✓ both | ✗ no MUSE pair | **meaning edges missing** — needs an ar-he dictionary (or pivot through en: ar→en→he) |

## The honest cross-language findings

- **Shallow-orthography targets (Spanish, Irish) are the sweet spot**: espeak
  G2P is near-perfect, so sound edges are trustworthy with no per-pair tuning.
- **Semitic targets (Hebrew, Arabic) need three things English↔Romance didn't**:
  native-script input to espeak (transliteration breaks G2P), RTL-aware
  display, and — because their roots are templatic (consonant skeletons with
  vowel patterns) — the equivalence model should probably weight consonants
  over vowels. That's a `learn_costs.py` run on a seed set, not new code.
- **Pivot meaning edges** solve the missing-MUSE problem generally: for any
  pair lacking a direct bilingual dict, route meaning through English
  (ar→en→he), which the graph already supports as two meaning hops.

## What's generic vs per-pair (so adding a pair is a config edit)

Generic (untouched per pair): `matcher.py`, `phonetic_decoder.py` (trie +
beam), `chain_game.py`/`weave.py` (graph + chains), `enrich.py`, the
embedding anchor, the periphrasis layer.

Per-pair (all in `multilang.PAIRS`): espeak voices, wordfreq codes, MUSE
URL, RTL flag, optional wordlist path. Adding `en-it` or `en-de` is one
dataclass line.

## Next to fully light up a new pair
1. add the `LangPair` line (done for the five above)
2. `python multilang.py <src> <tgt>` — builds lexicons + mines dictionary
3. point `chain_game.build_graph` at `pairs/<pair>/dictionary.json` + the
   pair's MUSE file (one path swap, already parameterized in spirit)
4. optional: `learn_costs.py` on the S-tier to specialize the cost model
