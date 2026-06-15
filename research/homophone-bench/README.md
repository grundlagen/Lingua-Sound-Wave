# Homophonic Engine — dual-channel translation (sound *and* sense)

A translation engine for the form Van Rooten and Luis van Rooten played by
hand: render a text in another language so it **sounds like the original**
while **carrying its meaning**. Drop in a paragraph, get the dual rendering.
Built to be tinkered with — for poetry, lyrics, puns, or translation-prize
experiments.

```bash
python engine.py --text "the pale moon lights the quiet sea"
echo "your poem here" | python engine.py --drift 0.7 --show-work
python engine.py --pair en-es --text "the sea is cold"
```

```
ORIGINAL    the pale moon lights the quiet sea
HOMOPHONIC  Pâli des riment light soft si      [fr, drift=0.55]
```

## One run, ordinary resources — and that is the finding

You do **not** need a parallel corpus, a trained model, or paid APIs. The
whole engine stands on four free, offline-cacheable resources:

| resource | role | free? |
|---|---|---|
| **espeak-ng** | grapheme→IPA for ~100 languages | yes, apt |
| **wordfreq** | frequency word lists per language | yes, pip |
| **panphon** | articulatory features (language-agnostic phoneme distance) | yes, pip |
| **MUSE** bilingual dicts | meaning edges (en-fr, en-es, en-he, …) | yes, download |
| Datamuse (optional) | synonyms + glosses for periphrasis | yes, no key, cached |

**Is the seed enough to learn pronunciation rules?** Yes — proven. panphon
gives a working phoneme-distance baseline for *any* pair with no training.
Running `seed_learn.py` on a freshly mined pair then refines it: from the
en-es seed alone it recovered textbook English→Spanish rules — lax vowels
/ɪ ɛ/ → /e i/ (Spanish has none), reduced /ʌ ə æ/ → /a/, diphthongs →
monophthongs — with **no parallel corpus and no hand-coding**. The seed *is*
the training set. Bigger curated lexicons help recall at the margins, but
nothing is *required* beyond the four resources above.

## The knobs (it's a toy as much as a tool)

- `--drift` 0→1: pure sound (surreal soramimi) ↔ hold meaning hard. The dial
  the whole project turns on.
- `--pair en-fr | en-es | …`: any pair in `multilang.PAIRS` with a built
  target lexicon.
- `--show-work`: per-word sound/sense scores and the chain/periphrasis used.

## How a word transfers (the pipeline in one breath)

source word → {itself, synonyms, glossed descriptions, last-resort poetic
periphrases} → each candidate decoded homophonically into the target
language (phoneme beam search over the target pronunciation trie, with
learned + featural costs) → keep the rendering with the best
`sound · (sound↔sense by --drift)` blend, anchored to the source meaning by
multilingual embeddings so it can't drift into nonsense.

## Add a language in one line

`multilang.PAIRS` holds every pair-specific fact (espeak voices, wordfreq
codes, MUSE url, RTL flag). Then:

```bash
python multilang.py en it          # build lexicons + mine the dictionary
python seed_learn.py en-it         # learn it->en pronunciation rules from the seed
python engine.py --pair en-it --text "..."
```

Status today: **en-fr** fully built (10.6k-pair dictionary, woven chain-web,
loop-certified gold layer); **en-es** proven (783 pairs mined, Spanish rules
learned from seed); **en-ga / en-he / ar-he** registered with honest
per-pair notes in `MULTILANG.md`.

## How close is it? (measured, honest)

On an 11-content-word paragraph: **homophonic rendering 100%** (every word
gets a high-sound target rendering — a paragraph that *sounds* like the
source is producible now), **dual sound+sense ≈63% per word**. The gap to
full literary dual translation is **sentence-level fluency** — the target
reads word-salad because word order follows the source sound stream
(inherent to homophony). Closing it is a *generation* problem (arrange the
rendered material into target-grammatical lines), best done by an LLM
sitting on top of this deterministic engine as proposer/judge — the engine
owns the phonetics and never hallucinates them. That's the one piece that
wants an API key; everything else runs offline.

## File map

- `engine.py` — the button (drop text → dual rendering)
- `multilang.py` — language-pair registry + lexicon/dictionary builder
- `seed_learn.py` — learn a pair's pronunciation rules from its seed
- `matcher.py` — language-agnostic phoneme matcher (panphon + n-gram + costs)
- `phonetic_decoder.py` — homophonic beam decoder (Knight–Graehl style)
- `chain_game.py` / `weave.py` — the sound+meaning graph and chain-web
- `RESULTS.md`, `MULTILANG.md`, `PARAGRAPH_STATUS.md`, `LLM_RECIPE.md` — the
  measured findings and design notes
