# Why generating the Mother Goose rhymes is hard (tested)

Tried to generate homophonic Mother Goose from the reranked v5 table
(`test_mothergoose_gen.py`, public-domain 1916 source lines only). It fails, and
the failure is diagnostic: it shows the generation problem is **not** a dictionary
problem at all. Three compounding causes, in increasing depth.

## 1. Coverage holes — the iconic words have no homophone

The most memorable nursery words are non-lexical, so the table simply has no
entry:

| line | word-for-word FR | coverage |
|---|---|---|
| Jack and Jill went up the hill | jack end [jill] ouest happe [the] bill | 71% |
| Hickory dickory dock | [hickory] [dickory] déc | 33% |
| Hey diddle diddle | [hey] [diddle] [diddle] | **0%** |
| Humpty Dumpty sat on a wall | [humpty] [dumpty] sites âne [a] walk | 50% |

`humpty`, `dumpty`, `diddle`, `hickory`, `jill` — the words that *make* the rhyme
— are exactly the ones with no French homophone headword. A word-keyed table can
never cover them, because they aren't words.

## 2. Coherence collapse — independent picks are word-salad

Even where coverage exists, choosing each word's homophone *independently*
ignores French syntax, so the result means nothing: "jack end ouest happe the
bill" scores L2-coherence ~0.27. The reranked table optimizes each word in
isolation; a line needs the words to cohere, which per-word lookup cannot do.

## 3. The structural wall — homophony lives ACROSS word boundaries

This is the real reason, and word-for-word can *never* fix it. Generation by
substitution preserves the **English** word boundaries. But real homophonic
translation **re-cuts the phoneme stream** into the target language's *own*
boundaries, which fall in different places:

```
EN  (6 words):  Humpty Dumpty sat on a wall      hʌmptidʌmptisætɔnɐwɔl
human FR (9):   Un petit d'un petit s'étonne...  œ̃pətidœ̃pətisetɔnoal
```

The two phoneme streams match — but "Humpty" (one English word) maps *across*
"un petit" (two French words). The boundaries dissolve and re-form elsewhere.
**A dictionary keyed on English words structurally cannot produce this**, no
matter how perfect its entries: it can only swap word→word, never 1-word→2-words
or split a word's phonemes between two French words.

Worse, line-sound stays low (~0.42–0.56) even per word, because concatenating
per-word homophones breaks the cross-boundary liaison/elision that makes the real
rendering *sound* like the English in the first place.

## What this says the generator must be

The dictionary (v5 / reranked) is the **parts bin** — necessary, near-ceiling,
but the wrong *operation* for generation. Producing Mother Goose needs:

1. **Whole-stream resegmentation (the oronym/juncture engine, N3).** Decode the
   line's phoneme stream by re-cutting it into French words — the
   `phonetic_decoder` / `fragment_weave` direction, not table lookup. Word
   boundaries become free.
2. **An L2-coherence model in the search (impetus I).** So the re-cut lands on
   *fluent* French, not salad — the gating component, still missing (bigram only).
3. **Content selection + human/LLM judgement (impetus III).** Whole-line decoding
   is low-yield (the over-constraint wall: 25% in the earlier test), because the
   non-lexical words don't carve cleanly. Van Rooten didn't render every line —
   he *chose* rhymes that re-cut well and accepted absurd French sense. The
   machine equivalent is content-selected generation + a coherence judge, not
   forcing a fixed line.

So the problem with generating Mother Goose is not that v5 is too weak — it's
that **generation is resegmentation-under-coherence, a different operation from
the word-for-word lookup v5 perfects.** The reranked table tops v5 word-for-word
(done); it cannot, and was never going to, generate the poems. That needs the
juncture engine + an L2 model + selection — exactly the open frontier the rest of
this session converged on.
