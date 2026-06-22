# Homophonic writing demo: rules, production, and arbiter verification

Run 2026-06-22 (espeak-ng 1.51 + panphon + numpy + wordfreq + bigram LMs).
Reproduce: `python demo_homophonic_writing.py`. Everything below is generated
or verified live — no hand-picking past the stated thresholds.

## 1. The phonetic rules (why the matcher hits AUC 0.993)

`matcher.combo = 0.5·ngram_dice + 0.5·feat_nw_sharp` (RESULTS.md, ROC-AUC
0.993 on 105 hand-labeled EN↔FR pairs, hard-negative AUC 0.992).

- **ngram_dice** — Dice coefficient over exact phoneme bigrams. High precision,
  order-aware; negatives sit near 0.04.
- **feat_nw_sharp** — Needleman–Wunsch over panphon articulatory features, with
  the single biggest lever in the study: **sharpen** the per-segment distance
  (÷0.35, clamp to 1) so unrelated segments saturate. Raised feat-nw 0.941→0.993.
- **EN↔FR equivalence floor** caps the substitution cost of pairs that legitimately
  cross the languages: p~b/t~d/k~ɡ at 0.20, i~ɪ / e~ɛ at 0.10 (French has no lax
  vowels), θ~s 0.25 (no English TH in French), ŋ~n 0.15, y~i 0.20; rhotic map
  ʁ ʀ ɾ r → ɹ; nasal split ɑ̃ → ɑn; offglides/schwa/h are cheap to delete.

Verified live:

| EN | FR | combo | channels | verdict |
|---|---|---:|---|---|
| shoe | chou | 1.00 | ng 1.00 / ft 1.00 | homophone |
| mayday | m'aider | 0.67 | ng 0.36 / ft 0.97 | homophone |
| dog | chien | 0.35 | ng 0.00 / ft 0.70 | not alike (correct) |

Benchmark re-run confirms: `combo` AUC **0.993**, hard 0.992, loose 0.997.

## 2. The fragments (the generative grammar)

`fragments.tsv` indexes EN→FR sound-blocks pulled from v5 alignments. The
high-count blocks are **sound-identical in both languages**, so a chain of them
is one IPA stream pronounceable as English *and* French:

    st ×372   ɹi ×244   ɛk ×227   ɛs ×199   ks ×191   ɛn ×184   tɹ ×183

Chain → decode through the EN word-trie = English phrase; decode the *same*
stream through the FR word-trie = a French phrase that sounds the same.

## 3. Production + arbiter (the honesty loop)

The generator can fool its own decoder, so every produced pair is re-scored by
`matcher.homophone_score` — the AUC-0.993 judge that never saw the generator.
A 140 s LM-steered run produced 117 novel pairs; **70/117 (60%) pass the
arbiter at combo ≥ 0.45**. Top LM-beam results sound-true *and* fluent both sides:

| EN | FR (same sound) | combo | en_flu | fr_flu |
|---|---|---:|---:|---:|
| in to | in tout | 0.65 | 0.87 | 0.76 |
| us if | as if | 0.57 | 0.87 | 1.00 |
| but serve | but savent | 0.65 | 0.77 | 0.64 |
| in receive | in rêve | 0.55 | 0.71 | 0.69 |

## 4. Where the goal is actually met today (the strong result)

The system's sweet spot is **one English word → a fluent French word-sequence**,
arbiter-confirmed. Of 1093 dictionary candidates (score ≥ 0.9), **37 are
sound-true (combo ≥ 0.55) AND read as real French** (fr_flu ≥ 0.5):

| EN word | FR phrase (same sound) | combo | fr_flu |
|---|---|---:|---:|
| beetroot | but route | 0.84 | 0.61 |
| monkey | mon qui | 0.73 | 0.62 |
| metal | met elle | 0.71 | 0.55 |
| medal | met elle | 0.73 | 0.55 |
| training | train in | 0.67 | 0.63 |
| beaches | but chaise | 0.68 | 0.61 |
| morning | mon in | 0.55 | 0.75 |
| mountain | mat in | 0.61 | 0.77 |
| precise | pris as | 0.71 | 0.57 |
| canoes | con use | 0.66 | 0.50 |

These are genuine cross-lingual homophones: *beetroot* spoken aloud is *but
route*, *monkey* is *mon qui*, *metal* is *met elle*.

## 5. Honest state of the goal

- **Word-level homophony: solved.** The matcher is at AUC 0.993 and the
  dictionary yields dozens of arbiter-confirmed, sensical EN-word↔FR-phrase pairs.
- **Phrase↔phrase, both sides fluent: the active frontier.** The LM-in-beam
  layer is the right lever — it surfaced "in to / in tout" and rejected
  "cessent cours"-grade salad — but only ~3 produced pairs clear the strict bar
  (combo ≥ 0.55 AND both fluencies ≥ 0.65). The bottleneck is FR-side phrase
  fluency: the 19th-century corpus under-scores colloquial bigrams.
- **Next lever:** seed chains FROM the certified phrase pairs (not just corpus
  bigrams), and add a modern FR subtitle corpus so the bigram LM rewards
  "c'est / il y a / qu'est-ce"-grade fluency the literary corpus misses.
