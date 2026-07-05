# Is the stream built from 2-phoneme chunks? Can it be single-phoneme? (tested)

Your question was about the **composition** side (fragment_weave), where a shared
phoneme stream is *chained from chunks*. Checked and tested.

## What the chunks actually are

`fragments.tsv` (the blocks fragment_weave chains) are **2–4 phonemes**, minimum 2:

| chunk size | count | examples |
|---|---|---|
| 2-phoneme | 953 | st, ɹi, ɛk, ks, ɛn |
| 3-phoneme | 1149 | ɛks, ɛst, stɹ, ksp |
| 4-phoneme | 538 | ɛksp, kspl, ɹɛst, spɛk |

There are **zero single-phoneme chunks** — by design: `fragments.py` extracts
runs of length ≥2 (`MIN_LEN=2`). So yes: the stream is composed of 2-phoneme (and
up to 4) chunks, never single ones.

## Chunk size is only a generation-time choice

Confirmed directly: a stream built from 2-phoneme chunks (`st`+`ɹi`+`ks`) and the
same stream written as single phonemes (`s`+`t`+`ɹ`+`i`+`k`+`s`) are the
**identical segment sequence**. Once a stream exists it is just a phoneme string;
the decoder re-cuts it at **single-phoneme grain regardless** of how it was
chunked. So chunk size changes *which streams get generated*, not how any stream
is matched.

## Can it be single-phoneme? Yes — and it generates BETTER

I built a single-phoneme block set (the phoneme inventory, weighted by
occurrence) and compared generative yield against the 2–4 phoneme chunks: random
streams from each, decoded through both the English and French word tries, counting
how many land on **real words in both languages**.

| composition grain | streams fluent in BOTH languages |
|---|---|
| 2–4 phoneme chunks | 26/160 (**16%**) |
| **single-phoneme chunks** | 80/160 (**50%**) |

**Single-phoneme composition more than triples the dual-decodable yield.** The
reason is the opposite of what I'd assumed: the multi-phoneme chunks are
**consonant-cluster-biased** (st, ks, tɹ, ɛks, kspl) — clusters are exactly what
French *avoids*, so cluster-heavy streams are hard to carve into French words.
Single phonemes give the chain the freedom to land on simple CV sequences that
decode cleanly in both languages.

## So: your intuition was right

- The stream **is** built from ≥2-phoneme chunks today, and it **can** be built
  from single phonemes.
- Doing so **helps generation** (16% → 50% dual-decodable), because it drops the
  cluster bias the attested 2-phoneme blocks carry.

### The one caveat — yield vs attestation
The 2-phoneme chunks aren't worthless: each is an *attested cross-lingual* sound
unit (st~st proven in real alignments), so it may give more *natural* matches even
at lower yield. The yield test measures decodability, not naturalness. The likely
best inventory is **mixed**: single phonemes for coverage/freedom + the attested
multi-phoneme chunks as high-confidence anchors — finest grain for reach, chunks
for quality. That's a concrete fragment_weave improvement: add single phonemes to
the block pool (a `--fine` mode) and let the arbiter rank the results.

Net: matching already runs at single-phoneme grain; **generation does not, and the
test says it should** — composing from single phonemes (alongside the chunks)
is a real, measured win.
