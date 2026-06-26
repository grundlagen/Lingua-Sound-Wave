# Drive trawl — methods from the old Lingua Weaver notebooks

Trawled the exported Drive notebooks (your own work). What's there, and how it
folds into the engine.

## Found methods

- **`phoneme_mapping_reference.py`** — a comprehensive EN phoneme **equivalence
  map** (48 groups) + 18 named **linguistic rules** (aspiration, th-fronting,
  dark-l, flapping, h-dropping, final devoicing, linking-r, nasal assimilation,
  vowel reduction…). Extracted to `drive_phoneme_map.py` and entered in the
  tournament. The richest hand-curated EQUIV we have.
- **Untitled29** — articulatory **word2vec** for phonemes (phoneme embeddings).
  This is the right "embeds for sound" instinct; we now derive it data-first from
  gold (`phoneme_map.py` -> PMI/SVD), which is more robust than training word2vec
  on a small phoneme corpus.
- **Untitled2 / Copy-of-34/35/36 / frenchaccent** — **wav2vec / TTS-weight
  embeddings** (the "embeds from TTS weights" fascination). See assessment below.
- **Untitled26** — homophone phoneme-embedding model (train); **Untitled15** —
  espeak-ng IPA + meter pipeline; many Untitleds — homophone matching iterations.

## Tournament result (sound channel, 105 labelled pairs)

```
combo                0.993 AUC  sep +0.50   <- winner overall
feat_nw_sharp        0.993 AUC  sep +0.33
drive_equiv_seqmatch 0.988 AUC  sep +0.62   <- YOUR Drive method, strong
ngram_dice           0.986 AUC  sep +0.66
prosody              0.957 AUC  sep +0.23   <- recall signal, not a discriminator
```

Integration: **keep combo** as the sound judge (unchanged, as requested); your
Drive equivalence method is competitive and worth blending; prosody stays a
re-rank/recall signal, not the judge. The gold-learned `phoneme-map-gold.tsv`
confirms both from data.

## On "embeds from TTS weights" (frenchaccent / wav2vec)

The fascination: derive a sound-similarity space from a TTS/ASR model's internal
weights/activations. Honest verdict (matches where you landed, and our own audio
investigation): it's **hard and low-yield for THIS task** — acoustic embeddings
(wav2vec/whisper) score everything high (poor separation; AUDIO_INVESTIGATION.md),
and TTS weight-space isn't a clean phoneme metric. What actually works:
- **articulatory features** (panphon) as the base sound embedding, +
- **gold-learned substitutions** (`phoneme_map.py`) to weight them empirically.
That gives a phoneme embedding grounded in real homophone data, without the
acoustic noise. Keep wav2vec only as an optional multi-voice re-ranker on a
shortlist, never the judge.
