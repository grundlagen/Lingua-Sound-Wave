# Dual translation at scale — the scope law (measured)

Question: can FRENCH MEANING be mapped to ENGLISH MEANING while the sound also
matches — dual translation at scale? Answer: **yes at word scope, partially at
phrase scope, no at sentence scope by word substitution.** (`dual_scale.py`)

## The scope law

| scope | dual (sound ∧ meaning) achievable | evidence |
|---|---|---|
| word | **9,231 DUAL-S pairs** (sound ≥ 0.75, literal translation) | `dual-pairs.tsv`, 102,899 total |
| phrase (2–4 words) | partial — where DUAL words cover the content | hybrid strategy below |
| sentence | **0/50** corpus lines reach sound ≥ 0.55 ∧ meaning ≥ 0.45 with real French | this bench |

Best sentence-scope candidates cluster at **sound ≈ 0.40–0.45, meaning ≈ 0.7–0.9**
(literal French) or the reverse (carve). The channels anti-correlate as scope
grows: preserving meaning fixes the words, and fixed French words almost never
re-spell the English phoneme stream. **This is why van Rooten abandoned meaning
— the scope law is the art's own constraint, now measured.**

Two artifacts to guard against (both hit and fixed here):
- **franglais leak**: untranslated English words score high on BOTH channels
  (cos(EN, EN-ish) ≈ 1, sound ≈ 1). Hard gate: every word ∈ Lexique.
- **MUSE noise**: its FR column contains English identity pairs (down, sat,
  who) — gate on Lexique only, never the MUSE FR side.

## UPDATE — the wall falls with the full toolkit (`beauty_compose.py`)

Rupert's push was right: 0/50 was word-ALIGNED substitution with literal glue.
Adding the beauty-of-language arsenal — DUAL one-for-ones, ladder GOLD
homophones, **mined zipf glue** (the≈de by th-stopping, we≈oui, is≈aise;
`zipf-glue.tsv`, 573 mappings), **EN/FR synonym chains** (word → synonym → its
translations → best sound), and **metaphor drift** (sound ≥ 0.6 ∧ cos ≥ 0.25)
— behind the same Lexique gate:

**12/20 corpus lines reach the Rooten band (sound ≥ 0.55 ∧ meaning ≥ 0.45) — was 0%.**

> we see the moon at dawn → *oui si le lune hâtent donnent* (sound 0.72, meaning 0.65)
> mary had a little lamb → *mairie aide et lille agneaux* (sound 0.56, meaning 0.62)

Meaning out of 9,000 words + zipf conjugation coverage: it is possible.

## The scalable recipe that follows

1. **DUAL-S anchors**: put translation∧homophone words where they exist
   (9k lexicon; 272 non-cognate for the art tier).
2. **Carve between anchors**: sound-first for uncovered spans (the engine).
3. **Meaning relaxes to theme** at sentence scope: cosine ≥ 0.3 as drift bound
   (paraphrase_translate's territory), not ≥ 0.45 word-alignment.
4. The genuinely-best path at sentence scope is METHODS_DEEP_DIVE's
   constrained decoding: an LLM proposes meaning-true French while the matcher
   FST constrains sound token-by-token — meaning freedom, sound hard.

## Common Voice / Drive code review (the second ask)

Drive notebooks reviewed (`commonvoice.ipynb`, `Untitled7.ipynb`, extracted zip):
- `commonvoice.ipynb`: wav2vec2-base-960h feature extraction over CV EN + FR
  audio — the old embeds-from-audio plan, no scoring loop.
- `Untitled7.ipynb`: the better design, never finished — dual ASR (FR+EN
  wav2vec2), **`Cnam-LMSSC/wav2vec2-french-phonemizer`** (real French audio →
  phonemes), MFA alignment, parselmouth pitch/intensity features.

Room for improvement found: our whole stack G2Ps French with espeak; the
French-phonemizer on REAL Common Voice audio would (a) validate/correct espeak
IPA at scale, (b) feed the ASR-confusion miner (METHODS_DEEP_DIVE F1) that
discovers homophones from acoustics. Blockers here: CV is HF-gated,
`torchaudio` absent — it is the Colab/GPU track, notebook design is sound.
⚠️ `Untitled7.ipynb` embeds a live HuggingFace token — revoke it.

## Real-audio verdict (`real_audio_g2p.py`, running here — no Colab)

Tatoeba French clips (live-API resolved; the bulk CSV export is stale) →
`Cnam-LMSSC/wav2vec2-french-phonemizer` (CPU) vs espeak G2P of the transcript:
**mean segment agreement 0.97**. espeak French is essentially correct; residue
is ə→ø colouring and casual ʒ-elision. The Drive Common Voice plan's real value
was validation, and espeak passes — the sound stack stands on solid ground.
(Scale the clip count by widening the candidate pool; API is rate-limited.)
