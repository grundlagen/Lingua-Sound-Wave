# Sentence-level generation: architecture, training method, model choice

Where the project stands: the 13M char-LSTM proves EN→FR word transduction is
learnable from the 9,803 strict-gold pairs (`shadow→chaude`), but sentence
output today is **word-by-word concatenation**, and the bigram juncture miner
already proved why that ceiling exists: *word boundaries are where homophonic
translation lives or dies*. "petit ami" ≠ "petit" + "ami". The liaison branch
(`claude/lingua-elision-schwa-liaison-0lud9i`) built cross-word juncture
scoring for the judge; this plan puts the same phonology into **generation**.

## 1. The core shift: carve the stream, not the words

Sentence generation must operate on the sentence's **continuous phoneme
stream with French sandhi applied**, not on per-word lookups glued together:

```
EN sentence
  └─ EN g2p (deterministic)            "the shadow falls" → ð ə ʃ æ d oʊ f ɔː l z
       └─ TRANSDUCER (seq2seq LLM)     phoneme stream → French words
            └─ sandhi_fr.py            liaison/elision/enchaînement on the FR side
                 └─ Agent B + combo    judge the SPOKEN stream, not the written words
```

Two consequences:

- **Training targets** must be sandhi-processed: the FR side of every
  sentence-level example is phonemized *as spoken* (liaison consonants
  realized, elisions applied, schwas dropped) before scoring/filtering.
  Otherwise the model learns to match written French, which is not what
  Agent B hears.
- **Word boundaries become free material**: liaison consonants (`les amis` →
  z) and enchaînement give the carver extra phonemes to spend — exactly the
  segments the word-by-word system wastes.

`research/qwen-finetune/sandhi_fr.py` implements the rules deterministically
(elision, obligatory/forbidden liaison, h-aspiré blocking, enchaînement
annotation) so both the data builder and the judge share one phonology.

## 2. Model: ByT5 as the transducer, Qwen as the composer

The "more technical LLM" question has a two-part answer, because the task has
two different shapes:

### Transducer — **ByT5-base** (580M; `google/byt5-base`)

The phoneme→French-words step is a *character transduction* problem, and the
known failure mode of normal LLMs here is the tokenizer: BPE hides sound
structure (this is why the Qwen schema shoves IPA into the prompt). **ByT5 is
token-free** — it reads and writes raw bytes — which makes it the standard
architecture for exactly this class of task (G2P, transliteration,
morphological inflection; see the ByT5 paper's word-noise results). Compared
to the 13M LSTM it brings pretrained multilingual byte-level knowledge of
French orthography, so it doesn't have to learn from 9,803 pairs that French
words end in `-eau` and `-tion`.

- Formulation: **P2G with context** — input `EN-IPA stream [SEP] EN text`,
  output French word sequence. Encoder-decoder beats decoder-only for pure
  transduction at this scale, and byt5-base trains on one 24 GB GPU.
- ByT5-small (300M) for iteration speed; byt5-base for the real model.

### Composer — **Qwen3-8B-Instruct** (LoRA, the existing kit)

The transducer proposes sound-true candidates; it knows nothing about
*meaningful* French. The composer picks and smooths: given the EN sentence
and n transducer candidates per span, produce the most fluent French line
that stays inside the verified sound budget. This is a language task, so a
real instruction LLM is right — and it's the fix for the standing ceiling in
`read me first` ("the recurring limiter is the fluency/L2 model"; the bigram
LM scores adjacency, not sense). The existing `research/qwen-finetune/`
scripts train it unchanged; only the prompt gains a CANDIDATES block.

An honest alternative if you want one model instead of two: Qwen3-8B alone
with IPA-in-prompt does both jobs adequately after DPO, but expect weaker
novel-word carving than ByT5 — bytes beat BPE for sound.

## 3. Training method: curriculum SFT → verifier DPO

**Stage 1 — curriculum SFT** (transducer). Order the data by juncture
complexity, since sentence skill is exactly juncture skill:

1. *Words* — the 9,803 strict-gold pairs (what the LSTM already saw).
2. *Bigrams with junctures* — mined pairs from `bigram_juncture_miner.py`
   with the sandhi-applied stream as input; this is where liaison/elision
   enter the model.
3. *Phrases* — phrase bank + whole-line carves (`whole_line_carve.py`).
4. *Sentences* — composed synthetically: sample gold word/phrase units,
   join them, apply `sandhi_fr.py` to the FR side, keep only compositions
   whose spoken-stream combo ≥ 0.5. Cheap, unlimited, and juncture-correct
   by construction.

**Stage 2 — verifier loop** (both models). Already scaffolded in
`sample_and_dpo.py`: best-of-16 at temperature 0.9, reward =
`combo(EN-stream, sandhi(FR)) + 0.3·fluency(FR)` — note the reward scores the
**sandhi-processed** French, closing the loop with the phonology. Best sample
→ SFT row; (best, worst) with gap ≥ 0.15 → DPO pair; β = 0.1; iterate 2–3
rounds. This is where 0.29-combo words become 0.7s: the verifier is trusted
(AUC 0.993), so the model only ever learns from sound-verified output.

**Stage 3 — the flywheel.** Every loop-accepted sentence (spoken-stream
combo ≥ 0.55) goes back into the stage-4 curriculum pool. The system
generates its own next dataset, verifier-filtered.

## 4. The babel rules, concretely

| rule | example | generation effect |
|---|---|---|
| elision | `le` + vowel → `l'` | frees a syllable; obligatory, deterministic |
| obligatory liaison | `les amis` → /le.z‿a.mi/ | inserts /z t n/ etc. — free consonants for the carve |
| forbidden liaison | `et` never liaises; h-aspiré (`les héros` /le e.ʁo/) | judge must NOT credit these consonants |
| enchaînement | `une amie` → /y.n‿a.mi/ | resyllabification — word boundaries move |
| schwa drop | `petite` /pə.tit/ → /ptit/ in flow | shortens the stream; optional, style-dependent |

Rules live in `sandhi_fr.py` with a word-level lexicon for latent consonants
and h-aspiré. Use it in three places: building training targets (§3 stage 1),
the DPO reward (§3 stage 2), and Agent C's judging stream — one phonology,
three consumers. The juncture-scoring work already on the liaison branch
slots in as empirical calibration for the optional rules (which liaisons
French TTS actually realizes).

## 5. Order of execution

1. Merge/port the liaison-branch juncture scorer; land `sandhi_fr.py` (done,
   this commit — pure Python, no deps, self-testing).
2. Build stage-2 juncture bigram data + stage-4 synthetic sentences.
3. SFT ByT5-small end-to-end to validate the P2G formulation (one evening on
   the 4090); scale to byt5-base.
4. Run the verifier DPO loop with the sandhi-aware reward.
5. Train the Qwen composer on (EN sentence + candidates → fluent FR line).
6. Wire both into `three_agent_v2.py`: transducer proposes → composer
   selects → sandhi → Agent B hears → Agent C two-comparison judge repairs.
