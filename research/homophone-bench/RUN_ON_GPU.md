# GPU runbook — for a terminal Claude session with GPU access

_Everything below is prepared; the box only needs one GPU (T4 is enough) and
`pip install transformers trl datasets accelerate sentencepiece`. Work from
`research/homophone-bench/`. In a terminal-Claude session, Claude itself
replaces Haiku at the proposer/fixer stages (see §4)._

## 1. Train the dual translator (the corpus is ready)

```bash
python build_train_corpus.py                    # -> train-dual-v1.jsonl (168k rows, 40s)
cd selflearn
python train_selflearn.py --base Qwen/Qwen2.5-1.5B-Instruct \
    --data ../train-dual-v1.jsonl --rounds 4 --eval_llm \
    --ckpt_dir /path/to/persistent/ckpt        # checkpoint/resume built in
```
Expert iteration is inside: best-of-N sampling scored by the local reward
(prosody × meaning), self-distilled each round. Status pushes to the
`selflearn-status` branch if GITHUB_TOKEN+GITHUB_REPO are set.

## 2. TRUE constrained decoding (E40 full form — the endgame)

```bash
pip install llama-cpp-python                    # or vllm
# any small instruct GGUF (Qwen2.5-1.5B/7B-instruct)
```
Recipe (30 lines of glue; METHODS_DEEP_DIVE §"Fuse them" has the design):
at each decode step, call `matcher` forward over the remaining EN phoneme
stream to find which next-tokens keep ≥1 sound-legal path alive, and pass that
set as `logit_bias`/allowed-tokens. The LLM supplies French fluency+meaning;
the matcher supplies the hard sound mask. `constrained_poet.py` is the
phrase-granular prototype to port.

## 3. Scale the audio validation / ASR-confusion miner

```bash
pip install torchaudio
python real_audio_g2p.py --n 300               # more Tatoeba clips
# then the miner: decode FRENCH audio with an ENGLISH wav2vec2 -- its
# "hallucinations" are acoustically-discovered homophones (METHODS_DEEP_DIVE F1)
```

## 4. Claude-in-the-loop (replacing Haiku at the weak stages)

In a terminal session the running Claude does these jobs directly — no API
calls, interactive, judged by the same verifiers:
- **paraphrase proposer** (`paraphrase_search.P_FR/P_EN`): stronger, more
  varied paraphrases than Haiku's;
- **sound-bender** (`P_REFINE`): Claude knows IPA — ask it to bend wording
  toward a target phoneme stream, verify with `combo`;
- **grammar fixer** (`dual_poet.haiku_fix` prompt): sound-preserving repair;
- **judge arbitration** on strict_judge disagreement cases.
The law stays: Claude proposes, `matcher`/`semantic_cosine` dispose.

## 5. Quality queue (from METHODS_STATUS.md)
1. per-channel logistic calibration on strict-gold (closes 50%→55%+ composer gap)
2. paragraph set-cover (`set_dual.py`) fused with paraphrase_search
3. window-index speed tier (82k units — pre-bucket by first phoneme)
4. rhyme-scheme composition over `rhyme_pick.py` families
