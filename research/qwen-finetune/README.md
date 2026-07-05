# Qwen Fine-tuning Kit — Homophone Agent A

Two implementations available:
- **Qwen-based** (`QwenCarver`): Full transformer pipeline trained via LoRA + DPO
- **Char-LSTM** (`CharLSTMCarver`): Lightweight character-level model, no GPU needed

## What's here

| File | Purpose |
|------|---------|
| `agent_a_generative.py` | **Drop-in generative Agent A** with both QwenCarver and CharLSTMCarver |
| `build_sft_data.py` | ChatML SFT JSONL from dictionary v7 + phrase bank |
| `build_repair_data.py` | Multi-turn REVISE examples for the repair loop |
| `train_lora.py` | LoRA fine-tune Qwen3-4B on SFT data (24GB GPU) |
| `sample_and_dpo.py` | Best-of-16 sampling + DPO pair generation |
| `train_dpo.py` | DPO training on best/worst pairs |
| `common.py` | prompts, bench-dir imports (matcher/g2p), JSONL helpers |

## Quick start (Char-LSTM, no GPU needed)

```bash
cd /home/mint/Lingua-Sound-Wave/research/qwen-finetune
python3 agent_a_generative.py
```

Output: French homophone candidates for test words. Uses 6,143-pair lookup DB
with fallback to nearest-match for unknown words.

## Full Qwen pipeline (needs GPU)

Implements the training path from `docs/qwen-three-agent-schema.md`. The
three-agent loop's architecture is proven (two-comparison judge, repair
targeting), but the lookup Agent A can't repair: the 6k-pair database lacks
French words that sound like most English targets. These scripts turn Agent A
into a generative Qwen model that *produces* novel French spellings, so the
repair loop can converge.

All scripts find your local pipeline (matcher, g2p, dictionary, phrase bank)
via `HOMOPHONE_BENCH_DIR` or `--bench-dir` pointing at
`research/homophone-bench`. Nothing here re-implements the scorer — the
AUC-0.993 combo matcher stays the single source of sound truth.

```bash
export HOMOPHONE_BENCH_DIR=~/Lingua-Sound-Wave/research/homophone-bench
pip install "transformers>=4.45" peft trl datasets accelerate bitsandbytes

# 1. SFT data from what you already have (dictionary v7, phrase bank, carves)
python build_sft_data.py --min-combo 0.45 --out-dir data/
#    -> data/agent_{a,b}_sft_{train,val}.jsonl
#    NOTE: pass --honest-tsv for Agent B (fr<TAB>heard_en, ~30% mix) or B
#    learns to report the intended English instead of what it hears.

# 2. Synthetic REVISE turns (corrupt one span of a good carve; fix = original)
python build_repair_data.py --out data/agent_a_repair_train.jsonl

# 3. LoRA SFT (single 24 GB GPU, 4-bit)
python train_lora.py --data data/agent_a_sft_train.jsonl \
    --extra-data data/agent_a_repair_train.jsonl \
    --val data/agent_a_sft_val.jsonl --out ckpt/agent-a-lora
python train_lora.py --data data/agent_b_sft_train.jsonl \
    --val data/agent_b_sft_val.jsonl --out ckpt/agent-b-lora

# 4. Verifier loop: best-of-16 scored by combo + fluency -> SFT rows + DPO pairs
python sample_and_dpo.py --adapter ckpt/agent-a-lora \
    --inputs data/en_inputs.txt --n 16 --out-dir data/
python train_dpo.py --data data/agent_a_dpo.jsonl \
    --adapter ckpt/agent-a-lora --out ckpt/agent-a-dpo
# iterate 3-4 two or three times: fold agent_a_bofn_sft.jsonl back in, resample

# 5. Smoke-test generation
python agent_a_generative.py --adapter ckpt/agent-a-dpo the ocean remembers
```

## Wiring into three_agent_v2.py

### QwenCarver (trained model)

`agent_a_generative.QwenCarver` is the drop-in candidate provider:

- **carve(en_span)** — call where the lookup misses (keep the lookup first;
  it's free and verified). Re-verify every generated candidate with the combo
  matcher before accepting — generation is a proposal, never a verdict.
- **revise(...)** — call from the repair branch with Agent C's span info
  (`en_span`, `heard`, `span_combo`, candidate shortlist). This is trained on
  the synthetic REVISE distribution from step 2, so it keeps the good words
  and fixes only the flagged span.

Every loop-accepted line (sound_combo ≥ 0.55) goes back into the SFT pool —
the verifier-filtered flywheel.

### CharLSTMCarver (lightweight)

```python
from agent_a_generative import CharLSTMCarver
carver = CharLSTMCarver()
picks, fr_text = carver.carve_phrase(" ".join(current_words))
```

## Environment

`HOMOPHONE_BENCH_DIR` — path to research/homophone-bench (default: auto-detected)
