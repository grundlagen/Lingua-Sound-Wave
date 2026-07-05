# Qwen Fine-tuning Kit — Homophone Agent A

## What's here

| File | Purpose |
|------|---------|
| `agent_a_generative.py` | **Drop-in generative Agent A** for three_agent_v2.py |
| `build_sft_data.py` | ChatML SFT JSONL from dictionary v7 + phrase bank |
| `build_repair_data.py` | Multi-turn REVISE examples for the repair loop |
| `train_lora.py` | LoRA fine-tune Qwen3-4B on SFT data (24GB GPU) |
| `sample_and_dpo.py` | Best-of-16 sampling + DPO pair generation |
| `train_dpo.py` | DPO training on best/worst pairs |

## Quick start (what works NOW)

```bash
cd /home/mint/Lingua-Sound-Wave/research/qwen-finetune
python3 agent_a_generative.py
```

Output: French homophone candidates for test words. Uses 6,143-pair lookup DB
with fallback to nearest-match for unknown words.

## Full pipeline (needs GPU with internet)

```bash
# 1. Build training data from existing artifacts
HOMOPHONE_BENCH_DIR=../homophone-bench python3 build_sft_data.py

# 2. Build repair examples  
python3 build_repair_data.py

# 3. Train (on vast.ai RTX 4090 or similar)
python3 train_lora.py --base Qwen/Qwen2.5-1.5B-Instruct

# 4. Improve with DPO
python3 sample_and_dpo.py && python3 train_dpo.py

# 5. Use the trained model
python3 agent_a_generative.py   # now uses Qwen instead of char-LSTM
```

## Wiring into three_agent_v2.py

```python
# In three_agent_v2.py, replace:
#   a_picks = agent_A(current_words)
# With:
from agent_a_generative import GenerativeCarver
carver = GenerativeCarver()
picks, fr_text = carver.carve_phrase(" ".join(current_words))
```

## Environment

`HOMOPHONE_BENCH_DIR` — path to research/homophone-bench (default: auto-detected)
