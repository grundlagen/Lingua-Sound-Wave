# Self-learning homophonic carver — run on a GPU

A model that teaches itself to rewrite English as same-sounding French, scored by
our **reward** (stress-weighted prosody sound-match × French-validity) — no human
labels. SFT warm-start on `train-homophonic.jsonl`, then best-of-N
self-improvement (expert iteration): generate → score → train on its own bests →
repeat.

## There is no GPU in the Claude sandbox

This repo runs CPU-only here, so training happens on **your** GPU. The reward
(`reward.py`) and judge (`prosody.py`, `fr_coherence.py`) are CPU/​API and tested;
`train_selflearn.py` is the GPU piece.

## Where to get a GPU (cheapest → easiest)

| option | GPU | cost | notes |
|---|---|---|---|
| **Google Colab** | T4 (free) / L4 (Pro) | free–$10/mo | easiest; upload the repo, `pip install transformers trl`, run |
| **Kaggle** | P100 / T4×2 | free (30h/wk) | free, persistent-ish |
| **RunPod / Vast.ai** | RTX 3090/4090 | ~$0.2–0.4/hr | on-demand box, full control |
| **Modal** | A10/A100 serverless | free credits | wrap `train_selflearn.py` in a Modal fn |
| **Together / Fireworks** | managed fine-tune | per-token | NO loop — just SFT our JSONL (no self-learning) |

You have an **NVIDIA NIM** key (inference) and **OpenRouter** (inference) — those
run Nemotron for the *judge*, but cannot fine-tune. For weights you need one of
the above. Simplest first run: **Colab T4 + Qwen2.5-1.5B-Instruct**.

## Run

```bash
# on the GPU box, inside research/homophone-bench/
pip install transformers trl datasets accelerate wordfreq panphon
sudo apt-get install -y espeak-ng           # the matcher/prosody need it
cd selflearn
python train_selflearn.py --base Qwen/Qwen2.5-1.5B-Instruct --rounds 4 --k 8
```

- `--base` any small instruct model that knows some French (Qwen2.5-1.5B,
  Llama-3.2-1B, gemma-2-2b).
- `--k` samples per phrase (more = better bests, more compute).
- `--keep_thresh` reward floor for a self-sample to become a training target.

Periodically check progress with the LLM judge:
`python ../bank_composer.py --llm` (uses your OpenRouter/Nemotron key).

## Why this is "self-learning"

The model improves against a reward it can compute itself — it is its own teacher.
The reward is the linguistics we encoded: prosodic (stress-weighted) sound match +
real-French-word validity. As it learns, raise `--keep_thresh` so it chases ever
better carves. Swap the local reward's French-validity term for the distilled LLM
judge (label a batch with `fr_coherence`, train a tiny regressor) to make the
teacher smarter without per-sample API calls.

## Disconnect-proofing, monitoring & admin

- **Checkpoint + resume:** `--ckpt_dir <Drive path>` saves the model + `status.json`
  each round. If Colab drops, re-run the cells — it resumes from the last round.
- **Keep-alive:** the notebook arms a best-effort JS heartbeat (keep the tab open;
  Colab Pro is more reliable for multi-hour runs).
- **Remote monitoring:** set `GITHUB_TOKEN` + `GITHUB_REPO`; each round the trainer
  PUTs `selflearn/status.json` (round, kept, mean reward, sample carves + Nemotron
  scores) to a **`selflearn-status`** branch via the API. Claude reads that branch
  on request and advises / tunes (`--keep_thresh`, `--k`, base model) by pushing to
  `claude/phrase-weave-multiword`; re-run the clone cell to pull and continue.
- **Per-round eval:** `--eval_llm` scores each round's bests with the live judge so
  you see real French-coherence climbing, not just the local reward.
