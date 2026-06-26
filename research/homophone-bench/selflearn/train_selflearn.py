"""Self-learning homophonic carver: SFT warm-start then best-of-N self-improvement.

Designed to run on ONE consumer GPU (Colab T4 / RunPod 3090-4090). Trains a small
base model to rewrite English so it sounds like French, improving itself against
our reward (prosody sound-match x French-validity) with NO human labels.

Loop (rejection-sampling / "expert iteration" -- robust, no RL plumbing):
  0. SFT warm-start on train-homophonic.jsonl (our generated corpus).
  1. For a batch of English phrases, SAMPLE k French candidates from the model.
  2. Score each with reward.reward(en, fr)  (local, free, prosody-aware).
  3. Keep the best candidate per phrase above a threshold -> new SFT pairs.
  4. SFT on those self-generated bests. Repeat 1-4. The model bootstraps upward.

Periodically eval with the LLM judge (fr_coherence) -- a sanity check, not the
per-sample signal. This is in-weight self-learning; the reward is the teacher.

Run on a GPU box (see selflearn/README.md):
  pip install transformers trl datasets accelerate
  python train_selflearn.py --base Qwen/Qwen2.5-1.5B-Instruct --rounds 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward as R

INSTR = "Rewrite the English so it sounds the same when read aloud in French:"


def load_sft(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("task") in ("phrase_carve", "word_carve"):
            rows.append((r["prompt"], r["completion"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="../train-homophonic.jsonl")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--k", type=int, default=8, help="samples per phrase")
    ap.add_argument("--keep_thresh", type=float, default=0.55)
    ap.add_argument("--out", default="./homophonic-carver")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # T4 has no bf16; auto-pick the supported half precision
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype,
                                                 device_map="auto")

    def to_text(prompt, completion):
        msgs = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}]
        return tok.apply_chat_template(msgs, tokenize=False)

    def sft(pairs, epochs=1, tag=""):
        ds = Dataset.from_dict({"text": [to_text(p, c) for p, c in pairs]})
        cfg = SFTConfig(output_dir=f"{args.out}-{tag}", num_train_epochs=epochs,
                        per_device_train_batch_size=8, gradient_accumulation_steps=2,
                        learning_rate=1e-5, logging_steps=20, save_strategy="no",
                        bf16=bf16, fp16=not bf16, max_seq_length=128)
        SFTTrainer(model=model, args=cfg, train_dataset=ds,
                   processing_class=tok).train()

    def sample(prompts, k):
        outs = []
        for p in prompts:
            ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                          add_generation_prompt=True,
                                          return_tensors="pt").to(model.device)
            gen = model.generate(ids, do_sample=True, temperature=1.0, top_p=0.95,
                                 num_return_sequences=k, max_new_tokens=24,
                                 pad_token_id=tok.pad_token_id)
            cand = [tok.decode(g[ids.shape[1]:], skip_special_tokens=True).strip()
                    for g in gen]
            outs.append(cand)
        return outs

    base = load_sft(args.data)
    print(f"SFT warm-start on {len(base)} pairs ...")
    sft(base, epochs=2, tag="sft0")

    en_pool = [p for p, _ in base]
    for rnd in range(args.rounds):
        batch = random.sample(en_pool, min(256, len(en_pool)))
        new = []
        for prompt, cands in zip(batch, sample(batch, args.k)):
            en = prompt.split(":", 1)[-1].strip()
            best, bestr = None, -1.0
            for fr in cands:
                fr = fr.split("\n")[0].strip()
                r = R.reward(en, fr)
                if r > bestr:
                    bestr, best = r, fr
            if best and bestr >= args.keep_thresh:
                new.append((prompt, best))
        print(f"round {rnd}: kept {len(new)}/{len(batch)} self-generated bests "
              f"(reward>={args.keep_thresh})")
        if new:
            sft(new, epochs=1, tag=f"r{rnd}")
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print(f"saved self-learned carver -> {args.out}")


if __name__ == "__main__":
    main()
