"""Verifier loop: best-of-n sampling scored by the combo matcher → DPO pairs.

For each English input: sample Agent A n times at high temperature, score
every sample with reward = mean word-combo(EN, FR) + lambda * fluency(FR),
then emit
  - the best sample as a new SFT row (rejection sampling), and
  - (best, worst) as a DPO pair when the reward gap >= --min-gap.

Fluency uses bigram_lm.py from the bench dir if present, else 0.

Usage:
  python sample_and_dpo.py --adapter ckpt/agent-a-lora \
      --inputs data/en_inputs.txt --bench-dir ../homophone-bench \
      --n 16 --out-dir data/
Then:  python train_dpo.py --data data/agent_a_dpo.jsonl --adapter ckpt/agent-a-lora ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    SYSTEM_A,
    chatml_sft,
    load_g2p,
    load_matcher,
    user_prompt_a,
    write_jsonl,
    _load_module,
)


def load_fluency(bench_dir):
    try:
        mod = _load_module("bigram_lm", bench_dir)
        for attr in ("score", "logprob", "fluency"):
            if hasattr(mod, attr):
                fn = getattr(mod, attr)
                return lambda text: float(fn(text))
    except FileNotFoundError:
        pass
    print("WARNING: bigram_lm not found; fluency term = 0")
    return lambda text: 0.0


def reward_fn(combo_score, fluency, lam):
    def reward(en: str, fr: str) -> float:
        en_w, fr_w = en.split(), fr.split()
        if not fr_w:
            return -1.0
        # mean per-position combo over the shorter alignment; crude but
        # monotone with the line-level judge, and fast enough for n=16
        k = min(len(en_w), len(fr_w))
        combos = [combo_score(en_w[i], fr_w[i]) for i in range(k)]
        sound = sum(combos) / k if k else 0.0
        return sound + lam * fluency(fr)
    return reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--inputs", type=Path, required=True,
                    help="one English line per row")
    ap.add_argument("--bench-dir", type=Path, default=None)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--min-gap", type=float, default=0.15)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    combo_score = load_matcher(args.bench_dir)
    g2p = load_g2p(args.bench_dir)
    fluency = load_fluency(args.bench_dir)
    reward = reward_fn(combo_score, fluency, args.lam)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    sft_rows, dpo_rows = [], []
    lines = [l.strip() for l in args.inputs.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    for en in lines:
        try:
            ipa = g2p(en)
        except Exception:
            continue
        prompt_msgs = [
            {"role": "system", "content": SYSTEM_A},
            {"role": "user", "content": user_prompt_a(en, ipa)},
        ]
        text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                num_return_sequences=args.n,
                max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        samples = {
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in out
        }
        scored = sorted(((reward(en, fr), fr) for fr in samples if fr),
                        reverse=True)
        if not scored:
            continue
        (best_r, best_fr), (worst_r, worst_fr) = scored[0], scored[-1]
        sft_rows.append(chatml_sft(SYSTEM_A, user_prompt_a(en, ipa), best_fr))
        if best_r - worst_r >= args.min_gap and best_fr != worst_fr:
            dpo_rows.append({
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": best_fr}],
                "rejected": [{"role": "assistant", "content": worst_fr}],
            })
        print(f"{en!r}: best {best_r:.3f} {best_fr!r} | worst {worst_r:.3f}")

    n1 = write_jsonl(args.out_dir / "agent_a_bofn_sft.jsonl", sft_rows)
    n2 = write_jsonl(args.out_dir / "agent_a_dpo.jsonl", dpo_rows)
    print(f"rejection-sampled SFT rows: {n1}; DPO pairs: {n2}")


if __name__ == "__main__":
    main()
