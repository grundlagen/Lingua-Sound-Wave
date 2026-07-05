"""LoRA SFT for Agent A or B on Qwen (transformers + peft + trl).

Deps: pip install "transformers>=4.45" peft trl datasets accelerate bitsandbytes

Usage:
  python train_lora.py --data data/agent_a_sft_train.jsonl \
      --extra-data data/agent_a_repair_train.jsonl \
      --val data/agent_a_sft_val.jsonl \
      --model Qwen/Qwen3-4B-Instruct-2507 --out ckpt/agent-a-lora

Runs on a single 24 GB GPU with 4-bit quantization (default); drop
--load-4bit on bigger cards.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--extra-data", type=Path, action="append", default=[])
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--load-4bit", action="store_true", default=True)
    ap.add_argument("--no-load-4bit", dest="load_4bit", action="store_false")
    args = ap.parse_args()

    import torch
    from datasets import concatenate_datasets, load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    files = [str(args.data)] + [str(p) for p in args.extra_data]
    train_ds = concatenate_datasets(
        [load_dataset("json", data_files=f, split="train") for f in files]
    ).shuffle(seed=13)
    eval_ds = (load_dataset("json", data_files=str(args.val), split="train")
               if args.val else None)

    quant = None
    if args.load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    cfg = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        # only train on assistant turns — repair examples have mid-dialogue
        # assistant turns that must all contribute loss
        assistant_only_loss=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
