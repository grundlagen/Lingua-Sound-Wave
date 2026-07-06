"""Train the ByT5 transducer: EN IPA stream -> French carve.

Byte-level seq2seq — no tokenizer to hide sound structure. Curriculum via
the `stage` field in the data (words first, then phrases, then lines):
pass --curriculum to train stages sequentially, otherwise shuffled together.

Deps: pip install "transformers>=4.45" datasets accelerate sentencepiece
Fits a 24 GB GPU at byt5-small/base; byt5-base recommended for the real run.

  python train_byt5.py --data byt5_train.jsonl --model google/byt5-small \
      --out ckpt/byt5-carver --curriculum
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default="google/byt5-small")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-src", type=int, default=256)
    ap.add_argument("--max-tgt", type=int, default=96)
    ap.add_argument("--curriculum", action="store_true",
                    help="train stage 1 -> 2 -> 3 sequentially")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    ds = load_dataset("json", data_files=str(args.data), split="train")

    def prep(batch):
        enc = tok(batch["src"], max_length=args.max_src, truncation=True)
        lab = tok(text_target=batch["tgt"], max_length=args.max_tgt,
                  truncation=True)
        enc["labels"] = lab["input_ids"]
        return enc

    stages = sorted(set(ds["stage"])) if args.curriculum else [None]
    for stage in stages:
        part = ds.filter(lambda r: r["stage"] == stage) if stage else ds
        part = part.shuffle(seed=13).map(prep, batched=True,
                                         remove_columns=part.column_names)
        cfg = Seq2SeqTrainingArguments(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            bf16=True,
            logging_steps=50,
            save_strategy="epoch",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            report_to="none",
        )
        trainer = Seq2SeqTrainer(
            model=model, args=cfg, train_dataset=part,
            data_collator=DataCollatorForSeq2Seq(tok, model=model),
        )
        print(f"=== training stage {stage or 'all'}: {len(part)} rows ===")
        trainer.train()

    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
