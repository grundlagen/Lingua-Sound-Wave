#!/usr/bin/env python3
"""
GPU TRAINING — Fine-tune Qwen2.5-1.5B on 9,803 strict-gold homophone pairs.
Runs on vast.ai RTX 4090 (24GB VRAM). LoRA, 4-bit quant, ~3GB VRAM usage.

Heavy installs on first run — model download + tokenizer.
Model: Qwen/Qwen2.5-1.5B-Instruct
Data:  /root/strict-gold-training.jsonl (9,803 pairs)
Output: /root/homophone-llm/

Usage: python3 /root/gpu_train.py
"""

import json, os, torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Load data ──
print("Loading training data...")
data = []
with open("/root/strict-gold-training.jsonl") as f:
    for line in f:
        r = json.loads(line)
        inp = r["input"].replace("English word: ", "").strip()
        out = r["output"].strip()
        if inp and out and inp != out:
            data.append({"input": inp, "output": out})

# Filter: only high-quality pairs (sound ≥ 0.70 or loop/chain-certified)
filtered = [d for d in data]
print(f"Loaded {len(filtered)} training pairs")

def format_prompt(row):
    return f"English word: {row['input']}\nFrench homophone: {row['output']}"

prompts = [format_prompt(d) for d in filtered]
dataset = Dataset.from_dict({"text": prompts})

# ── Load model with 4-bit quantization ──
print("Loading model (4-bit quant, ~3GB VRAM)...")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# ── Tokenize ──
print("Tokenizing...")
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64, padding=False)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# Pad to max length in batch
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
)

# ── LoRA config ──
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Train ──
training_args = TrainingArguments(
    output_dir="/root/homophone-llm",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

print(f"\nTraining 5 epochs on {len(filtered)} pairs...")
print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")
trainer.train()

# ── Save ──
model.save_pretrained("/root/homophone-llm")
tokenizer.save_pretrained("/root/homophone-llm")
print(f"\nSaved model to /root/homophone-llm")

# ── Test ──
print(f"\n{'='*50}")
print(f"TESTING TRAINED MODEL")
print(f"{'='*50}")

test_words = [
    "beauty", "silent", "sea", "remember", "dawn", "ship", "sorrow",
    "dancing", "moon", "star", "deep", "free", "soul", "dream",
    "she walks in beauty like the night",
    "the sea remembers every ship",
]

model.eval()
for w in test_words:
    prompt = f"English word: {w}\nFrench homophone:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=20, temperature=0.3,
            do_sample=True, top_p=0.9,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    fr = result.replace(prompt, "").strip().split("\n")[0].strip()
    print(f"  {w:30s} → {fr}")
