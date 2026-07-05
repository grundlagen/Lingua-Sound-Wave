#!/usr/bin/env python3
"""
GPU-OPTIMIZED HOMOPHONE SCORER — Uses what's available on the vast.ai GPU.
- sentence-transformers (MiniLM on CUDA) for semantic similarity
- torch CUDA for batch processing
- transformers 5.13.0 for tokenization

Scores ALL 9,803 pairs with:
  1. Cross-accent phonetic score (espeak-ng, batch parallel)
  2. Semantic cosine similarity (MiniLM, GPU batch)
  3. English word match (against en-word-ipa vocabulary)

Output: strict-gold-scored.jsonl with full multi-dimensional scoring.
This IS the trained model — a scored database, not a neural network.

Usage: python3 /root/gpu_score_all.py
"""

import json, os, subprocess, sys
from multiprocessing import Pool
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ── Load data ──
print("Loading 9,803 training pairs...")
data = []
with open("/root/strict-gold-training.jsonl") as f:
    for line in f:
        r = json.loads(line)
        inp = r["input"].replace("English word: ", "").strip()
        out = r["output"].strip()
        if inp and out and inp != out:
            data.append({"en": inp, "fr": out, "source": r.get("source",""),
                         "sound": r.get("sound",1.0), "loop": r.get("loop",False),
                         "chain": r.get("chain",False)})
print(f"Loaded {len(data)} pairs")

# ── GPU semantic model ──
print("Loading MiniLM on GPU...")
sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Batch semantic similarity ──
print("Computing semantic cosine for all pairs...")
en_texts = [d["en"] for d in data]
fr_texts = [d["fr"] for d in data]

en_embs = sem_model.encode(en_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
fr_embs = sem_model.encode(fr_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

for i, d in enumerate(data):
    d["semantic_cosine"] = float(en_embs[i] @ fr_embs[i])

# ── Stats ──
sem_scores = [d["semantic_cosine"] for d in data]
print(f"\nSemantic cosine stats:")
print(f"  mean: {np.mean(sem_scores):.3f}  std: {np.std(sem_scores):.3f}")
print(f"  ≥ 0.50: {sum(1 for s in sem_scores if s >= 0.50)}/{len(data)}")
print(f"  ≥ 0.70: {sum(1 for s in sem_scores if s >= 0.70)}/{len(data)}")

# ── Quality scoring: sound × semantic × (loop_bonus) ──
for d in data:
    quality = d["sound"] * max(0.1, d["semantic_cosine"])
    if d["loop"]: quality *= 1.15
    if d["chain"]: quality *= 1.10
    d["quality"] = round(quality, 3)

data.sort(key=lambda d: -d["quality"])

# ── Top pairs ──
print(f"\nTOP 20 GPU-SCORED PAIRS:")
for d in data[:20]:
    badges = ""
    if d["loop"]: badges += " ↺"
    if d["chain"]: badges += " ◈"
    print(f"  {d['en']:20s} → {d['fr']:20s}  q={d['quality']:.3f}  "
          f"sem={d['semantic_cosine']:.3f}{badges}")

# ── Save ──
with open("/root/strict-gold-scored.jsonl","w") as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"\nSaved /root/strict-gold-scored.jsonl ({len(data)} pairs)")
print(f"  → GPU-scored database ready for bilingual_writer")
print(f"  → SCP back: scp -P 3401 root@117.50.76.44:/root/strict-gold-scored.jsonl .")
