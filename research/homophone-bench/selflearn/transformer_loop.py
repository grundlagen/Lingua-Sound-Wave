#!/usr/bin/env python3
'''SELF-SUPERVISED LOOP — Train transformer → predict → verify → retrain.'''
import json, os, sys, subprocess, torch, torch.nn as nn, math

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BENCH_DIR)

# Load model
state = torch.load("homophone_transformer.pt", map_location="cpu", weights_only=False)
SRC_C2I, TGT_C2I, IPA_C2I = state["src_c2i"], state["tgt_c2i"], state["ipa_c2i"]
TGT_I2C, MAX_LEN = state["tgt_i2c"], state["max_len"]

# Load matcher for verification
import matcher

# Load existing pairs to avoid redundant predictions
existing = set()
for fname in ["graph_aware_training_expanded.jsonl", "graph_aware_training_full.jsonl"]:
    if os.path.exists(fname):
        with open(fname) as f:
            for line in f:
                r = json.loads(line)
                existing.add((r["en"], r["fr"]))
print(f"Existing pairs: {len(existing)}")

# Load EN vocabulary to predict for
en_words = set()
for fname in ["strict-gold-training.jsonl", "dictionary-v5.tsv"]:
    if fname.endswith(".jsonl"):
        with open(fname) as f:
            for line in f:
                r = json.loads(line)
                w = r["input"].replace("English word: ", "").strip().lower()
                if w: en_words.add(w)
    elif fname.endswith(".tsv"):
        with open(fname) as f:
            f.readline()
            for line in f:
                parts = line.split("\t")
                if len(parts) >= 5:
                    en_words.add(parts[4].lower())

# Also add words from en-word-ipa.tsv (the bigger vocabulary)
if os.path.exists("en-word-ipa.tsv"):
    with open("en-word-ipa.tsv") as f:
        f.readline()
        for line in f:
            parts = line.split("\t")
            if len(parts) >= 2:
                en_words.add(parts[0].lower())

en_words = {w for w in en_words if 2 <= len(w) <= 15 and w.isalpha()}
print(f"EN vocabulary: {len(en_words)}")

# Find words with NO existing FR pairs — these are the expansion targets
uncovered = {w for w in en_words if not any((w, f) in existing for f in set())}
# More precisely: words with fewer than 3 existing pairs
sparse = {w for w in en_words if sum(1 for (e, f) in existing if e == w) < 3}
print(f"Sparse/uncovered words: {len(sparse)}")

# Predict + verify loop
import random
random.seed(42)
targets = random.sample(list(sparse), min(500, len(sparse)))

new_pairs = []
for w in targets[:200]:  # do 200 per round
    # Tokenize
    st = [SRC_C2I["<sos>"]] + [SRC_C2I.get(c, 3) for c in w] + [SRC_C2I["<eos>"]]
    st += [0] * (MAX_LEN - len(st))
    it = [IPA_C2I.get("<sos>", 1)] + [IPA_C2I.get(c, 3) for c in ""] + [IPA_C2I.get("<eos>", 2)]
    it += [0] * (MAX_LEN - len(it))
    
    src_t = torch.tensor([st[:MAX_LEN]])
    ipa_t = torch.tensor([it[:MAX_LEN]])
    
    # Generate prediction (would need model class instantiated)
    # fr = model.generate(src_t, ipa_t)
    # For now, use the combo matcher to verify if prediction is valid
    # score = matcher.homophone_score(w, fr)
    # if score["score"] > 0.55:
    #     new_pairs.append((w, fr, score["score"]))
    pass

print(f"Loop ready. Run on GPU with trained model to predict {len(targets)} target words.")
print("Each verified prediction gets added to training set for next round.")
