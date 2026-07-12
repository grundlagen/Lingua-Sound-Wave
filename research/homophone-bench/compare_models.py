#!/usr/bin/env python3
"""Compare LSTM (13M) vs Transformer (45M) homophone models."""
import torch, os, math

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Test words ──
tests = ["beauty","ocean","silent","wandered","twilight","river","forest",
         "mountain","thunder","whisper","shadow","dreamer","starlight",
         "moonlight","heartbeat","waterfall","sunrise","nightfall"]

# ── Load Transformer ──
print("Loading transformer...")
ts = torch.load("homophone_transformer.pt", map_location="cpu", weights_only=False)
ts_model = ts["model"]
ts_cfg = ts.get("config",{})
ts_src = ts.get("src_c2i", {}); ts_tgt_i2c = ts.get("tgt_i2c", {}); ts_max = ts.get("max_len",18)
ts_tgt_c2i = ts.get("tgt_c2i", {})
ts_ipa = ts.get("ipa_c2i", {})
params = sum(p.numel() for p in ts_model.values())
print(f"  Transformer: {params:,} params, epoch={ts.get('epoch',1)}, val_loss={ts.get('val_loss','?')}")

# ── Load LSTM ──
print("Loading LSTM...")
ls = torch.load("homophone_model_gpu.pt", map_location="cpu", weights_only=False)
ls_model = ls["model"]
ls_src = ls.get("src_c2i", {}); ls_tgt_i2c = ls.get("tgt_i2c", {}); ls_max = ls.get("max_len",16)
ls_tgt_c2i = ls.get("tgt_c2i", {})
lparams = sum(p.numel() for p in ls_model.values())
print(f"  LSTM:        {lparams:,} params")

# ── Carve function ──
def carve_transformer(word):
    tokens = [ts_src.get(c,0) for c in word]
    ipa_tok = [ts_ipa.get(c,0) for c in word]  # approximate
    tokens = [1]+tokens+[2]; ipa_tok = [1]+ipa_tok+[2]
    while len(tokens) < ts_max: tokens.append(0); ipa_tok.append(0)
    src_t = torch.tensor([tokens[:ts_max]])
    ipa_t = torch.tensor([ipa_tok[:ts_max]])
    # Use the saved model dict directly — just a forward pass
    return "?"  # need model class to instantiate

def carve_lstm(word):
    tokens = [1] + [ls_src.get(c,0) for c in word] + [2]
    while len(tokens) < ls_max: tokens.append(0)
    src = torch.tensor([tokens[:ls_max]])
    # Use run_gpu_model's H class
    import run_gpu_model
    return "?"  # same problem

print("\nTransformer model saved at epoch 1 (val_loss=0.649).")
print(f"Training still running on GPU — check later for epoch 150 model.")
print(f"\nFile: homophone_transformer.pt ({os.path.getsize('homophone_transformer.pt')/1e6:.0f}MB)")
print(f"Keys: {list(ts.keys())}")
