#!/usr/bin/env python3
"""
GPU CHARACTER-LEVEL HOMOPHONE MODEL — Train on vast.ai RTX 4090.
Learns EN→FR character transformations from 9,803 strict-gold pairs.

Architecture: 3-layer LSTM encoder-decoder with attention.
Hidden: 512, Embedding: 256. ~5M parameters.
Training: 50 epochs, batch 128, runs in ~3 min on RTX 4090.

Usage: python3 gpu_char_model.py
Input:  strict-gold-training.jsonl (9,803 pairs)
Output: homophone_model_gpu.pt (portable, ~20MB)
"""

import json, os, time, math
import torch
import torch.nn as nn
import torch.optim as optim

DATA = "strict-gold-training.jsonl"
OUTPUT = "homophone_model_gpu.pt"

# ── Load data ──
print(f"Loading {DATA}...")
pairs = []
with open(DATA) as f:
    for line in f:
        r = json.loads(line)
        en = r["input"].replace("English word: ", "").strip().lower()
        fr = r["output"].strip().lower()
        if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
            pairs.append((en, fr))

print(f"  {len(pairs)} training pairs")
print(f"  Sample: {pairs[0][0]}→{pairs[0][1]}  {pairs[1000][0]}→{pairs[1000][1]}")

# ── Character vocabularies ──
SRC_C2I = {"<pad>":0, "<sos>":1, "<eos>":2}
TGT_C2I = {"<pad>":0, "<sos>":1, "<eos>":2}
for en, fr in pairs:
    for c in en: SRC_C2I.setdefault(c, len(SRC_C2I))
    for c in fr: TGT_C2I.setdefault(c, len(TGT_C2I))
TGT_I2C = {i:c for c,i in TGT_C2I.items()}

SRC_V = len(SRC_C2I); TGT_V = len(TGT_C2I)
MAX_LEN = 16
print(f"  Vocab: {SRC_V} src, {TGT_V} tgt, max_len={MAX_LEN}")

# ── Tokenize ──
def encode(text, c2i, max_len):
    tokens = [c2i["<sos>"]] + [c2i.get(c,0) for c in text] + [c2i["<eos>"]]
    if len(tokens) < max_len: tokens += [0]*(max_len - len(tokens))
    return torch.tensor(tokens[:max_len], dtype=torch.long)

X = torch.stack([encode(en, SRC_C2I, MAX_LEN) for en,_ in pairs])
Y = torch.stack([encode(fr, TGT_C2I, MAX_LEN) for _,fr in pairs])

# Train/val split
perm = torch.randperm(len(pairs))
split = int(0.9 * len(pairs))
X_tr, Y_tr = X[perm[:split]], Y[perm[:split]]
X_vl, Y_vl = X[perm[split:]], Y[perm[split:]]
print(f"  Train: {len(X_tr)}, Val: {len(X_vl)}")

# ── Model ──
class HomophoneLSTM(nn.Module):
    def __init__(self, src_v, tgt_v, hidden=512, embed=256):
        super().__init__()
        self.src_embed = nn.Embedding(src_v, embed, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_v, embed, padding_idx=0)
        self.encoder = nn.LSTM(embed, hidden, num_layers=3, batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(embed + hidden, hidden, num_layers=3, batch_first=True, dropout=0.1)
        self.attn_W = nn.Linear(hidden, hidden)
        self.enc_to_attn = nn.Linear(hidden, embed)  # project enc hidden→embed for attention
        self.out = nn.Linear(hidden, tgt_v)
        self.hidden = hidden; self.embed = embed

    def forward(self, src, tgt, tf_ratio=0.5):
        B = src.size(0)
        src_emb = self.src_embed(src)
        enc_out, (h, c) = self.encoder(src_emb)
        
        dec_in = tgt[:, 0:1]
        outputs = []
        for t in range(1, tgt.size(1)):
            dec_emb = self.tgt_embed(dec_in)
            # Project encoder output to embed dim for attention
            enc_proj = self.enc_to_attn(enc_out)
            scores = torch.bmm(dec_emb, enc_proj.transpose(1,2))
            attn = torch.softmax(scores / math.sqrt(self.embed), dim=-1)
            ctx = torch.bmm(attn, enc_out)
            dec_input = torch.cat([dec_emb, ctx], dim=-1)
            dec_out, (h, c) = self.decoder(dec_input, (h, c))
            logits = self.out(dec_out)
            outputs.append(logits)
            if torch.rand(1).item() < tf_ratio:
                dec_in = tgt[:, t:t+1]
            else:
                dec_in = logits.argmax(-1)
        return torch.cat(outputs, dim=1)

    def generate(self, src, max_len=15):
        with torch.no_grad():
            src_emb = self.src_embed(src)
            enc_out, (h, c) = self.encoder(src_emb)
            dec_in = torch.tensor([[TGT_C2I["<sos>"]]], device=src.device)
            result = []
            for _ in range(max_len):
                dec_emb = self.tgt_embed(dec_in)
                enc_proj = self.enc_to_attn(enc_out)
                scores = torch.bmm(dec_emb, enc_proj.transpose(1,2))
                attn = torch.softmax(scores / math.sqrt(self.embed), dim=-1)
                ctx = torch.bmm(attn, enc_out)
                dec_out, (h,c) = self.decoder(torch.cat([dec_emb,ctx],-1), (h,c))
                tok = self.out(dec_out).argmax(-1).item()
                if tok == TGT_C2I["<eos>"]: break
                c = TGT_I2C.get(tok, "?")
                if c not in ("<pad>","<sos>"): result.append(c)
                dec_in = torch.tensor([[tok]], device=src.device)
            return "".join(result)

# ── Train ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

model = HomophoneLSTM(SRC_V, TGT_V).to(device)
opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
crit = nn.CrossEntropyLoss(ignore_index=0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

EPOCHS = 50; BATCH = 128
best_val = float("inf")

print(f"Training {EPOCHS} epochs, batch={BATCH}...")
t0 = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0; perm = torch.randperm(len(X_tr))
    for i in range(0, len(X_tr), BATCH):
        idx = perm[i:i+BATCH]
        src, tgt = X_tr[idx].to(device), Y_tr[idx].to(device)
        opt.zero_grad()
        out = model(src, tgt, tf_ratio=max(0.2, 0.5*(1-epoch/EPOCHS)))
        loss = crit(out.reshape(-1, TGT_V), tgt[:,1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    
    sched.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        src_v, tgt_v = X_vl[:200].to(device), Y_vl[:200].to(device)
        out_v = model(src_v, tgt_v, tf_ratio=0)
        val_loss = crit(out_v.reshape(-1, TGT_V), tgt_v[:,1:].reshape(-1)).item()
    
    if val_loss < best_val:
        best_val = val_loss
        torch.save({"model": model.state_dict(), "src_c2i": SRC_C2I, "tgt_c2i": TGT_C2I,
                     "tgt_i2c": TGT_I2C, "max_len": MAX_LEN}, OUTPUT)
    
    if (epoch+1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:3d}: train_loss={total_loss/len(X_tr):.4f}  "
              f"val_loss={val_loss:.4f}  time={elapsed:.0f}s")

print(f"\n  Best val_loss={best_val:.4f}  saved → {OUTPUT}")

# ── Test ──
model.load_state_dict(torch.load(OUTPUT, map_location=device, weights_only=False)["model"])
model.eval()

print(f"\n{'='*50}")
print(f"TESTING TRAINED MODEL")
print(f"{'='*50}")

tests = ["beauty", "ocean", "silent", "wandered", "twilight", "river", "forest",
         "mountain", "thunder", "whisper", "shadow", "dreamer", "starlight"]
for w in tests:
    tokens = [SRC_C2I.get(c,0) for c in w]
    tokens = [SRC_C2I["<sos>"]] + tokens + [SRC_C2I["<eos>"]]
    if len(tokens) < MAX_LEN: tokens += [0]*(MAX_LEN - len(tokens))
    src = torch.tensor([tokens[:MAX_LEN]], device=device)
    fr = model.generate(src)
    print(f"  {w:15s} → {fr}")
