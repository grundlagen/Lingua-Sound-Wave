#!/usr/bin/env python3
"""
TRANSFORMER HOMOPHONE MODEL — Graph-aware, multi-task, proper deep learning.

Architecture:
  - 6-layer transformer encoder (EN chars + EN IPA)
  - 6-layer transformer decoder → FR chars
  - Multi-task heads: sound quality regression, tier classification, loop detection
  - Graph-aware loss weighting by hops, meaning_proximity, and loop_certified
  - Proper training: 150 epochs, cosine annealing, gradient clipping, early stopping

Training data: graph_aware_training.jsonl (built by build_graph_training_data.py)

Usage:
    python train_transformer.py                     # Train (needs GPU, ~2-4 hours)
    python train_transformer.py --quick             # Fast test (1000 rows, 10 epochs)
    python train_transformer.py --resume ckpt.pt    # Resume from checkpoint

Output: homophone_transformer.pt (~80-150MB)
"""

import json, os, sys, math, time, argparse, random
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Config ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", default=True)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--nhead", type=int, default=8)
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--resume", default=None)
parser.add_argument("--output", default="homophone_transformer.pt")
parser.add_argument("--data", default="graph_aware_training.jsonl")
args = parser.parse_args()

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Build training data if needed ───────────────────────────────────
if not os.path.exists(args.data):
    print(f"[!] {args.data} not found. Building from build_graph_training_data.py...")
    import subprocess
    subprocess.run([sys.executable, "build_graph_training_data.py"], check=True)

# ── 2. Load and tokenize ───────────────────────────────────────────────
print(f"\n[1/5] Loading {args.data}...")
raw_data = []
with open(args.data) as f:
    for line in f:
        r = json.loads(line)
        en, fr = r["en"], r["fr"]
        if 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
            raw_data.append(r)

if args.quick:
    random.shuffle(raw_data)
    raw_data = raw_data[:1000]
    args.epochs = 10
    args.num_layers = 3
    args.d_model = 256
    args.nhead = 4

print(f"  {len(raw_data)} training rows")

# Build vocabularies
SRC_C2I = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
TGT_C2I = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
IPA_C2I = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}

for r in raw_data:
    for c in r["en"]: SRC_C2I.setdefault(c, len(SRC_C2I))
    for c in r["fr"]: TGT_C2I.setdefault(c, len(TGT_C2I))
    for c in r.get("en_ipa", ""): IPA_C2I.setdefault(c, len(IPA_C2I))

TGT_I2C = {i: c for c, i in TGT_C2I.items()}
SRC_V, TGT_V, IPA_V = len(SRC_C2I), len(TGT_C2I), len(IPA_C2I)
MAX_LEN = 18
print(f"  Vocab: {SRC_V} src, {IPA_V} ipa, {TGT_V} tgt, max_len={MAX_LEN}")

# Tier → index mapping
TIER_TO_IDX = {"S": 0, "A": 1, "B_safe": 2, "B_reservoir": 3, "B": 4}

# ── 3. Dataset ──────────────────────────────────────────────────────────
class HomophoneDataset(Dataset):
    def __init__(self, data, src_c2i, tgt_c2i, ipa_c2i, max_len):
        self.data = data
        self.src_c2i = src_c2i
        self.tgt_c2i = tgt_c2i
        self.ipa_c2i = ipa_c2i
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        # Source chars
        src = [self.src_c2i["<sos>"]] + [self.src_c2i.get(c, 3) for c in r["en"]] + [self.src_c2i["<eos>"]]
        src += [0] * (self.max_len - len(src))
        src = src[:self.max_len]

        # Source IPA
        ipa = [self.ipa_c2i["<sos>"]] + [self.ipa_c2i.get(c, 3) for c in r.get("en_ipa", "")] + [self.ipa_c2i["<eos>"]]
        ipa += [0] * (self.max_len - len(ipa))
        ipa = ipa[:self.max_len]

        # Target
        tgt = [self.tgt_c2i["<sos>"]] + [self.tgt_c2i.get(c, 3) for c in r["fr"]] + [self.tgt_c2i["<eos>"]]
        tgt += [0] * (self.max_len - len(tgt))
        tgt = tgt[:self.max_len]

        # Labels
        tier_idx = TIER_TO_IDX.get(r.get("tier", "B"), 4)
        score = float(r.get("score", 0.5))
        loop = 1.0 if r.get("loop_certified", False) else 0.0
        hops = float(r.get("graph_hops", 0))
        proximity = float(r.get("meaning_proximity", 1.0))

        # Loss weight: S-tier and loop-certified pairs matter more
        tier_w = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.8, 4: 1.0}.get(tier_idx, 1.0)
        graph_w = 1.0 + 0.5 * loop + 0.2 * proximity - 0.05 * hops
        weight = max(0.5, tier_w * graph_w)

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(ipa, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
            torch.tensor(tier_idx, dtype=torch.long),
            torch.tensor(score, dtype=torch.float32),
            torch.tensor(loop, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )

# Split
perm = torch.randperm(len(raw_data))
split = int(0.9 * len(raw_data))
train_ds = HomophoneDataset([raw_data[i] for i in perm[:split]], SRC_C2I, TGT_C2I, IPA_C2I, MAX_LEN)
val_ds = HomophoneDataset([raw_data[i] for i in perm[split:]], SRC_C2I, TGT_C2I, IPA_C2I, MAX_LEN)
train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
print(f"  Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

# ── 4. Model: Transformer with Multi-Task Heads ─────────────────────────
print(f"\n[2/5] Building transformer (d={args.d_model}, layers={args.num_layers}, heads={args.nhead})...")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class GraphAwareTransformer(nn.Module):
    def __init__(self, src_v, ipa_v, tgt_v, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_v, d_model, padding_idx=0)
        self.ipa_embed = nn.Embedding(ipa_v, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_v, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)

        # Fuse char + IPA embeddings
        self.input_fusion = nn.Linear(d_model * 2, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )

        # Output heads
        self.char_head = nn.Linear(d_model, tgt_v)         # Primary: predict FR chars
        self.quality_head = nn.Sequential(                  # Aux: predict sound quality
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )
        self.tier_head = nn.Linear(d_model, 5)              # Aux: classify tier (S/A/B_safe/B_reserv/B)
        self.loop_head = nn.Sequential(                     # Aux: predict loop certification
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )

    def forward(self, src, ipa, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # Encode source (char + IPA fused)
        se = self.src_embed(src) + self.ipa_embed(ipa)
        se = self.pos_encoder(se)
        # Encode target
        te = self.tgt_embed(tgt)
        te = self.pos_encoder(te)

        # Transformer
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask_out = (tgt == 0)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=src.device)

        output = self.transformer(
            se, te,
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask_out,
        )

        # Multi-task heads (from last layer output)
        char_logits = self.char_head(output)
        quality = self.quality_head(output.mean(dim=1)).squeeze(-1)
        tier_logits = self.tier_head(output.mean(dim=1))
        loop_prob = self.loop_head(output.mean(dim=1)).squeeze(-1)

        return char_logits, quality, tier_logits, loop_prob

    def generate(self, src, ipa, max_len=15):
        """Autoregressive generation."""
        self.eval()
        with torch.no_grad():
            B = src.size(0)
            se = self.src_embed(src) + self.ipa_embed(ipa)
            se = self.pos_encoder(se)
            src_mask = (src == 0)

            # Memory from encoder
            memory = self.transformer.encoder(se, src_key_padding_mask=src_mask)

            # Start decoding
            ys = torch.ones(B, 1).fill_(TGT_C2I["<sos>"]).long().to(src.device)
            for _ in range(max_len):
                te = self.tgt_embed(ys)
                te = self.pos_encoder(te)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1), device=src.device)
                out = self.transformer.decoder(te, memory, tgt_mask=tgt_mask)
                logits = self.char_head(out[:, -1:, :])
                next_token = logits.argmax(-1)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == TGT_C2I["<eos>"]:
                    break

            # Decode tokens
            result = []
            for tok in ys[0, 1:]:  # skip <sos>
                t = tok.item()
                if t == TGT_C2I["<eos>"]: break
                ch = TGT_I2C.get(t, "?")
                if ch not in ("<pad>", "<sos>", "<unk>"): result.append(ch)
            return "".join(result)

# ── 5. Training ─────────────────────────────────────────────────────────
print(f"\n[3/5] Initializing model...")
model = GraphAwareTransformer(
    SRC_V, IPA_V, TGT_V,
    d_model=args.d_model, nhead=args.nhead,
    num_layers=args.num_layers, dropout=args.dropout
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {params:,}")

if args.resume and os.path.exists(args.resume):
    print(f"  Resuming from {args.resume}")
    model.load_state_dict(torch.load(args.resume, map_location=device)["model"])

# Losses
char_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
quality_criterion = nn.MSELoss(reduction='none')
tier_criterion = nn.CrossEntropyLoss(reduction='none')
loop_criterion = nn.BCELoss(reduction='none')

opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

best_val = float("inf")
patience, patience_counter = 20, 0

print(f"\n[4/5] Training {args.epochs} epochs...")
t0 = time.time()

for epoch in range(args.epochs):
    model.train()
    total_loss = 0

    for batch in train_dl:
        src, ipa, tgt, tier, score, loop, weight = [b.to(device) for b in batch]

        opt.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                char_logits, quality_pred, tier_logits, loop_pred = model(src, ipa, tgt[:, :-1])
                tgt_out = tgt[:, 1:]

                # Character loss (primary) — weighted by tier + graph
                char_loss = char_criterion(char_logits.reshape(-1, TGT_V), tgt_out.reshape(-1))
                char_loss = char_loss.reshape(tgt_out.size(0), -1).mean(dim=1)
                char_loss = (char_loss * weight).mean()

                # Auxiliary losses (unweighted — just for regularization)
                quality_loss = quality_criterion(quality_pred, score).mean() * 0.1
                tier_loss = tier_criterion(tier_logits, tier).mean() * 0.05
                loop_loss = loop_criterion(loop_pred, loop).mean() * 0.05

                loss = char_loss + quality_loss + tier_loss + loop_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            char_logits, quality_pred, tier_logits, loop_pred = model(src, ipa, tgt[:, :-1])
            tgt_out = tgt[:, 1:]

            char_loss = char_criterion(char_logits.reshape(-1, TGT_V), tgt_out.reshape(-1))
            char_loss = char_loss.reshape(tgt_out.size(0), -1).mean(dim=1)
            char_loss = (char_loss * weight).mean()

            quality_loss = quality_criterion(quality_pred, score).mean() * 0.1
            tier_loss = tier_criterion(tier_logits, tier).mean() * 0.05
            loop_loss = loop_criterion(loop_pred, loop).mean() * 0.05

            loss = char_loss + quality_loss + tier_loss + loop_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        total_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dl:
            src, ipa, tgt, tier, score, loop, weight = [b.to(device) for b in batch]
            char_logits, _, _, _ = model(src, ipa, tgt[:, :-1])
            cl = char_criterion(char_logits.reshape(-1, TGT_V), tgt[:, 1:].reshape(-1))
            cl = cl.reshape(tgt.size(0) - 1, -1).mean() if tgt.size(0) > 1 else cl.mean()
            val_loss += cl.item()

    val_loss /= max(1, len(val_dl))

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        patience_counter = 0
        torch.save({
            "model": model.state_dict(),
            "src_c2i": SRC_C2I, "tgt_c2i": TGT_C2I, "ipa_c2i": IPA_C2I,
            "tgt_i2c": TGT_I2C, "max_len": MAX_LEN,
            "d_model": args.d_model, "nhead": args.nhead,
            "num_layers": args.num_layers,
        }, args.output)
    else:
        patience_counter += 1

    if (epoch + 1) % 15 == 0 or epoch == 0:
        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:3d}: train={total_loss/len(train_dl):.4f}  "
              f"val={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.0f}s")

    if patience_counter >= patience:
        print(f"  Early stopping at epoch {epoch+1}")
        break

print(f"\n  Best val_loss={best_val:.4f}  saved → {args.output}  ({params:,} params)")

# ── 5. Test ─────────────────────────────────────────────────────────────
print(f"\n[5/5] {'='*50}")
print(f"TESTING TRANSFORMER MODEL")
print(f"{'='*50}")

state = torch.load(args.output, map_location=device, weights_only=False)
model.load_state_dict(state["model"])
model.eval()

tests = ["beauty", "ocean", "silent", "shadow", "mountain", "thunder",
         "wandered", "twilight", "river", "forest", "dreamer", "starlight",
         "moonlight", "heartbeat", "waterfall", "sunrise"]

for w in tests:
    # Find IPA
    ipa_str = ""
    for r in raw_data:
        if r["en"] == w and r.get("en_ipa"):
            ipa_str = r["en_ipa"]
            break

    # Tokenize
    st = [SRC_C2I["<sos>"]] + [SRC_C2I.get(c, 3) for c in w] + [SRC_C2I["<eos>"]]
    st += [0] * (MAX_LEN - len(st))
    it = [IPA_C2I["<sos>"]] + [IPA_C2I.get(c, 3) for c in ipa_str] + [IPA_C2I["<eos>"]]
    it += [0] * (MAX_LEN - len(it))

    src_t = torch.tensor([st[:MAX_LEN]], device=device)
    ipa_t = torch.tensor([it[:MAX_LEN]], device=device)
    fr = model.generate(src_t, ipa_t)
    print(f"  {w:15s} → {fr}")

print(f"\nModel: {args.output} ({params:,} params, {args.num_layers} layers, d={args.d_model})")
if device.type == 'cuda':
    print(f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB peak")
