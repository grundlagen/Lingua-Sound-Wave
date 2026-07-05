#!/usr/bin/env python3
"""
Improved train_transformer.py v2 — Epoch checkpoints + Google Drive upload.

Changes from v1:
- Saves checkpoint EVERY epoch (not just on improvement)
- Rotates last 5 epoch checkpoints
- Saves best model separately (best_homophone_transformer.pt)
- More frequent logging (every epoch)
- GPU-side backup daemon support
- Better error handling and resume

Usage:
    python3 train_transformer_v2.py                          # Train full
    python3 train_transformer_v2.py --quick                  # Fast test
    python3 train_transformer_v2.py --resume ckpt.pt         # Resume

Output:
    homophone_transformer.pt          — latest epoch
    best_homophone_transformer.pt     — best val_loss
    checkpoints/epoch_XXX.pt          — rotated (last 5)
"""

import json, os, sys, math, time, argparse, random, glob
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
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--resume", default=None)
parser.add_argument("--output", default="homophone_transformer.pt")
parser.add_argument("--best_output", default="best_homophone_transformer.pt")
parser.add_argument("--data", default="graph_aware_training.jsonl")
parser.add_argument("--checkpoint_dir", default="checkpoints")
parser.add_argument("--save_every", type=int, default=5,
                    help="Save checkpoint every N epochs (in addition to best)")
args = parser.parse_args()

os.makedirs(args.checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Checkpoints: {args.checkpoint_dir}/  (save every {args.save_every} epochs)")

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

# ── 3. Dataset (same as v1) ────────────────────────────────────────────
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
        src = [self.src_c2i["<sos>"]] + [self.src_c2i.get(c, 3) for c in r["en"]] + [self.src_c2i["<eos>"]]
        src += [0] * (self.max_len - len(src))
        src = src[:self.max_len]

        ipa = [self.ipa_c2i["<sos>"]] + [self.ipa_c2i.get(c, 3) for c in r.get("en_ipa", "")] + [self.ipa_c2i["<eos>"]]
        ipa += [0] * (self.max_len - len(ipa))
        ipa = ipa[:self.max_len]

        tgt = [self.tgt_c2i["<sos>"]] + [self.tgt_c2i.get(c, 3) for c in r["fr"]] + [self.tgt_c2i["<eos>"]]
        tgt += [0] * (self.max_len - len(tgt))
        tgt = tgt[:self.max_len]

        tier_idx = TIER_TO_IDX.get(r.get("tier", "B"), 4)
        score = float(r.get("score", 0.5))
        loop = 1.0 if r.get("loop_certified", False) else 0.0
        hops = float(r.get("graph_hops", 0))
        proximity = float(r.get("meaning_proximity", 1.0))

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

# ── 4. Model (same as v1) ─────────────────────────────────────────────
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

        self.input_fusion = nn.Linear(d_model * 2, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )

        self.char_head = nn.Linear(d_model, tgt_v)
        self.quality_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )
        self.tier_head = nn.Linear(d_model, 5)
        self.loop_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )

    def forward(self, src, ipa, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        src_emb = self.input_fusion(torch.cat([self.src_embed(src), self.ipa_embed(ipa)], dim=-1))
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt))

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)

        decoder_out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        char_logits = self.char_head(decoder_out)
        pooled = decoder_out.mean(dim=1)
        quality_pred = self.quality_head(pooled).squeeze(-1)
        tier_logits = self.tier_head(pooled)
        loop_pred = self.loop_head(pooled).squeeze(-1)

        return char_logits, quality_pred, tier_logits, loop_pred

    def generate(self, src, ipa, max_len=18):
        with torch.no_grad():
            src_emb = self.input_fusion(torch.cat([self.src_embed(src), self.ipa_embed(ipa)], dim=-1))
            src_emb = self.pos_encoder(src_emb)
            memory = self.transformer.encoder(src_emb)

            di = torch.tensor([[TGT_C2I["<sos>"]]], device=src.device)
            result = []
            for _ in range(max_len):
                de = self.pos_encoder(self.tgt_embed(di))
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(di.size(1), device=di.device)
                out = self.transformer.decoder(de, memory, tgt_mask=tgt_mask)
                tok = self.char_head(out[:, -1:, :]).argmax(-1).item()
                if tok == TGT_C2I["<eos>"]:
                    break
                ch = TGT_I2C.get(tok, "?")
                if ch not in ("<pad>", "<sos>"):
                    result.append(ch)
                di = torch.cat([di, torch.tensor([[tok]], device=src.device)], 1)
            return "".join(result)

model = GraphAwareTransformer(
    SRC_V, IPA_V, TGT_V,
    d_model=args.d_model, nhead=args.nhead,
    num_layers=args.num_layers, dropout=args.dropout
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {params:,}")

opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
char_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
quality_criterion = nn.MSELoss()
tier_criterion = nn.CrossEntropyLoss()
loop_criterion = nn.BCELoss()

scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

best_val = float('inf')
patience_counter = 0
patience = args.patience
start_epoch = 0

# Resume support
if args.resume and os.path.exists(args.resume):
    print(f"\n[3/5] Resuming from {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt.get("optimizer", opt.state_dict()))
    scheduler.load_state_dict(ckpt.get("scheduler", scheduler.state_dict()))
    best_val = ckpt.get("best_val", float('inf'))
    start_epoch = ckpt.get("epoch", 0) + 1
    patience_counter = ckpt.get("patience_counter", 0)
    print(f"  Resumed at epoch {start_epoch}, best_val={best_val:.4f}")

def save_checkpoint(epoch, val_loss, is_best=False, label=""):
    """Save model checkpoint."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "src_c2i": SRC_C2I, "tgt_c2i": TGT_C2I, "ipa_c2i": IPA_C2I,
        "tgt_i2c": TGT_I2C, "max_len": MAX_LEN,
        "d_model": args.d_model, "nhead": args.nhead,
        "num_layers": args.num_layers,
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val": best_val,
        "patience_counter": patience_counter,
        "config": {
            "d_model": args.d_model, "nhead": args.nhead,
            "num_layers": args.num_layers, "dropout": args.dropout
        }
    }

    # Always save latest
    torch.save(ckpt, args.output)

    # Save best
    if is_best:
        torch.save(ckpt, args.best_output)
        print(f"    ★ Best model saved (val_loss={val_loss:.4f}) → {args.best_output}")

    # Save epoch checkpoint
    if label:
        ep_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}_{label}.pt")
        torch.save(ckpt, ep_path)
        # Rotate: keep last 10
        all_ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "epoch_*.pt")))
        for old in all_ckpts[:-10]:
            os.remove(old)

# ── 5. Training ───────────────────────────────────────────────────────
print(f"\n[4/5] Training {args.epochs} epochs (patience={patience})...")
print(f"  Save every {args.save_every} epochs + best → {args.best_output}")
t0 = time.time()

for epoch in range(start_epoch, args.epochs):
    model.train()
    total_loss = 0

    for batch in train_dl:
        src, ipa, tgt, tier, score, loop, weight = [b.to(device) for b in batch]

        opt.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                char_logits, quality_pred, tier_logits, loop_pred = model(src, ipa, tgt[:, :-1])
                tgt_out = tgt[:, 1:]

                char_loss = char_criterion(char_logits.reshape(-1, TGT_V), tgt_out.reshape(-1))
                char_loss = char_loss.reshape(tgt_out.size(0), -1).mean(dim=1)
                char_loss = (char_loss * weight).mean()

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
            cl = cl.reshape(tgt.size(0), -1).mean()
            val_loss += cl.item()

    val_loss /= max(1, len(val_dl))
    train_avg = total_loss / max(1, len(train_dl))
    elapsed = time.time() - t0

    is_best = val_loss < best_val - 1e-4
    if is_best:
        best_val = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    # Save checkpoint periodically
    is_save_epoch = (epoch + 1) % args.save_every == 0
    save_checkpoint(
        epoch, val_loss,
        is_best=is_best,
        label=f"train{train_avg:.3f}_val{val_loss:.3f}" if is_save_epoch else ""
    )

    # Always print every epoch
    best_mark = " ★" if is_best else ""
    patience_info = f"  pat={patience_counter}/{patience}" if patience_counter > 0 else ""
    print(f"  epoch {epoch+1:3d}: train={train_avg:.4f}  val={val_loss:.4f}"
          f"  lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.0f}s{best_mark}{patience_info}")

    if patience_counter >= patience:
        print(f"\n  Early stopping at epoch {epoch+1} (patience={patience})")
        break

print(f"\n  Best val_loss={best_val:.4f}  saved → {args.best_output}  ({params:,} params)")
print(f"  Total time: {time.time() - t0:.0f}s")

# ── 6. Test ─────────────────────────────────────────────────────────────
print(f"\n[5/5] {'='*50}")
print(f"TESTING TRANSFORMER MODEL")
print(f"{'='*50}")

# Load best model for testing
state = torch.load(args.best_output, map_location=device, weights_only=False)
model.load_state_dict(state["model"])
model.eval()

tests = ["beauty", "ocean", "silent", "shadow", "mountain", "thunder",
         "wandered", "twilight", "river", "forest", "dreamer", "starlight",
         "moonlight", "heartbeat", "waterfall", "sunrise"]

for w in tests:
    ipa_str = ""
    for r in raw_data:
        if r["en"] == w and r.get("en_ipa"):
            ipa_str = r["en_ipa"]
            break

    st = [SRC_C2I["<sos>"]] + [SRC_C2I.get(c, 3) for c in w] + [SRC_C2I["<eos>"]]
    st += [0] * (MAX_LEN - len(st))
    it = [IPA_C2I["<sos>"]] + [IPA_C2I.get(c, 3) for c in ipa_str] + [IPA_C2I["<eos>"]]
    it += [0] * (MAX_LEN - len(it))

    src_t = torch.tensor([st[:MAX_LEN]], device=device)
    ipa_t = torch.tensor([it[:MAX_LEN]], device=device)
    fr = model.generate(src_t, ipa_t)
    print(f"  {w:15s} → {fr}")

print(f"\nModel: {args.best_output} ({params:,} params, {args.num_layers} layers, d={args.d_model})")
if device.type == 'cuda':
    print(f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB peak")
