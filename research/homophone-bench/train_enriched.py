#!/usr/bin/env python3
"""
ENRICHED HOMOPHONE MODEL — Train on strict-gold + dictionary v5 + fragments + round-rabbit.

Dual encoder: character + IPA phoneme → FR character decoder.
Fragment-aware, tier-weighted, multi-task.

Usage:
    python train_enriched.py                    # CPU training
    python train_enriched.py --gpu              # GPU training (vast.ai RTX 4090)
    python train_enriched.py --quick            # Fast test run (100 pairs, 5 epochs)

Output: homophone_model_enriched.pt (~30MB portable)
"""

import json, os, sys, math, time, random, argparse
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim

# ── Config ──────────────────────────────────────────────────────────────
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BENCH_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--quick", action="store_true")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--hidden", type=int, default=512)
parser.add_argument("--embed", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--output", default="homophone_model_enriched.pt")
args = parser.parse_args()

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Load strict-gold training pairs ──────────────────────────────────
print("\n[1/6] Loading strict-gold pairs...")
gold_pairs = []
with open("strict-gold-training.jsonl") as f:
    for line in f:
        r = json.loads(line)
        en = r["input"].replace("English word: ", "").strip().lower()
        fr = r["output"].strip().lower()
        quality = r.get("quality", r.get("sound", 1.0))
        if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
            gold_pairs.append({"en": en, "fr": fr, "quality": quality, "source": "strict_gold"})
print(f"  {len(gold_pairs)} strict-gold pairs")

# ── 2. Join with dictionary-v5 for IPA, tier, alignment ─────────────────
print("\n[2/6] Joining with dictionary-v5...")
v5_map = {}
with open("dictionary-v5.tsv") as f:
    cols = f.readline().rstrip("\n").split("\t")
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        d = dict(zip(cols, parts))
        key = (d["en"].lower(), d["fr"].lower())
        if key not in v5_map or float(d.get("score", 0)) > float(v5_map[key].get("score", 0)):
            v5_map[key] = d

enriched = []
for p in gold_pairs:
    key = (p["en"], p["fr"])
    if key in v5_map:
        d = v5_map[key]
        p["en_ipa"] = d.get("en_ipa", "")
        p["fr_ipa"] = d.get("fr_ipa", "")
        p["tier"] = d.get("tier", "B")
        p["alignment"] = d.get("alignment", "")
        p["pivot"] = d.get("pivot", "")
        p["en_syll"] = int(d.get("en_syll", 0) or 0)
        p["fr_syll"] = int(d.get("fr_syll", 0) or 0)
        p["syllable_delta"] = int(d.get("syllable_delta", 0) or 0)
        p["gap_ratio"] = float(d.get("gap_ratio", 0) or 0)
        p["chunk_recipe"] = d.get("chunk_recipe", "")
        p["usable"] = int(d.get("usable_for_composition", 0) or 0)
        p["en_onset"] = d.get("en_onset", "")
        p["en_coda"] = d.get("en_coda", "")
        p["fr_onset"] = d.get("fr_onset", "")
        p["fr_coda"] = d.get("fr_coda", "")
    else:
        p["en_ipa"] = ""
        p["fr_ipa"] = ""
        p["tier"] = "B"
        p["alignment"] = ""
        p["pivot"] = ""
        p["en_syll"] = 0
        p["fr_syll"] = 0
        p["syllable_delta"] = 0
        p["gap_ratio"] = 0.0
        p["chunk_recipe"] = ""
        p["usable"] = 1
        p["en_onset"] = ""
        p["en_coda"] = ""
        p["fr_onset"] = ""
        p["fr_coda"] = ""
    enriched.append(p)

print(f"  {sum(1 for p in enriched if p['en_ipa'])} have IPA ({100*sum(1 for p in enriched if p['en_ipa'])/len(enriched):.0f}%)")
tiers = Counter(p["tier"] for p in enriched)
print(f"  Tiers: {dict(tiers)}")

# ── 3. Add generative-matches (fragment-chained pairs) ──────────────────
print("\n[3/6] Adding generative-matches...")
gen_count = 0
if os.path.exists("generative-matches.tsv"):
    with open("generative-matches.tsv") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                score, en, fr, en_ipa = parts[0], parts[1], parts[2], parts[3]
                chunk = parts[4] if len(parts) > 4 else ""
                try:
                    s = float(score)
                except ValueError:
                    continue
                en, fr = en.strip().lower(), fr.strip().lower()
                if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
                    if (en, fr) not in {(p["en"], p["fr"]) for p in enriched}:
                        enriched.append({
                            "en": en, "fr": fr, "quality": s, "source": "generative",
                            "en_ipa": en_ipa, "fr_ipa": "", "tier": "A" if s >= 0.95 else "B",
                            "alignment": "", "pivot": "", "en_syll": 0, "fr_syll": 0,
                            "syllable_delta": 0, "gap_ratio": 0.0, "chunk_recipe": chunk,
                            "usable": 1, "en_onset": "", "en_coda": "", "fr_onset": "", "fr_coda": ""
                        })
                        gen_count += 1
print(f"  Added {gen_count} generative-match pairs")

# ── 4. Add round-rabbit attachments (semantic neighbors) ─────────────────
print("\n[4/6] Adding round-rabbit attachments...")
rr_count = 0
if os.path.exists("round-rabbit.json"):
    rr = json.load(open("round-rabbit.json"))
    for row in rr["rows"]:
        for att in row.get("attachments", []):
            en, fr = att["en"].strip().lower(), att["fr"].strip().lower()
            score = float(att.get("score", 0))
            tier = att.get("tier", "B")
            kind = att.get("kind", "whole")
            if en and fr and en != fr and 2 <= len(en) <= 15 and 2 <= len(fr) <= 15:
                if (en, fr) not in {(p["en"], p["fr"]) for p in enriched}:
                    row_score = row.get("rank_score", score)
                    hops = row.get("homophonic_hops", 0)
                    enriched.append({
                        "en": en, "fr": fr, "quality": score, "source": f"round_rabbit_h{hops}",
                        "en_ipa": "", "fr_ipa": "", "tier": tier,
                        "alignment": "", "pivot": "", "en_syll": 0, "fr_syll": 0,
                        "syllable_delta": 0, "gap_ratio": 0.0, "chunk_recipe": "",
                        "usable": 1, "en_onset": "", "en_coda": "", "fr_onset": "", "fr_coda": ""
                    })
                    rr_count += 1
print(f"  Added {rr_count} round-rabbit pairs")

# ── Quick mode ───────────────────────────────────────────────────────────
if args.quick:
    random.shuffle(enriched)
    enriched = enriched[:500]
    args.epochs = 5
    print(f"  QUICK MODE: {len(enriched)} pairs, {args.epochs} epochs")

# ── 5. Build character + IPA vocabularies ────────────────────────────────
print(f"\n[5/6] Building vocabularies ({len(enriched)} pairs)...")
SRC_C2I = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
TGT_C2I = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
IPA_C2I = {"<pad>": 0, "<sos>": 1, "<eos>": 2}

for p in enriched:
    for c in p["en"]:
        SRC_C2I.setdefault(c, len(SRC_C2I))
    for c in p["fr"]:
        TGT_C2I.setdefault(c, len(TGT_C2I))
    for c in p["en_ipa"]:
        IPA_C2I.setdefault(c, len(IPA_C2I))

TGT_I2C = {i: c for c, i in TGT_C2I.items()}
SRC_V, TGT_V, IPA_V = len(SRC_C2I), len(TGT_C2I), len(IPA_C2I)
MAX_LEN = 16
print(f"  Vocab: {SRC_V} src chars, {IPA_V} IPA chars, {TGT_V} tgt chars")

# ── Tokenize ─────────────────────────────────────────────────────────────
def encode_chars(text, c2i, max_len):
    tokens = [c2i["<sos>"]] + [c2i.get(c, 0) for c in text] + [c2i["<eos>"]]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len], dtype=torch.long)

def encode_ipa(ipa, max_len):
    if not ipa:
        return torch.zeros(max_len, dtype=torch.long)
    tokens = [IPA_C2I["<sos>"]] + [IPA_C2I.get(c, 0) for c in ipa] + [IPA_C2I["<eos>"]]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len], dtype=torch.long)

# Tier weights for loss
TIER_W = {"S": 3.0, "A": 2.0, "B_safe": 1.0, "B_reservoir": 0.8, "B": 1.0}

X_chars = torch.stack([encode_chars(p["en"], SRC_C2I, MAX_LEN) for p in enriched])
X_ipa = torch.stack([encode_ipa(p["en_ipa"], MAX_LEN) for p in enriched])
Y = torch.stack([encode_chars(p["fr"], TGT_C2I, MAX_LEN) for p in enriched])
W = torch.tensor([TIER_W.get(p["tier"], 1.0) for p in enriched], dtype=torch.float32)

# Train/val split
perm = torch.randperm(len(enriched))
split = int(0.9 * len(enriched))
Xc_tr, Xi_tr, Y_tr, W_tr = X_chars[perm[:split]], X_ipa[perm[:split]], Y[perm[:split]], W[perm[:split]]
Xc_vl, Xi_vl, Y_vl, W_vl = X_chars[perm[split:]], X_ipa[perm[split:]], Y[perm[split:]], W[perm[split:]]
print(f"  Train: {len(Xc_tr)}, Val: {len(Xc_vl)}")

# ── 6. Model: Dual Encoder + Attention Decoder ───────────────────────────
print(f"\n[6/6] Training {args.epochs} epochs...")

class DualEncoderLSTM(nn.Module):
    def __init__(self, src_v, ipa_v, tgt_v, hidden=512, embed=256):
        super().__init__()
        self.src_embed = nn.Embedding(src_v, embed, padding_idx=0)
        self.ipa_embed = nn.Embedding(ipa_v, embed, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_v, embed, padding_idx=0)

        self.char_encoder = nn.LSTM(embed, hidden, 3, batch_first=True, dropout=0.1)
        self.phoneme_encoder = nn.LSTM(embed, hidden // 2, 2, batch_first=True, dropout=0.1)

        self.fusion = nn.Linear(hidden + hidden // 2, hidden)
        self.decoder = nn.LSTM(embed + hidden, hidden, 3, batch_first=True, dropout=0.1)
        self.attn_W = nn.Linear(hidden, hidden)
        self.enc_to_attn = nn.Linear(hidden, embed)
        self.out = nn.Linear(hidden, tgt_v)
        self.hidden, self.embed = hidden, embed

    def forward(self, src_chars, src_ipa, tgt, tf_ratio=0.5):
        B = src_chars.size(0)
        # Char encoder
        ce = self.src_embed(src_chars)
        co, (ch, cc) = self.char_encoder(ce)
        # Phoneme encoder
        pe = self.ipa_embed(src_ipa)
        po, (ph, pc) = self.phoneme_encoder(pe)
        # Fuse: use char path as primary, phoneme as supplement
        enc_out = self.fusion(torch.cat([co, po], dim=-1))
        h = torch.cat([ch[-1:], ph[-1:]], dim=-1).unsqueeze(0).repeat(3, 1, 1)
        c = torch.cat([cc[-1:], pc[-1:]], dim=-1).unsqueeze(0).repeat(3, 1, 1)

        dec_in = tgt[:, 0:1]
        outputs = []
        for t in range(1, tgt.size(1)):
            de = self.tgt_embed(dec_in)
            ep = self.enc_to_attn(enc_out)
            sc = torch.bmm(de, ep.transpose(1, 2))
            at = torch.softmax(sc / math.sqrt(self.embed), -1)
            ctx = torch.bmm(at, enc_out)
            di = torch.cat([de, ctx], -1)
            do, (h, c) = self.decoder(di, (h, c))
            logits = self.out(do)
            outputs.append(logits)
            if torch.rand(1).item() < tf_ratio:
                dec_in = tgt[:, t:t+1]
            else:
                dec_in = logits.argmax(-1)
        return torch.cat(outputs, dim=1)

    def generate(self, src_chars, src_ipa, max_len=15):
        with torch.no_grad():
            B = src_chars.size(0)
            ce = self.src_embed(src_chars)
            co, (ch, cc) = self.char_encoder(ce)
            pe = self.ipa_embed(src_ipa)
            po, (ph, pc) = self.phoneme_encoder(pe)
            enc_out = self.fusion(torch.cat([co, po], dim=-1))
            h = torch.cat([ch[-1:], ph[-1:]], dim=-1).unsqueeze(0).repeat(3, 1, 1)
            c = torch.cat([cc[-1:], pc[-1:]], dim=-1).unsqueeze(0).repeat(3, 1, 1)

            dec_in = torch.tensor([[TGT_C2I["<sos>"]]], device=src_chars.device)
            result = []
            for _ in range(max_len):
                de = self.tgt_embed(dec_in)
                ep = self.enc_to_attn(enc_out)
                sc = torch.bmm(de, ep.transpose(1, 2))
                at = torch.softmax(sc / math.sqrt(self.embed), -1)
                ctx = torch.bmm(at, enc_out)
                do, (h, c) = self.decoder(torch.cat([de, ctx], -1), (h, c))
                tok = self.out(do).argmax(-1).item()
                if tok == TGT_C2I["<eos>"]:
                    break
                ch = TGT_I2C.get(tok, "?")
                if ch not in ("<pad>", "<sos>"):
                    result.append(ch)
                dec_in = torch.tensor([[tok]], device=src_chars.device)
            return "".join(result)

# ── Train ────────────────────────────────────────────────────────────────
model = DualEncoderLSTM(SRC_V, IPA_V, TGT_V, args.hidden, args.embed).to(device)
opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
crit = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

best_val = float("inf")
params = sum(p.numel() for p in model.parameters())
print(f"  Model: {params:,} params")

t0 = time.time()
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    perm = torch.randperm(len(Xc_tr))
    for i in range(0, len(Xc_tr), args.batch):
        idx = perm[i:i+args.batch]
        sc, si, tg, wt = Xc_tr[idx].to(device), Xi_tr[idx].to(device), Y_tr[idx].to(device), W_tr[idx].to(device)
        opt.zero_grad()
        out = model(sc, si, tg, tf_ratio=max(0.2, 0.5 * (1 - epoch / args.epochs)))
        loss_per_token = crit(out.reshape(-1, TGT_V), tg[:, 1:].reshape(-1))
        loss_per_seq = loss_per_token.reshape(tg.size(0), -1).mean(dim=1)
        loss = (loss_per_seq * wt).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()

    sched.step()

    # Validation
    model.eval()
    with torch.no_grad():
        sv, iv, tv, wv = Xc_vl[:200].to(device), Xi_vl[:200].to(device), Y_vl[:200].to(device), W_vl[:200].to(device)
        ov = model(sv, iv, tv, tf_ratio=0)
        lv = crit(ov.reshape(-1, TGT_V), tv[:, 1:].reshape(-1))
        lv_seq = lv.reshape(tv.size(0), -1).mean(dim=1)
        val_loss = (lv_seq * wv).mean().item()

    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            "model": model.state_dict(),
            "src_c2i": SRC_C2I, "tgt_c2i": TGT_C2I, "ipa_c2i": IPA_C2I,
            "tgt_i2c": TGT_I2C, "max_len": MAX_LEN,
            "hidden": args.hidden, "embed": args.embed
        }, args.output)

    if (epoch + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:3d}: train_loss={total_loss/len(Xc_tr):.4f}  "
              f"val_loss={val_loss:.4f}  time={elapsed:.0f}s")

print(f"\n  Best val_loss={best_val:.4f}  saved → {args.output}  ({params:,} params)")

# ── Test ─────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("TEST")
print(f"{'='*50}")
model.load_state_dict(torch.load(args.output, map_location=device, weights_only=False)["model"])
model.eval()

tests = ["beauty", "ocean", "silent", "shadow", "mountain", "thunder",
         "wandered", "twilight", "river", "forest", "dreamer", "starlight"]
for w in tests:
    # Find IPA if available
    ipa = ""
    for p in enriched:
        if p["en"] == w and p["en_ipa"]:
            ipa = p["en_ipa"]
            break
    ct = encode_chars(w, SRC_C2I, MAX_LEN).unsqueeze(0).to(device)
    it = encode_ipa(ipa, MAX_LEN).unsqueeze(0).to(device)
    fr = model.generate(ct, it)
    print(f"  {w:15s} → {fr}")
