#!/usr/bin/env python3
"""
TRAIN HOMOPHONE MODEL — Character-level LSTM seq2seq, 200 pairs, CPU training.
No transformers needed. Just torch + numpy. Trains in ~2 minutes on CPU.

Learns: English word → French homophone spelling
The model discovers character-level transformation rules (sh→ch, th→t, etc.)

Output: homophone_model.pt (small, portable)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json, random, os

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ── Load training data ──
with open("train-homophonic-full.jsonl") as f:
    data = [json.loads(line) for line in f]

# Parse: "English word: shared" → "shared" / "chers"
pairs = []
for r in data:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    if en and fr and en != fr:
        pairs.append((en, fr))

print(f"Loaded {len(pairs)} training pairs")
print(f"Sample: {pairs[0][0]} → {pairs[0][1]}")
print(f"        {pairs[500][0]} → {pairs[500][1]}")
print(f"        {pairs[3000][0]} → {pairs[3000][1]}")

# ── Build character vocabularies ──
SRC_CHARS = set()
TGT_CHARS = set()
for en, fr in pairs:
    SRC_CHARS.update(en)
    TGT_CHARS.update(fr)

SRC_CHARS = ["<pad>", "<sos>", "<eos>"] + sorted(SRC_CHARS)
TGT_CHARS = ["<pad>", "<sos>", "<eos>"] + sorted(TGT_CHARS)
src_to_idx = {c: i for i, c in enumerate(SRC_CHARS)}
tgt_to_idx = {c: i for i, c in enumerate(TGT_CHARS)}
idx_to_tgt = {i: c for c, i in tgt_to_idx.items()}

SRC_VOCAB_SIZE = len(SRC_CHARS)
TGT_VOCAB_SIZE = len(TGT_CHARS)
print(f"Vocab: {SRC_VOCAB_SIZE} src chars, {TGT_VOCAB_SIZE} tgt chars")

# ── Model: 2-layer LSTM encoder-decoder with attention ──
class HomophoneModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, hidden=128):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, hidden, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab, hidden, padding_idx=0)
        self.encoder = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(hidden * 2, hidden, num_layers=2, batch_first=True)
        self.attn = nn.Linear(hidden * 3, hidden)
        self.out = nn.Linear(hidden, tgt_vocab)
        self.hidden = hidden

    def forward(self, src, tgt, teacher_forcing=0.5):
        # Encode
        src_emb = self.src_embed(src)  # (B, S_src, H)
        enc_out, (h, c) = self.encoder(src_emb)

        # Decode step by step
        B, T_tgt = tgt.shape
        dec_input = tgt[:, 0:1]  # <sos> token
        outputs = []

        for t in range(1, T_tgt):
            dec_emb = self.tgt_embed(dec_input)  # (B, 1, H)

            # Attention over encoder outputs
            enc_expanded = enc_out.unsqueeze(1).expand(-1, 1, -1, -1)
            attn_weights = torch.softmax(
                torch.sum(dec_emb.unsqueeze(2) * enc_out.unsqueeze(1), dim=-1),
                dim=-1,
            )
            context = torch.sum(attn_weights.unsqueeze(-1) * enc_out.unsqueeze(1), dim=2)

            # Decoder step
            dec_input_full = torch.cat([dec_emb, context], dim=-1)
            dec_out, (h, c) = self.decoder(dec_input_full, (h, c))
            logits = self.out(dec_out)  # (B, 1, tgt_vocab)
            outputs.append(logits)

            # Teacher forcing or autoregressive
            if random.random() < teacher_forcing:
                dec_input = tgt[:, t:t+1]
            else:
                dec_input = logits.argmax(-1)

        return torch.cat(outputs, dim=1)  # (B, T_tgt-1, tgt_vocab)

    def generate(self, src, max_len=20):
        """Autoregressive generation from source sequence."""
        src_emb = self.src_embed(src)
        enc_out, (h, c) = self.encoder(src_emb)

        dec_input = torch.tensor([[tgt_to_idx["<sos>"]]], device=src.device)
        result = []

        for _ in range(max_len):
            dec_emb = self.tgt_embed(dec_input)
            attn_weights = torch.softmax(
                torch.sum(dec_emb.unsqueeze(2) * enc_out.unsqueeze(1), dim=-1), dim=-1
            )
            context = torch.sum(attn_weights.unsqueeze(-1) * enc_out.unsqueeze(1), dim=2)
            dec_in_full = torch.cat([dec_emb, context], dim=-1)
            dec_out, (h, c) = self.decoder(dec_in_full, (h, c))
            logits = self.out(dec_out)
            token = logits.argmax(-1).item()
            if token == tgt_to_idx["<eos>"]:
                break
            if token != tgt_to_idx["<pad>"]:
                result.append(idx_to_tgt.get(token, "?"))
            dec_input = torch.tensor([[token]], device=src.device)

        return "".join(result)

# ── Tokenize ──
def encode(text, char_to_idx, max_len=12):
    tokens = [char_to_idx.get(c, 0) for c in text]
    tokens = [char_to_idx["<sos>"]] + tokens + [char_to_idx["<eos>"]]
    if len(tokens) < max_len:
        tokens += [char_to_idx["<pad>"]] * (max_len - len(tokens))
    return tokens[:max_len]

def decode(indices, idx_to_char):
    result = []
    for i in indices:
        c = idx_to_char.get(i, "")
        if c in ("<pad>", "<sos>"):
            continue
        if c == "<eos>":
            break
        result.append(c)
    return "".join(result)

# ── Prepare data ──
MAX_LEN = 12
X = torch.tensor([encode(en, src_to_idx, MAX_LEN) for en, _ in pairs])
Y = torch.tensor([encode(fr, tgt_to_idx, MAX_LEN) for _, fr in pairs])

# Train/val split
indices = list(range(len(pairs)))
random.shuffle(indices)
split = int(0.85 * len(pairs))
train_idx, val_idx = indices[:split], indices[split:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

# ── Train ──
model = HomophoneModel(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, hidden=256)
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore pad

EPOCHS = 100
BATCH = 64

print(f"\nTraining {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i in range(0, len(train_idx), BATCH):
        src = X_train[i:i+BATCH]
        tgt = Y_train[i:i+BATCH]
        optimizer.zero_grad()
        output = model(src, tgt)  # (B, T-1, vocab)
        loss = criterion(output.reshape(-1, TGT_VOCAB_SIZE), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_output = model(X_val, Y_val, teacher_forcing=0)
            val_loss = criterion(val_output.reshape(-1, TGT_VOCAB_SIZE), Y_val[:, 1:].reshape(-1))
        print(f"  epoch {epoch+1:3d}: train_loss={total_loss/len(train_idx):.4f}  val_loss={val_loss.item():.4f}")

# ── Evaluate ──
model.eval()
print(f"\n{'='*50}")
print(f"EVALUATION — Holdout set ({len(val_idx)} pairs)")
print(f"{'='*50}")

exact_matches = 0
char_accuracy = 0
total_chars = 0
for i in range(len(val_idx)):
    en, fr_true = pairs[val_idx[i]]
    src = X_val[i:i+1]
    pred = model.generate(src)
    en_chars = len(en)
    fr_chars = len(fr_true)
    correct = sum(1 for a, b in zip(pred, fr_true) if a == b)
    char_accuracy += correct
    total_chars += max(en_chars, fr_chars)
    if pred == fr_true:
        exact_matches += 1
        mark = "✓"
    else:
        mark = "~"
    if i < 15:
        print(f"  {mark} {en:15s} → pred:{pred:15s}  true:{fr_true}")

print(f"\n  Exact matches: {exact_matches}/{len(val_idx)} ({100*exact_matches/len(val_idx):.0f}%)")
print(f"  Char accuracy: {100*char_accuracy/total_chars:.1f}%")

# ── Test on new words ──
print(f"\n{'='*50}")
print(f"TEST — Novel words (not in training)")
print(f"{'='*50}")

test_words = ["beauty", "silent", "sea", "remember", "dawn", "ship", "sorrow",
              "dancing", "moon", "star", "deep", "free", "soul", "dream"]

for w in test_words:
    tokens = torch.tensor([encode(w, src_to_idx, MAX_LEN)])
    pred = model.generate(tokens)
    # Find closest known word
    if pred:
        print(f"  {w:15s} → {pred}")
    else:
        print(f"  {w:15s} → (no output)")

# ── Save ──
torch.save({
    "model_state": model.state_dict(),
    "src_chars": SRC_CHARS,
    "tgt_chars": TGT_CHARS,
    "src_vocab_size": SRC_VOCAB_SIZE,
    "tgt_vocab_size": TGT_VOCAB_SIZE,
    "max_len": MAX_LEN,
}, "homophone_model.pt")
print(f"\nSaved homophone_model.pt")
