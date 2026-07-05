#!/usr/bin/env python3
"""Run the GPU-trained 13M-param homophone model — correct key names."""
import torch, torch.nn as nn, math, os

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")
state = torch.load("homophone_model_gpu.pt", map_location="cpu", weights_only=False)
SRC_C2I = state["src_c2i"]; TGT_C2I = state["tgt_c2i"]
TGT_I2C = state["tgt_i2c"]; MAX_LEN = state["max_len"]

# Match the EXACT attribute names from the saved model
class H(nn.Module):
    def __init__(self, sv, tv, h=512, e=256):
        super().__init__()
        self.src_embed = nn.Embedding(sv, e, padding_idx=0)
        self.tgt_embed = nn.Embedding(tv, e, padding_idx=0)
        self.encoder = nn.LSTM(e, h, 3, batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(e+h, h, 3, batch_first=True, dropout=0.1)
        self.attn_W = nn.Linear(h, h)
        self.enc_to_attn = nn.Linear(h, e)
        self.out = nn.Linear(h, tv)
        self.hidden, self.embed = h, e

    def generate(self, src, ml=15):
        with torch.no_grad():
            se = self.src_embed(src); eo, (h, c) = self.encoder(se)
            di = torch.tensor([[TGT_C2I["<sos>"]]], device=src.device)
            r = []
            for _ in range(ml):
                de = self.tgt_embed(di)
                ep = self.enc_to_attn(eo)
                sc = torch.bmm(de, ep.transpose(1,2))
                at = torch.softmax(sc / math.sqrt(self.embed), -1)
                ctx = torch.bmm(at, eo)
                do, (h, c) = self.decoder(torch.cat([de,ctx],-1), (h,c))
                tok = self.out(do).argmax(-1).item()
                if tok == TGT_C2I["<eos>"]: break
                ch = TGT_I2C.get(tok, "?")
                if ch not in ("<pad>","<sos>"): r.append(ch)
                di = torch.tensor([[tok]], device=src.device)
            return "".join(r)

model = H(len(SRC_C2I), len(TGT_C2I))
model.load_state_dict(state["model"]); model.eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, ready.\n")

def carve(word):
    t = [SRC_C2I["<sos>"]] + [SRC_C2I.get(c,0) for c in word] + [SRC_C2I["<eos>"]]
    t += [0]*(MAX_LEN-len(t))
    return model.generate(torch.tensor([t[:MAX_LEN]]))

tests = ["beauty","ocean","silent","wandered","twilight","river","forest",
         "mountain","thunder","whisper","shadow","dreamer","starlight",
         "moonlight","heartbeat","waterfall","sunrise","nightfall","wildfire","snowfall"]
for w in tests:
    print(f"  {w:15s} → {carve(w)}")
