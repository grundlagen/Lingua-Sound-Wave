#!/usr/bin/env python3
"""Bilingual sentence test using GPU-trained model + lookup + Agent B."""
import torch, torch.nn as nn, math, json, subprocess, os, sys
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Load GPU model ──
state = torch.load("homophone_model_gpu.pt", map_location="cpu", weights_only=False)
SRC_C2I = state["src_c2i"]; TGT_C2I = state["tgt_c2i"]
TGT_I2C = state["tgt_i2c"]; MAX_LEN = state["max_len"]

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
                de = self.tgt_embed(di); ep = self.enc_to_attn(eo)
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

def model_carve(word):
    t = [SRC_C2I["<sos>"]] + [SRC_C2I.get(c,0) for c in word] + [SRC_C2I["<eos>"]]
    t += [0]*(MAX_LEN-len(t))
    return model.generate(torch.tensor([t[:MAX_LEN]]))

# ── Load lookup DB ──
with open("strict-gold-training.jsonl") as f:
    raw = [json.loads(line) for line in f]
lookup = {}
for r in raw:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    q = r.get("quality", r.get("sound", 1.0))
    if en and fr and en != fr:
        if en not in lookup or q > lookup[en][1]: lookup[en] = (fr, q)

def word_sim(a,b):
    sa,sb=set(a),set(b); return len(sa&sb)/max(1,len(sa|sb))

# ── Agent B: what does the French sound like? ──
en_ipa_dict = {}
for i,line in enumerate(open("en-word-ipa.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[1] and "(fr)" not in p[0]:
        en_ipa_dict[p[0].lower()] = p[1].replace(" ","")

def tts(text, voice):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text], capture_output=True, text=True)
    ipa = r.stdout.strip(); 
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    return ipa

def ndice(a,b,n=2):
    A={a[i:i+n] for i in range(len(a)-n+1)} if len(a)>=n else {a}
    B={b[i:i+n] for i in range(len(b)-n+1)} if len(b)>=n else {b}
    return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

def agent_B(fr_word):
    en_ipa = tts(fr_word, "en-us").replace(" ","")
    best_en, best_s = fr_word, 0
    for en_w, en_w_ipa in list(en_ipa_dict.items())[:3000]:
        s = ndice(en_ipa, en_w_ipa)
        if s > best_s: best_s, best_en = s, en_w
    return best_en, best_s

# ═══════════════════════════════════════════════
sentences = [
    "the shadow of the mountain falls across the silent river",
    "she wandered through the deep forest at twilight",
    "a gentle stream becomes a mighty rushing waterfall",
]

for sent in sentences:
    words = [w.lower().strip(".,;:!?\"") for w in sent.split() if w.strip(".,;:!?\"")]
    results = []
    for w in words:
        if w in lookup:
            fr, q = lookup[w]; src = "lookup"
        else:
            fr = model_carve(w); q = 0.7; src = "model"
        heard, hs = agent_B(fr)
        results.append((w, fr, q, src, heard, hs))
    
    fr_text = " ".join(fr for _,fr,_,_,_,_ in results)
    en_heard = " ".join(h for _,_,_,_,h,_ in results)
    
    print(f"\nEN: {sent}")
    print(f"FR: {fr_text}")
    print(f"HE: {en_heard}")
    for w,fr,q,src,heard,hs in results:
        m = "★" if src=="lookup" else " "
        print(f"  {m}{w:12s} → {fr:15s} (q={q:.2f}) → B:{heard:12s} ({hs:.2f})")
