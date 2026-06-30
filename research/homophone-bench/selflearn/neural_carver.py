"""CPU neural carver -- a small char-level seq2seq that GENERATES French carvings
from English sound, trainable HERE (no GPU, no money, no transformers/TRL).

This is the runnable sibling of train_selflearn.py (which needs a GPU + Qwen).
Same idea, shrunk to fit the box we actually have:

  source = English IPA   (bench.g2p_ipa -> the SOUND to imitate)
  target = French text    (the carving that should sound like it)

  model  = bi-GRU encoder + GRU decoder with Luong attention (pure torch CPU)
  train  = teacher-forced cross-entropy on the STRICT-GOLD corpus
  infer  = sample N candidates per word, RE-RANK by the judge reward
           reward = sqrt( rule-aware sound  x  semantic meaning )   [sound x meaning]
  loop   = expert iteration: fold high-reward generations back into training

The neural generator's job is RECALL (propose carvings, incl. multi-word ones the
dictionary doesn't list); the judge stays the arbiter. Honest expectation: with
~1k pairs on CPU this is a proof-of-loop, not a strong model -- the strong model
is the GPU LLM path (train_selflearn.py) seeded by the same reward.

Run: python neural_carver.py --epochs 12 --demo
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))      # parent: bench, rule_aware, semantic_cosine

import bench
import rule_aware
try:
    from semantic_cosine import semantic_cosine
    _HAVE_SEM = True
except Exception:
    _HAVE_SEM = False

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ["<pad>", "<sos>", "<eos>", "<unk>"]
DEV = torch.device("cpu")


# ----------------------------------------------------------------- vocab / data
class Vocab:
    def __init__(self, texts):
        chars = sorted({c for t in texts for c in t})
        self.itos = SPECIAL + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def enc(self, t, sos=False, eos=False):
        ids = [self.stoi.get(c, UNK) for c in t]
        if sos:
            ids = [SOS] + ids
        if eos:
            ids = ids + [EOS]
        return ids

    def dec(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            if i >= len(SPECIAL):
                out.append(self.itos[i])
        return "".join(out)


def load_pairs(strict_path, gold_path, limit=None):
    pairs, seen = [], set()
    # prefer the grown STRICT-GOLD corpus, then top up from v7 GOLD
    if os.path.exists(strict_path):
        for i, line in enumerate(open(strict_path, encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and (p[0], p[1]) not in seen:
                seen.add((p[0], p[1])); pairs.append((p[0], p[1]))
    for i, line in enumerate(open(gold_path, encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6 and p[5] == "GOLD" and (p[0], p[1]) not in seen:
            seen.add((p[0], p[1])); pairs.append((p[0], p[1]))
    random.Random(0).shuffle(pairs)
    return pairs[:limit] if limit else pairs


# ----------------------------------------------------------------------- model
class Encoder(nn.Module):
    def __init__(self, vsz, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vsz, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hid, hid)

    def forward(self, x):
        e = self.emb(x)
        out, h = self.gru(e)                       # out: [B,T,2H]
        h = torch.tanh(self.bridge(torch.cat([h[0], h[1]], dim=1)))  # [B,H]
        return out, h.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, vsz, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vsz, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.att = nn.Linear(2 * hid, hid)
        self.out = nn.Linear(hid + 2 * hid, vsz)

    def forward(self, tok, h, enc_out, enc_mask):
        e = self.emb(tok)                          # [B,1,E]
        o, h = self.gru(e, h)                      # o: [B,1,H]
        scores = (self.att(enc_out) @ o.transpose(1, 2)).squeeze(2)  # [B,T]
        scores = scores.masked_fill(~enc_mask, -1e9)
        a = F.softmax(scores, dim=1).unsqueeze(1)  # [B,1,T]
        ctx = a @ enc_out                          # [B,1,2H]
        logits = self.out(torch.cat([o, ctx], dim=2)).squeeze(1)    # [B,V]
        return logits, h


class Carver(nn.Module):
    def __init__(self, svocab, tvocab, hid=128):
        super().__init__()
        self.svocab, self.tvocab = svocab, tvocab
        self.enc = Encoder(len(svocab), hid=hid)
        self.dec = Decoder(len(tvocab), hid=hid)

    def src(self, en):
        ipa = bench.g2p_ipa(en, "en")
        return torch.tensor([self.svocab.enc(ipa)], dtype=torch.long)

    def forward(self, src, tgt):
        enc_out, h = self.enc(src)
        mask = src != PAD
        loss = 0.0
        tok = tgt[:, :1]
        for t in range(1, tgt.size(1)):
            logits, h = self.dec(tok, h, enc_out, mask)
            loss = loss + F.cross_entropy(logits, tgt[:, t], ignore_index=PAD)
            tok = tgt[:, t:t + 1]
        return loss / max(1, tgt.size(1) - 1)

    @torch.no_grad()
    def sample(self, en, k=8, temp=0.9, max_len=20):
        enc_out, h0 = self.enc(self.src(en))
        mask = torch.ones(enc_out.shape[:2], dtype=torch.bool)
        outs = []
        for _ in range(k):
            h = h0.clone()
            tok = torch.tensor([[SOS]])
            ids = []
            for _ in range(max_len):
                logits, h = self.dec(tok, h, enc_out, mask)
                p = F.softmax(logits / temp, dim=1)
                nxt = int(torch.multinomial(p, 1))
                if nxt == EOS:
                    break
                ids.append(nxt)
                tok = torch.tensor([[nxt]])
            outs.append(self.tvocab.dec(ids))
        return outs


# --------------------------------------------------------------------- reward
def reward(en, fr):
    if not fr.strip():
        return 0.0
    sound = rule_aware.rule_aware_combo(en, fr)
    if _HAVE_SEM:
        meaning = max(0.0, semantic_cosine(en, fr))
        return float((sound * meaning) ** 0.5)     # sound x meaning
    return sound


# ----------------------------------------------------------------------- train
def batches(pairs, svocab, tvocab, bs=32):
    random.shuffle(pairs)
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        src = [svocab.enc(bench.g2p_ipa(en, "en")) for en, _ in chunk]
        tgt = [tvocab.enc(fr, sos=True, eos=True) for _, fr in chunk]
        sm, tm = max(len(s) for s in src), max(len(t) for t in tgt)
        S = torch.full((len(chunk), sm), PAD, dtype=torch.long)
        T = torch.full((len(chunk), tm), PAD, dtype=torch.long)
        for j, (s, t) in enumerate(zip(src, tgt)):
            S[j, :len(s)] = torch.tensor(s)
            T[j, :len(t)] = torch.tensor(t)
        yield S, T


def train(model, pairs, epochs, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for S, T in batches(list(pairs), model.svocab, model.tvocab):
            opt.zero_grad()
            loss = model(S, T)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); n += 1
        print(f"  epoch {ep + 1:2d}/{epochs}  loss {tot / max(1, n):.3f}")


def best_of_n(model, en, k=10):
    cands = set(model.sample(en, k=k))
    scored = sorted(((reward(en, c), c) for c in cands if c), reverse=True)
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--iters", type=int, default=1, help="expert-iteration rounds")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    random.seed(0); torch.manual_seed(0)

    pairs = load_pairs(os.path.join(os.path.dirname(_HERE), "strict-gold.tsv"),
                       os.path.join(os.path.dirname(_HERE), "dictionary-v7-remined.tsv"),
                       limit=args.limit)
    print(f"training pairs: {len(pairs)} (STRICT-GOLD + v7 GOLD)")
    src_texts = [bench.g2p_ipa(en, "en") for en, _ in pairs]
    svocab = Vocab(src_texts)
    tvocab = Vocab([fr for _, fr in pairs])
    model = Carver(svocab, tvocab, hid=args.hid)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}  "
          f"src-vocab {len(svocab)}  tgt-vocab {len(tvocab)}")

    for it in range(args.iters):
        print(f"\n=== train round {it} ===")
        train(model, pairs, args.epochs)
        if it < args.iters - 1:
            # expert iteration: generate, keep high-reward, fold back in
            seeds = [en for en, _ in random.sample(pairs, min(120, len(pairs)))]
            added = 0
            for en in seeds:
                scored = best_of_n(model, en, k=args.k)
                if scored and scored[0][0] >= 0.55:
                    cand = scored[0][1]
                    if (en, cand) not in set(pairs):
                        pairs.append((en, cand)); added += 1
            print(f"expert iteration: folded {added} high-reward generations back "
                  f"(corpus now {len(pairs)})")

    torch.save({"model": model.state_dict(),
                "svocab": svocab.itos, "tvocab": tvocab.itos, "hid": args.hid},
               os.path.join(_HERE, "neural_carver.pt"))
    print(f"\nsaved -> {os.path.join(_HERE, 'neural_carver.pt')}")

    if args.demo:
        words = ["happy", "money", "water", "english", "little", "thunder",
                 "october", "freedom", "machine"]
        print("\nbest-of-N generation, RE-RANKED by reward (sound x meaning):")
        print(f"{'EN':10s} {'best French carving':24s} {'reward':>6s}   runners-up")
        print("-" * 78)
        for en in words:
            scored = best_of_n(model, en, k=max(args.k, 12))
            if not scored:
                continue
            r, best = scored[0]
            ru = ", ".join(f"{c}({s:.2f})" for s, c in scored[1:4] if c)
            print(f"{en:10s} {best:24s} {r:6.2f}   {ru}")
        print("\nThe model proposes; the judge disposes. High-reward generations "
              "become new STRICT-GOLD seeds -- the same expert-iteration loop as "
              "self_improve.py, now closing through a neural generator.")


if __name__ == "__main__":
    main()
