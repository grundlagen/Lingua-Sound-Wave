#!/usr/bin/env python3
"""
UNIFIED AGENT A — Neural EN→FR homophone carver.

Drop-in replacement for the 6,143-pair lookup in:
  - three_agent_v2.py
  - dual_agent_writer.py
  - three_agent_refinery.py
  - generate_sentence.py

Loads the transformer model if available, falls back to strict-gold lookup.
Also uses dictionary-v5 for tier-weighted candidates.
Also uses homophone-class expansion for broader coverage.

Usage:
    from agent_a import AgentA
    agent = AgentA()                          # auto-detect best backend
    agent = AgentA(model="homophone_transformer.pt")  # explicit model
    candidates = agent.carve("ocean", top_k=5)
    # → [("aucuns", 0.95, "gold"), ("océan", 0.82, "model"), ...]
"""

import json, os, sys, math
from collections import defaultdict

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if BENCH_DIR:
    os.chdir(BENCH_DIR)

# ═══════════════════════════════════════════════════════════════════════
class AgentA:
    """Unified EN→FR homophone carver. Multiple backends, auto-selection."""

    def __init__(self, model_path: str | None = None):
        self.backend = "lookup"  # default
        self.model = None
        self.SRC_C2I = {}
        self.TGT_C2I = {}
        self.TGT_I2C = {}
        self.IPA_C2I = {}
        self.MAX_LEN = 16

        # ── Load strict-gold lookup (always available) ──
        self.gold = {}
        gold_file = "strict-gold-training.jsonl"
        if os.path.exists(gold_file):
            with open(gold_file) as f:
                for line in f:
                    r = json.loads(line)
                    en = r["input"].replace("English word: ", "").strip().lower()
                    fr = r["output"].strip().lower()
                    q = r.get("quality", r.get("sound", 1.0))
                    if en and fr and en != fr:
                        if en not in self.gold or q > self.gold[en][1]:
                            self.gold[en] = (fr, min(q, 1.0))

        # ── Load dictionary-v5 (tier-weighted candidates) ──
        self.dict_idx = defaultdict(list)
        dict_file = "dictionary-v5.tsv"
        if os.path.exists(dict_file):
            with open(dict_file) as f:
                cols = f.readline().rstrip("\n").split("\t")
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 6:
                        continue
                    d = dict(zip(cols, parts))
                    en = d["en"].lower()
                    fr = d["fr"].lower()
                    score = float(d.get("score", 0))
                    tier = d.get("tier", "B")
                    en_ipa = d.get("en_ipa", "")
                    usable = int(d.get("usable_for_composition", 0) or 0)
                    self.dict_idx[en].append((fr, score, tier, en_ipa, usable))
            for en in self.dict_idx:
                self.dict_idx[en].sort(key=lambda x: -x[1])

        # ── Load transformer model if available ──
        import torch
        if model_path is None:
            for candidate in ["homophone_transformer.pt", "homophone_model_enriched.pt",
                              "homophone_model_gpu.pt"]:
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        if model_path and os.path.exists(model_path):
            try:
                import torch.nn as nn
                state = torch.load(model_path, map_location="cpu", weights_only=False)

                self.SRC_C2I = state["src_c2i"]
                self.TGT_C2I = state["tgt_c2i"]
                self.TGT_I2C = state["tgt_i2c"]
                self.MAX_LEN = state["max_len"]
                self.IPA_C2I = state.get("ipa_c2i", {})

                # Determine model type from saved state
                has_ipa = len(self.IPA_C2I) > 3
                d_model = state.get("d_model", state.get("hidden", 512))
                nhead = state.get("nhead", 8)
                num_layers = state.get("num_layers", 6)
                is_transformer = "d_model" in state or "num_layers" in state

                if is_transformer:
                    # Transformer model
                    class DualEncoderLSTM(nn.Module):
                        def __init__(self, sv, iv, tv, hidden=512, embed=256):
                            super().__init__()
                            self.src_embed = nn.Embedding(sv, embed, padding_idx=0)
                            self.ipa_embed = nn.Embedding(iv, embed, padding_idx=0) if iv > 3 else None
                            self.tgt_embed = nn.Embedding(tv, embed, padding_idx=0)
                            self.char_encoder = nn.LSTM(embed, hidden, 3, batch_first=True, dropout=0.1)
                            self.phoneme_encoder = nn.LSTM(embed, hidden//2, 2, batch_first=True, dropout=0.1) if iv > 3 else None
                            in_dim = hidden + hidden//2 if iv > 3 else hidden
                            self.fusion = nn.Linear(in_dim, hidden)
                            self.decoder = nn.LSTM(embed + hidden, hidden, 3, batch_first=True, dropout=0.1)
                            self.attn_W = nn.Linear(hidden, hidden)
                            self.enc_to_attn = nn.Linear(hidden, embed)
                            self.out = nn.Linear(hidden, tv)
                            self.hidden, self.embed = hidden, embed

                        def generate(self, src_chars, src_ipa, max_len=15):
                            with torch.no_grad():
                                B = src_chars.size(0)
                                ce = self.src_embed(src_chars)
                                co, (ch, cc) = self.char_encoder(ce)
                                if self.phoneme_encoder is not None and src_ipa is not None:
                                    pe = self.ipa_embed(src_ipa)
                                    po, (ph, pc) = self.phoneme_encoder(pe)
                                    enc_out = self.fusion(torch.cat([co, po], dim=-1))
                                    h = torch.cat([ch[-1:], ph[-1:]], dim=-1).unsqueeze(0).repeat(3, 1, 1)
                                    c = torch.cat([cc[-1:], pc[-1:]], dim=-1).unsqueeze(0).repeat(3, 1, 1)
                                else:
                                    enc_out = self.fusion(co)
                                    h, c = ch, cc
                                dec_in = torch.tensor([[self.TGT_C2I.get("<sos>", 1)]], device=src_chars.device)
                                result = []
                                for _ in range(max_len):
                                    de = self.tgt_embed(dec_in)
                                    ep = self.enc_to_attn(enc_out)
                                    sc = torch.bmm(de, ep.transpose(1, 2))
                                    at = torch.softmax(sc / math.sqrt(self.embed), -1)
                                    ctx = torch.bmm(at, enc_out)
                                    do, (h, c) = self.decoder(torch.cat([de, ctx], -1), (h, c))
                                    tok = self.out(do).argmax(-1).item()
                                    if tok == self.TGT_C2I.get("<eos>", 2):
                                        break
                                    ch = self.TGT_I2C.get(tok, "?")
                                    if ch not in ("<pad>", "<sos>"):
                                        result.append(ch)
                                    dec_in = torch.tensor([[tok]], device=src_chars.device)
                                return "".join(result)

                    sv = len(self.SRC_C2I)
                    iv = len(self.IPA_C2I) if self.IPA_C2I else 3
                    tv = len(self.TGT_C2I)
                    self.model = DualEncoderLSTM(sv, iv, tv, d_model, state.get("embed", 256))
                    self.model.load_state_dict(state["model"])
                    self.model.eval()
                    self.backend = "transformer"

                else:
                    # Char-LSTM model (original GPU model)
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
                                se = self.src_embed(src)
                                eo, (h, c) = self.encoder(se)
                                di = torch.tensor([[self.TGT_C2I.get("<sos>", 1)]], device=src.device)
                                r = []
                                for _ in range(ml):
                                    de = self.tgt_embed(di)
                                    ep = self.enc_to_attn(eo)
                                    sc = torch.bmm(de, ep.transpose(1, 2))
                                    at = torch.softmax(sc / math.sqrt(self.embed), -1)
                                    ctx = torch.bmm(at, eo)
                                    do, (h, c) = self.decoder(torch.cat([de, ctx], -1), (h, c))
                                    tok = self.out(do).argmax(-1).item()
                                    if tok == self.TGT_C2I.get("<eos>", 2):
                                        break
                                    ch = self.TGT_I2C.get(tok, "?")
                                    if ch not in ("<pad>", "<sos>"):
                                        r.append(ch)
                                    di = torch.tensor([[tok]], device=src.device)
                                return "".join(r)

                    sv = len(self.SRC_C2I)
                    tv = len(self.TGT_C2I)
                    self.model = H(sv, tv, d_model, state.get("embed", 256))
                    self.model.load_state_dict(state["model"])
                    self.model.eval()
                    self.backend = "lstm"

                params = sum(p.numel() for p in self.model.parameters())
                print(f"[AgentA] Loaded {self.backend} model: {params:,} params from {model_path}")

            except Exception as e:
                print(f"[AgentA] Could not load model: {e}")
                print(f"[AgentA] Falling back to lookup + dictionary")
                self.model = None
                self.backend = "lookup"
        else:
            print(f"[AgentA] No model found. Using lookup + dictionary ({len(self.gold)} gold + {len(self.dict_idx)} dict)")

    def carve(self, en_word: str, top_k: int = 5) -> list[tuple[str, float, str]]:
        """Return top-K candidate French renderings with scores and sources.
        Returns: list of (fr_word, score, source) where source is 'gold'/'model'/'dict_S'/'dict_A' etc."""

        candidates = []

        # 1. Strict-gold lookup (free, trusted, score=1.0)
        if en_word in self.gold:
            fr, q = self.gold[en_word]
            candidates.append((fr, q, "gold"))

        # 2. Dictionary-v5 (tier-weighted)
        for fr, score, tier, en_ipa, usable in self.dict_idx.get(en_word, [])[:5]:
            tier_w = {"S": 1.0, "A": 0.9, "B_safe": 0.75, "B_reservoir": 0.65, "B": 0.7}
            w = tier_w.get(tier, 0.7)
            candidates.append((fr, score * w, f"dict_{tier}"))

        # 3. Neural model (transformer or LSTM)
        if self.model:
            try:
                import torch
                # Tokenize
                src_tokens = [self.SRC_C2I.get("<sos>", 1)]
                src_tokens += [self.SRC_C2I.get(c, 3) for c in en_word]
                src_tokens += [self.SRC_C2I.get("<eos>", 2)]
                src_tokens += [0] * (self.MAX_LEN - len(src_tokens))
                src_t = torch.tensor([src_tokens[:self.MAX_LEN]])

                # If IPA available, use it
                ipa_str = ""
                if en_word in self.dict_idx and self.dict_idx[en_word]:
                    ipa_str = self.dict_idx[en_word][0][3]

                ipa_tokens = [self.IPA_C2I.get("<sos>", 1)] if self.IPA_C2I else []
                ipa_tokens += [self.IPA_C2I.get(c, 3) for c in ipa_str] if self.IPA_C2I else []
                ipa_tokens += [self.IPA_C2I.get("<eos>", 2)] if self.IPA_C2I else []
                ipa_tokens += [0] * (self.MAX_LEN - len(ipa_tokens))
                ipa_t = torch.tensor([ipa_tokens[:self.MAX_LEN]]) if ipa_tokens else None

                if ipa_t is not None and len(ipa_tokens) > 0:
                    fr = self.model.generate(src_t, ipa_t)
                else:
                    fr = self.model.generate(src_t)

                if fr and len(fr) >= 2 and fr != en_word:
                    candidates.append((fr, 0.72, "model"))
            except Exception:
                pass

        # Deduplicate, keep best score
        seen = {}
        for fr, score, source in candidates:
            if fr not in seen or score > seen[fr][0]:
                seen[fr] = (score, source)
        results = [(fr, s, src) for fr, (s, src) in seen.items()]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def carve_phrase(self, en_phrase: str) -> tuple[list, str]:
        """Carve a full phrase: each word → best French."""
        words = [w.lower().strip(".,;:!?'\"") for w in en_phrase.split()
                 if w.strip(".,;:!?'\"")]
        picks = []
        for w in words:
            cands = self.carve(w, top_k=5)
            picks.append(cands[0] if cands else (w, 0.0, "miss"))
        fr_words = [fr for fr, _, _ in picks]
        return picks, " ".join(fr_words)


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    agent = AgentA()
    print(f"\nBackend: {agent.backend}")
    print(f"Gold pairs: {len(agent.gold)}")
    print(f"Dict entries: {sum(len(v) for v in agent.dict_idx.values())}")

    tests = ["ocean", "beauty", "shadow", "silent", "mountain", "river",
             "thunder", "dreamer", "wandered", "starlight"]
    for w in tests:
        cands = agent.carve(w)
        print(f"  {w:12s}:")
        for fr, score, src in cands[:3]:
            print(f"    → {fr:15s} ({score:.2f}, {src})")
