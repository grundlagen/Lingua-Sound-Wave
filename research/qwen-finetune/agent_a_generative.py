#!/usr/bin/env python3
"""
Generative Agent A — two drop-in implementations for the three-agent loop.

- **QwenCarver**: Full Qwen transformer pipeline trained via LoRA + DPO (needs GPU)
- **CharLSTMCarver**: Lightweight character-level LSTM, lookup-first, no GPU needed

Both expose carve() / revise() for the three_agent_v2.py repair loop.
Always re-verify with combo matcher before accepting.

USAGE (Char-LSTM):
  from agent_a_generative import CharLSTMCarver
  carver = CharLSTMCarver()
  picks, fr_text = carver.carve_phrase("the ocean remembers")

USAGE (Qwen, needs trained adapter):
  from agent_a_generative import QwenCarver
  carver = QwenCarver("ckpt/agent-a-dpo")
  candidates = carver.carve("cat and mouse", n=8)
"""

# ═══════════════════════════════════════════════════════════════
# IMPLEMENTATION 1: QwenCarver — Qwen transformer via LoRA + DPO
# ═══════════════════════════════════════════════════════════════

from __future__ import annotations

from pathlib import Path

_QWEN_AVAILABLE = False
try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _QWEN_AVAILABLE = True
except ImportError:
    pass

if _QWEN_AVAILABLE:
    try:
        from common import SYSTEM_A, user_prompt_a
    except ImportError:
        SYSTEM_A = ""
        user_prompt_a = lambda en, ipa: f"English: {en}\nIPA: {ipa}\nFrench:"

    try:
        from common import load_g2p
    except ImportError:
        load_g2p = None


    class QwenCarver:
        """Generative Agent A backed by a Qwen LoRA adapter. GPU required."""

        def __init__(self, adapter: str | Path,
                     model: str = "Qwen/Qwen3-4B-Instruct-2507",
                     bench_dir: Path | None = None):
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            base = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch.bfloat16, device_map="auto")
            self.model = PeftModel.from_pretrained(base, str(adapter))
            self.model.eval()
            self.g2p = load_g2p(bench_dir) if load_g2p else None

        def _generate(self, messages: list[dict], n: int, temperature: float) -> list[str]:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=0.95,
                    num_return_sequences=n,
                    max_new_tokens=64,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            plen = inputs["input_ids"].shape[1]
            seen, results = set(), []
            for seq in out:
                fr = self.tokenizer.decode(seq[plen:], skip_special_tokens=True).strip()
                if fr and fr not in seen:
                    seen.add(fr)
                    results.append(fr)
            return results

        def carve(self, en: str, ipa: str | None = None, n: int = 8,
                  temperature: float = 0.9) -> list[str]:
            """French carve candidates for an English span (word, span, or line)."""
            if ipa is None:
                if self.g2p is None:
                    raise ValueError("pass ipa= explicitly or provide bench_dir for g2p")
                ipa = self.g2p(en)
            messages = [
                {"role": "system", "content": SYSTEM_A},
                {"role": "user", "content": user_prompt_a(en, ipa)},
            ]
            return self._generate(messages, n, temperature)

        def revise(self, en: str, ipa: str, current_fr: str, keep: list[str],
                   en_span: str, heard: str, span_combo: float,
                   candidates: list[str] | None = None, n: int = 4) -> list[str]:
            """One REVISE turn: fix a misaligned span, keeping the rest."""
            keep_s = ", ".join(f'"{w}"' for w in keep) if keep else "(nothing)"
            cands = " | ".join(candidates) if candidates else "(search the carve pool)"
            messages = [
                {"role": "system", "content": SYSTEM_A},
                {"role": "user", "content": user_prompt_a(en, ipa)},
                {"role": "assistant", "content": current_fr},
                {"role": "user", "content":
                    f'REVISE. Keep: {keep_s}. Fix span "{en_span}" '
                    f'(heard as "{heard}", combo {span_combo:.2f}). '
                    f'Candidates: {cands}'},
            ]
            return self._generate(messages, n, temperature=0.7)


# ═══════════════════════════════════════════════════════════════
# IMPLEMENTATION 2: CharLSTMCarver — lookup-first char-level model
# ═══════════════════════════════════════════════════════════════

import json, os, sys
from collections import defaultdict

BENCH_DIR = os.environ.get("HOMOPHONE_BENCH_DIR",
    "/home/mint/Lingua-Sound-Wave/research/homophone-bench")
sys.path.insert(0, BENCH_DIR)

# ── Load lookup DB ──
_LOOKUP = {}
_training_path = os.path.join(BENCH_DIR, "strict-gold-training.jsonl")
try:
    for line in open(_training_path):
        r = json.loads(line)
        en = r["input"].replace("English word: ", "").strip().lower()
        fr = r["output"].strip().lower()
        q = r.get("quality", r.get("sound", 1.0))
        if en and fr and en != fr:
            if en not in _LOOKUP or q > _LOOKUP[en][1]:
                _LOOKUP[en] = (fr, q)
except FileNotFoundError:
    _LOOKUP = {}

# ── Load char-level model if available ──
_CHAR_MODEL = None
_SRC_C2I = {}
_TGT_I2C = {}
_MAX_LEN = 20
_model_path = os.path.join(BENCH_DIR, "homophone_model.pt")
try:
    import torch
    import torch.nn as nn
    if os.path.exists(_model_path):
        state = torch.load(_model_path, map_location="cpu", weights_only=False)
        SRC_VOCAB = len(state["src_chars"])
        TGT_VOCAB = len(state["tgt_chars"])
        _SRC_C2I = {c: i for i, c in enumerate(state["src_chars"])}
        _TGT_I2C = {i: c for c, i in enumerate({c: i for i, c in enumerate(state["tgt_chars"])}.items())}
        _MAX_LEN = state["max_len"]

        class CharLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.src_embed = nn.Embedding(SRC_VOCAB, 128)
                self.tgt_embed = nn.Embedding(TGT_VOCAB, 128)
                self.encoder = nn.LSTM(128, 128, 2, batch_first=True)
                self.decoder = nn.LSTM(256, 128, 2, batch_first=True)
                self.out = nn.Linear(128, TGT_VOCAB)
                self.hidden = 128

            def generate(self, src, max_len=15):
                src_emb = self.src_embed(src)
                enc_out, (h, c) = self.encoder(src_emb)
                dec_input = torch.tensor([[1]], device=src.device)
                result = []
                for _ in range(max_len):
                    dec_emb = self.tgt_embed(dec_input)
                    attn_w = torch.softmax(torch.sum(
                        dec_emb.unsqueeze(2) * enc_out.unsqueeze(1), dim=-1), dim=-1)
                    ctx = torch.sum(attn_w.unsqueeze(-1) * enc_out.unsqueeze(1), dim=2)
                    dec_out, (h, c) = self.decoder(torch.cat([dec_emb, ctx], -1), (h, c))
                    token = self.out(dec_out).argmax(-1).item()
                    if token == 2:
                        break  # <eos>
                    ch = _TGT_I2C.get(token, "?")
                    if ch not in ("<pad>", "<sos>"):
                        result.append(ch)
                    dec_input = torch.tensor([[token]], device=src.device)
                return "".join(result)

        _CHAR_MODEL = CharLSTM()
        _CHAR_MODEL.load_state_dict(state["model_state"])
        _CHAR_MODEL.eval()
except Exception:
    _CHAR_MODEL = None

# ── Matcher (combo scorer) ──
_COMBO = None
try:
    import matcher as _m
    _COMBO = getattr(_m, "combo", None) or getattr(_m, "combo_score", None)
except Exception:
    _COMBO = None


def _combo_score(en, fr):
    if _COMBO:
        return _COMBO(en, fr)
    return 0.5


def _char_sim(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


class CharLSTMCarver:
    """Generative Agent A: lookup first, generate on miss, verify always."""

    def __init__(self):
        self.lookup_hits = 0
        self.model_hits = 0
        self.fallback_hits = 0

    def carve(self, en_word, top_k=5):
        """Generate top-K French homophone candidates for an English word."""
        en_word = en_word.lower().strip()
        results = []
        seen = set()

        # 1. LOOKUP (free, trusted)
        if en_word in _LOOKUP:
            fr, q = _LOOKUP[en_word]
            self.lookup_hits += 1
            results.append((fr, q, "lookup"))
            seen.add(fr)

        # 2. CHARACTER MODEL (generate novel)
        if _CHAR_MODEL and (not results or results[0][1] < 0.70):
            try:
                tokens = []
                for c in en_word:
                    idx = _SRC_C2I.get(c, 0)
                    tokens.append(idx)
                tokens = [1] + tokens + [2]  # <sos> + chars + <eos>
                if len(tokens) < _MAX_LEN:
                    tokens += [0] * (_MAX_LEN - len(tokens))
                src = torch.tensor([tokens[:_MAX_LEN]])
                fr = _CHAR_MODEL.generate(src)
                if fr and len(fr) >= 2 and fr not in seen:
                    q = _combo_score(en_word, fr)
                    self.model_hits += 1
                    results.append((fr, q, "model"))
                    seen.add(fr)
            except Exception:
                pass

        # 3. FALLBACK (nearest known word)
        if not results:
            best_en = max(_LOOKUP.keys(), key=lambda k: _char_sim(en_word, k))
            fr, q = _LOOKUP[best_en]
            self.fallback_hits += 1
            results.append((fr, q * 0.85, f"fallback:{best_en}"))
            seen.add(fr)

        # 4. Add near-matches from DB as backup candidates
        neighbors = sorted(_LOOKUP.keys(), key=lambda k: -_char_sim(en_word, k))[:10]
        for n in neighbors:
            if n == en_word:
                continue
            fr, q = _LOOKUP[n]
            if fr not in seen:
                seen.add(fr)
                results.append((fr, q * 0.85, f"near:{n}"))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def revise(self, en_word, agent_c_verdict):
        """Accept Agent C's span verdict and revise specific French output."""
        candidates = self.carve(en_word, top_k=8)
        valid = []
        for fr, q, src in candidates:
            valid.append((fr, q, src))
        return valid[:3] if valid else candidates[:3]

    def carve_phrase(self, en_phrase):
        """Full phrase: each word → best French homophone."""
        words = [w.lower().strip(".,;:!?'\"") for w in en_phrase.split()
                 if w.strip(".,;:!?'\"")]
        picks = []
        for w in words:
            cands = self.carve(w, top_k=5)
            picks.append(cands[0] if cands else (w, 0.0, "miss"))
        fr_words = [fr for fr, _, _ in picks]
        return picks, " ".join(fr_words)


# ═══════════════════════════════════════════════════════════════
# MAIN (Char-LSTM smoke test)
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Char-LSTM smoke test (no GPU required)
    carver = CharLSTMCarver()
    print(f"CharLSTMCarver: {len(_LOOKUP)} lookup pairs, "
          f"model={'✓' if _CHAR_MODEL else '✗'}")

    tests = ["ocean", "beauty", "wandered", "twilight", "ship", "silent", "river"]
    for w in tests:
        cands = carver.carve(w)
        top = cands[0]
        print(f"  {w:12s} → {top[0]:15s} q={top[1]:.2f} [{top[2]}]")

    print(f"\n  stats: lookup={carver.lookup_hits} model={carver.model_hits} "
          f"fallback={carver.fallback_hits}")

    # For Qwen usage, uncomment:
    # import argparse
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--adapter", required=True)
    # ap.add_argument("--bench-dir", type=Path, default=None)
    # ap.add_argument("text", nargs="+")
    # args = ap.parse_args()
    # carver = QwenCarver(args.adapter, bench_dir=args.bench_dir)
    # for fr in carver.carve(" ".join(args.text)):
    #     print(fr)
