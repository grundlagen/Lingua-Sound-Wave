"""Generative Agent A: drop-in candidate provider for the three-agent loop.

Replaces (or augments) the 6,143-pair lookup in three_agent_v2.py. Exposes:

    carver = GenerativeCarver("ckpt/agent-a-dpo")
    candidates = carver.carve("cat and mouse", n=8)      # full-line carves
    fixed = carver.revise(en, ipa, current_fr, keep, en_span, heard,
                          span_combo, candidates=[...])  # one REVISE turn

Wire-in (local three_agent_v2.py): wherever Agent A looks up FR candidates
per word and fails, call carver.carve() on the span instead — the generative
model produces novel French spellings the database lacks, which is what lets
the repair loop converge. Keep the lookup as the first try (it's free and
verified); fall back to generation on misses, and always re-verify generated
output with the combo matcher before accepting it.
"""

from __future__ import annotations

from pathlib import Path

from common import SYSTEM_A, user_prompt_a

try:
    from common import load_g2p
except ImportError:  # pragma: no cover
    load_g2p = None


class GenerativeCarver:
    def __init__(self, adapter: str | Path,
                 model: str = "Qwen/Qwen3-4B-Instruct-2507",
                 bench_dir: Path | None = None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        base = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map="auto")
        self.model = PeftModel.from_pretrained(base, str(adapter))
        self.model.eval()
        self.g2p = load_g2p(bench_dir) if load_g2p else None

    def _generate(self, messages: list[dict], n: int, temperature: float) -> list[str]:
        import torch

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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--bench-dir", type=Path, default=None)
    ap.add_argument("text", nargs="+")
    args = ap.parse_args()
    carver = GenerativeCarver(args.adapter, bench_dir=args.bench_dir)
    for fr in carver.carve(" ".join(args.text)):
        print(fr)
