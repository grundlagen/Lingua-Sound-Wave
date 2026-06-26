"""French coherence scoring via a real LLM (the L2-model upgrade) -- multi-provider.

Replaces the bigram fluency as a FINAL RE-RANK only (a handful of calls, not one
per beam node). Providers, auto-detected from env keys (loaded by _load_env):
  OPENROUTER_API_KEY  -> openrouter.ai      (route to Nemotron / Claude / etc.)
  NVIDIA_NIM_API_KEY  -> integrate.api.nvidia.com  (Nemotron directly)
  ANTHROPIC_API_KEY   -> api.anthropic.com  (Claude)
Falls back to the bigram LM if no key / no network.

Secrets are never printed or committed; keys come from .env.local at runtime.

Usage:
  from fr_coherence import FRCoherence
  s = FRCoherence()                 # auto-detects provider, else bigram fallback
  s.batch(["t elle est elle", ...]) # -> [0.0-1.0, ...]  (one API call)
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

try:
    import _load_env
    _load_env.load_keys()
except Exception:
    pass

# OpenAI-compatible endpoints (OpenRouter and NVIDIA NIM both speak this).
PROVIDERS = [
    ("openrouter", "OPENROUTER_API_KEY",
     "https://openrouter.ai/api/v1/chat/completions",
     os.environ.get("FR_MODEL", "nvidia/nemotron-4-340b-instruct")),
    ("nim", "NVIDIA_NIM_API_KEY",
     "https://integrate.api.nvidia.com/v1/chat/completions",
     os.environ.get("FR_MODEL_NIM", "nvidia/llama-3.1-nemotron-70b-instruct")),
]

PROMPT = (
    "You are a strict French grammar judge. For each numbered French phrase, rate "
    "how coherent it is as real French (word order, agreement, sense) from 0 to "
    "100. Reply ONLY with a JSON array of integers, one per phrase, in order.\n\n"
)


class FRCoherence:
    def __init__(self):
        self.provider = None
        for name, env, url, model in PROVIDERS:
            if os.environ.get(env):
                self.provider = (name, os.environ[env], url, model)
                break
        self._bigram = None
        if self.provider is None:
            try:
                import bigram_lm
                self._bigram = bigram_lm.load("fr")
            except Exception:
                pass

    def available(self):
        return self.provider is not None

    def _bigram_score(self, fr):
        if self._bigram is None:
            return min(1.0, len(fr.split()) / 6.0)
        return self._bigram.fluency([w.lower() for w in fr.split()])

    def batch(self, phrases):
        if not phrases:
            return []
        if self.provider is None:
            return [self._bigram_score(p) for p in phrases]
        name, key, url, model = self.provider
        listing = "\n".join(f"{i+1}. {p}" for i, p in enumerate(phrases))
        body = json.dumps({
            "model": model, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT + listing}],
            "max_tokens": 30 + 6 * len(phrases),
        }).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.load(r)
            txt = out["choices"][0]["message"]["content"]
            nums = re.findall(r"\d+", txt)
            scores = [min(1.0, int(n) / 100.0) for n in nums[:len(phrases)]]
            if len(scores) == len(phrases):
                return scores
        except Exception as e:
            print(f"[fr_coherence: {name} failed ({e}); bigram fallback]")
        return [self._bigram_score(p) for p in phrases]


if __name__ == "__main__":
    s = FRCoherence()
    print("provider:", s.provider[0] if s.provider else "bigram-fallback")
    tests = ["t elle est elle en hâte avec un tel", "oui doue ci de groupe",
             "un petit un petit et on vol", "chie cède telle mi mot"]
    for p, sc in zip(tests, s.batch(tests)):
        print(f"  {sc:.2f}  {p}")
