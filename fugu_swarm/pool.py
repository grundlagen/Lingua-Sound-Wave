"""fugu_swarm.pool — build an OpenFugu litellm worker pool, no ChatGPT.

The whole pool is swappable in Fugu because the checkpoint's slot labels are
just training metadata. We exploit that to enforce one hard policy invariant:
**no OpenAI / ChatGPT / GPT-family models in the pool** unless explicitly
overridden. `slot_csv` is what feeds OpenFugu's `serve.py --slot-models`.
"""
from __future__ import annotations

import os

# litellm provider prefixes that route to OpenAI/ChatGPT.
_BANNED_PREFIXES = ("openai/", "azure/", "azure_ai/")
# model-name fragments that betray a GPT/ChatGPT/Codex/o-series model
# regardless of prefix (e.g. someone proxying "gpt-4o" under a custom base url).
_BANNED_SUBSTRINGS = ("gpt-", "gpt4", "gpt3", "chatgpt", "codex", "o1-", "o3-", "o4-")


def is_openai(model_id: str) -> bool:
    """True if `model_id` resolves to an OpenAI / ChatGPT-family model."""
    m = model_id.strip().lower()
    if m.startswith(_BANNED_PREFIXES):
        return True
    return any(s in m for s in _BANNED_SUBSTRINGS)


def slot_csv(models, allow_openai: bool = False) -> str:
    """Join model ids into the comma-separated list serve.py expects.

    Raises ValueError if any model is OpenAI/ChatGPT and `allow_openai` is False
    — this is the enforced "no ChatGPT" guarantee, not just a convention.
    """
    models = [m.strip() for m in models if m and m.strip()]
    if not models:
        raise ValueError("empty worker pool")
    banned = [m for m in models if is_openai(m)]
    if banned and not allow_openai:
        raise ValueError(
            f"ChatGPT/OpenAI models are excluded by policy: {banned}. "
            "Pass allow_openai=True to override.")
    return ",".join(models)


def required_env(models) -> list[str]:
    """Env vars litellm will need for the given pool (best-effort, for preflight)."""
    need: set[str] = set()
    for m in models:
        p = m.split("/", 1)[0].lower()
        need.add({
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
            "deepseek": "DEEPSEEK_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",   # Qwen
            "openrouter": "OPENROUTER_API_KEY",
            "novita": "NOVITA_API_KEY",
            "together_ai": "TOGETHER_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }.get(p, f"{p.upper()}_API_KEY"))
    return sorted(need)


def missing_env(models) -> list[str]:
    return [k for k in required_env(models) if not os.environ.get(k)]
