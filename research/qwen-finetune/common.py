"""Shared plumbing for the Qwen fine-tune kit.

Everything that touches the existing homophone-bench code goes through here,
so the rest of the kit stays independent of where that code lives. Point
BENCH_DIR at your local research/homophone-bench checkout (env var or CLI
flag); we import matcher / g2p from there lazily.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

BENCH_DIR = Path(os.environ.get("HOMOPHONE_BENCH_DIR", "../homophone-bench")).resolve()

SYSTEM_A = (
    "You carve English sound into French. Given English text and its IPA, "
    "output French words that, read aloud in French, sound like the English. "
    "The French must be real French words. Output only the French."
)
SYSTEM_B = (
    "You are an English ear. You hear French text as if it were English. "
    "Output the English words the French sounds like when read aloud in "
    "French. Do not translate."
)


def _load_module(name: str, bench_dir: Path | None = None):
    bench = Path(bench_dir) if bench_dir else BENCH_DIR
    path = bench / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Set HOMOPHONE_BENCH_DIR (or --bench-dir) to "
            "your research/homophone-bench directory."
        )
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_matcher(bench_dir: Path | None = None):
    """Return combo_score(en_word, fr_word) -> float from matcher.py."""
    mod = _load_module("matcher", bench_dir)
    for attr in ("combo", "combo_score", "score"):
        if hasattr(mod, attr):
            return getattr(mod, attr)
    raise AttributeError("matcher.py exposes none of: combo, combo_score, score")


def load_g2p(bench_dir: Path | None = None):
    """Return en_ipa(text) -> str using lexicon_g2p.py, spaced phonemes."""
    mod = _load_module("lexicon_g2p", bench_dir)
    for attr in ("g2p", "to_ipa", "en_to_ipa", "transcribe"):
        if hasattr(mod, attr):
            fn = getattr(mod, attr)
            return lambda text: _normalize_ipa(fn(text))
    raise AttributeError("lexicon_g2p.py exposes none of: g2p, to_ipa, en_to_ipa, transcribe")


def _normalize_ipa(ipa) -> str:
    if isinstance(ipa, (list, tuple)):
        return " ".join(str(p) for p in ipa)
    return str(ipa)


def chatml_sft(system: str, user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def user_prompt_a(en: str, ipa: str) -> str:
    return f"EN: {en}\nIPA: {ipa}"


def user_prompt_b(fr: str) -> str:
    return f"FR: {fr}"


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
