"""llm_layer.py — the fluency layer (LLM_RECIPE Job 3), DeepSeek-ready.

The deterministic engine guarantees the SOUND and the per-word meaning
options; it cannot make the target line read as fluent grammar, because word
order follows the source sound stream. That last step is generation, and
this is where an LLM belongs — as arranger/judge ON TOP of the engine, never
inventing phonetics (the engine owns those).

Key handling: reads DEEPSEEK_API_KEY (or OPENAI_API_KEY + OPENAI_BASE_URL)
from the ENVIRONMENT only. Nothing is hardcoded, logged, or printed. Set it:

    export DEEPSEEK_API_KEY=sk-...

No SDK required (uses urllib). If no key is set, callers fall back to the
raw deterministic output with a one-line notice.

Two jobs:
  arrange_line(src, options)  given the source line and, per source word, the
      candidate target renderings (each a real homophone fragment with its
      sound score), write the most fluent target-language line that keeps
      those sounds and preserves meaning. The model SELECTS and lightly
      connects; it must not stray from the offered sounds.
  paraphrase(src, n)          propose n fluent source paraphrases to widen
      the decoder's search (the +0.25 sem lever measured earlier).
"""
from __future__ import annotations

import json
import os
import urllib.request


def _endpoint():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return "https://api.deepseek.com/chat/completions", key, "deepseek-chat"
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return base.rstrip("/") + "/chat/completions", key, os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return None, None, None


def available() -> bool:
    return _endpoint()[1] is not None


def _chat(messages, temperature=0.7, max_tokens=400):
    url, key, model = _endpoint()
    if not key:
        return None
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"__error__:{type(e).__name__}"


def arrange_line(src: str, tgt_lang: str, options: list[dict]) -> dict | None:
    """options: [{"src_word":..., "renderings":[{"tgt":..., "sound":0.9}, ...]}]
    Returns {"line": fluent target line, "note": rationale} or None."""
    if not available():
        return None
    opt_text = "\n".join(
        f'  "{o["src_word"]}" sounds like: '
        + ", ".join(f'{r["tgt"]}({r["sound"]:.2f})' for r in o["renderings"][:5])
        for o in options)
    sys_msg = (
        f"You assemble HOMOPHONIC translations into {tgt_lang}. You are given an "
        f"English line and, per word, {tgt_lang} fragments that SOUND like that "
        f"word (with sound-match scores). Write ONE {tgt_lang} line that (a) keeps "
        f"the overall SOUND of the English as closely as possible by using the "
        f"offered fragments in order, (b) reads as grammatical, evocative "
        f"{tgt_lang}, and (c) leans toward the English meaning where the sounds "
        f"allow. You may add small function words and inflect, but do not "
        f"introduce sounds far from those offered. Reply as JSON: "
        f'{{"line": "...", "gloss": "literal back-translation", "note": "..."}}')
    usr = f"English line: {src!r}\nPer-word {tgt_lang} sound options:\n{opt_text}"
    out = _chat([{"role": "system", "content": sys_msg},
                 {"role": "user", "content": usr}], temperature=0.8)
    if not out or out.startswith("__error__"):
        return {"line": None, "error": out}
    try:
        s = out[out.index("{"):out.rindex("}") + 1]
        return json.loads(s)
    except Exception:
        return {"line": out.strip(), "gloss": "", "note": "unparsed"}


def paraphrase(src: str, n: int = 4) -> list[str]:
    if not available():
        return []
    out = _chat([{"role": "system", "content":
                  "Rephrase the line keeping its meaning, varied wording, same "
                  "register. Reply as a JSON list of strings only."},
                 {"role": "user", "content": src}], temperature=0.9)
    if not out or out.startswith("__error__"):
        return []
    try:
        return [s for s in json.loads(out[out.index("["):out.rindex("]") + 1])
                if isinstance(s, str)][:n]
    except Exception:
        return []


if __name__ == "__main__":
    import sys
    if not available():
        print("No key in env. Set one to enable the fluency layer:")
        print("    export DEEPSEEK_API_KEY=sk-...")
        sys.exit(0)
    print("LLM layer live via", _endpoint()[2])
    demo = arrange_line("the quiet sea", "French", [
        {"src_word": "quiet", "renderings": [{"tgt": "soft", "sound": 1.0},
                                             {"tgt": "coi", "sound": 0.9}]},
        {"src_word": "sea", "renderings": [{"tgt": "si", "sound": 1.0},
                                          {"tgt": "scie", "sound": 1.0}]}])
    print(json.dumps(demo, ensure_ascii=False, indent=2))
