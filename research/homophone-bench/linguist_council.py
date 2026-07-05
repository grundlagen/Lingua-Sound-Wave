"""Linguistics council: ask several LLMs (DeepSeek, Gemini, Nemotron) for clues
on improving METER and MEANING scoring for EN<->FR homophonic translation, and
SAVE every model's reply. Each brings a different slice of linguistic knowledge.

Providers (auto-detected from env via _load_env):
  DeepSeek  native  api.deepseek.com           (strong reasoning / linguistics)
  Gemini    native  generativelanguage API     (broad corpus)  -> OpenRouter fallback
  Nemotron  OpenRouter                          (free)
Outputs -> LINGUIST_COUNCIL.md (+ printed). Review before encoding into matcher/prosody.

Run: python linguist_council.py
"""
from __future__ import annotations

import json
import os
import urllib.request

import _load_env
_load_env.load_keys()

CONTEXT = """\
We build a homophonic-translation engine (English rewritten as French that sounds
the same, Van Rooten style). Our scorer already does:
- stress-weighted phoneme matching (espeak ˈ/ˌ): stressed syllables matter,
  unstressed 2nd/3rd syllables cheaped out;
- DIVERGED prosody: English stress-timed, early/mid peak, DECLINING pitch; French
  syllable-timed, PHRASE-FINAL prominence, even rhythm;
- onset salience, rhythm/syllable-count match;
- rhotic map (FR ʁ = EN ɹ), nasal split (ɑ̃=ɑn), schwa/offglide cheap gaps,
  liaison s/x/z->z, d/t->t, n->n.
Sample mined EN->FR phoneme-fragment matches: st->st, ɹi->ɹi, ɛk->ɛk, ks->ks,
boot->bout, grasp->grappes, Humpty Dumpty->un petit un petit."""

PROMPT = CONTEXT + """

Give CONCRETE, codeable improvements, numbered, in two parts:
(A) METER / PROSODY scoring -- French vs English metrics, syllable weight,
    e-muet count in verse, hemistich/caesura, the rising vs falling contour,
    secondary-stress handling -- what would make the sound-judge better?
(B) MEANING scoring -- how to judge that the French output is not just sound-true
    but plausibly meaningful/poetic, beyond word-validity (collocation, semantic
    coherence, image). For each: the rule, an IPA/word example, and how to encode
    (a weight, a gap, a feature, or a model call). Keep under 550 words."""


def _openai_chat(url, key, model, extra_headers=None):
    body = json.dumps({"model": model, "temperature": 0.3,
                       "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 1600}).encode()
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if extra_headers:
        h.update(extra_headers)
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def ask_deepseek():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    return _openai_chat("https://api.deepseek.com/chat/completions", key, "deepseek-chat")


def ask_nemotron():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    return _openai_chat("https://openrouter.ai/api/v1/chat/completions", key,
                        "nvidia/nemotron-3-super-120b-a12b:free")


def ask_gemini():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        for model in ("gemini-2.0-flash", "gemini-1.5-flash"):
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={key}")
            body = json.dumps({"contents": [{"parts": [{"text": PROMPT}]}]}).encode()
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.load(r)
                return d["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                continue
    # fallback: Gemini via OpenRouter
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        try:
            return _openai_chat("https://openrouter.ai/api/v1/chat/completions", k,
                                "google/gemini-flash-1.5")
        except Exception:
            return None
    return None


def main():
    council = [("DeepSeek", ask_deepseek), ("Gemini", ask_gemini),
               ("Nemotron", ask_nemotron)]
    out = ["# Linguistics council — meter & meaning improvement clues\n",
           "_Saved advisories from multiple LLMs; review before encoding._\n"]
    for name, fn in council:
        print(f"--- asking {name} ---", flush=True)
        try:
            txt = fn()
        except Exception as e:
            txt = None
            print(f"[{name} failed: {e}]")
        if txt:
            out.append(f"\n## {name}\n\n{txt}\n")
            print(txt[:800])
        else:
            out.append(f"\n## {name}\n\n(unavailable)\n")
    open("LINGUIST_COUNCIL.md", "w", encoding="utf-8").write("\n".join(out))
    print("\nsaved -> LINGUIST_COUNCIL.md")


if __name__ == "__main__":
    main()
