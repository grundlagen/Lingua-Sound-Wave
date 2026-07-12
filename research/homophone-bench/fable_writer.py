"""Judge-verified homophonic fable writer.

The one-shot idea (LLM rewrites a whole fable so it SOUNDS like the original)
made repo-lawful: the LLM only PROPOSES; every line is scored by the
matcher.py combo (judge of record) before it is kept. Nothing unverified is
ever assembled into the output text.

Two modes:
  --mode fr   EN fable line -> French that sounds like it (the dual task).
              Gates: combo sound + Lexique French-ness (franglais guard)
              + optional MiniLM meaning if sentence_transformers is present.
  --mode en   EN fable line -> surreal English mondegreen (same sound,
              different words). Gates: combo sound + surface-change (shared
              real-word fraction < 0.4, else it's a copy scoring 1.0).

Three proposal backends:
  --backend anthropic   API via .env.local key (default model: sonnet).
  --backend ollama      local ollama chat endpoint (llama3 etc).
  --backend file        proposals TSV written by a Claude session acting as
                        proposer (the in-the-loop role from CLAUDE.md):
                        emit prompts with --emit-prompts, answer them into
                        proposals.tsv (line_idx<TAB>candidate), re-run.

Per-line, not whole-fable: one-shot whole-text output drifts off the phoneme
stream within a clause; splitting into breath-lines (<=8 words) keeps each
proposal inside the judge's reliable span. --one-shot still exists so the
whole-fable variant can be benched against per-line honestly.

Tiers (sentence scope, cf. DUAL_SCALE.md sound means 0.52-0.68):
  STRONG >= 0.75    VERIFIED >= 0.60    PARTIAL >= 0.45 (recorded, not claimed)

Usage:
    python fable_writer.py --demo --emit-prompts            # in-the-loop step 1
    python fable_writer.py --demo --backend file --proposals proposals.tsv
    python fable_writer.py fable.txt --backend anthropic --mode fr
    python fable_writer.py fable.txt --backend ollama --model llama3 --mode en
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

import matcher

STRONG, VERIFIED, PARTIAL = 0.75, 0.60, 0.45
DEMO_FABLE = (
    "A hungry fox saw fine grapes. They hung high on the vine. "
    "He leapt for the fruit, but he fell short. "
    "So he walked away and said: they are sour anyway."
)

# ---------------------------------------------------------------- judging

def combo(en: str, other: str, lang: str = "fr") -> float:
    try:
        qi, ci = matcher.g2p(en, "en"), matcher.g2p(other, lang)
        return 0.5 * matcher._ngram_channel(qi, ci) + 0.5 * matcher._feat_channel(qi, ci)
    except Exception:
        return 0.0


_FRVOCAB: set | None = None


def frvocab() -> set:
    global _FRVOCAB
    if _FRVOCAB is None:
        _FRVOCAB = set()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lexique.tsv")
        for line in open(path, encoding="utf-8", errors="ignore"):
            w = line.split("\t")[0].strip().lower()
            if w:
                _FRVOCAB.add(w)
    return _FRVOCAB


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zàâäçéèêëîïôöùûüÿœæ'-]+", text.lower()) if t]


def lexique_gate(fr: str) -> float:
    """Fraction of FR tokens found in Lexique (elisions split on apostrophe)."""
    vocab = frvocab()
    toks: list[str] = []
    for t in tokens(fr):
        parts = [p for p in re.split(r"['-]", t) if p]
        toks.extend(parts if parts else [t])
    if not toks:
        return 0.0
    hit = sum(1 for t in toks if t in vocab or (t + "'") in vocab or t in ("l", "d", "j", "s", "c", "n", "m", "qu", "t"))
    return hit / len(toks)


def surface_change(src: str, cand: str) -> float:
    """EN mode guard: 1 - shared-word fraction. Copying the source scores 0."""
    a, b = set(tokens(src)), set(tokens(cand))
    if not b:
        return 0.0
    return 1.0 - len(a & b) / len(b)


_MEANING = None


def meaning(en: str, fr: str) -> float | None:
    global _MEANING
    if _MEANING is None:
        try:
            from semantic_cosine import semantic_cosine
            semantic_cosine("test", "test")  # deps load lazily inside
            _MEANING = semantic_cosine
        except Exception:
            _MEANING = False
    return _MEANING(en, fr) if _MEANING else None


def judge(src: str, cand: str, mode: str) -> dict:
    lang = "fr" if mode == "fr" else "en"
    c = combo(src, cand, lang)
    row = {"cand": cand, "combo": round(c, 3)}
    if mode == "fr":
        row["lexique"] = round(lexique_gate(cand), 3)
        m = meaning(src, cand)
        if m is not None:
            row["meaning"] = round(m, 3)
        ok_gate = row["lexique"] >= 0.999
    else:
        row["change"] = round(surface_change(src, cand), 3)
        ok_gate = row["change"] >= 0.6
    row["tier"] = (
        "STRONG" if ok_gate and c >= STRONG else
        "VERIFIED" if ok_gate and c >= VERIFIED else
        "PARTIAL" if c >= PARTIAL else
        "FAIL" if ok_gate else "GATED"
    )
    return row

# ---------------------------------------------------------------- lines

def breath_lines(text: str, max_words: int = 8) -> list[str]:
    parts = re.split(r"(?<=[.,;:!?])\s+", text.replace("\n", " ").strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        ws = p.split()
        while len(ws) > max_words:
            out.append(" ".join(ws[:max_words]))
            ws = ws[max_words:]
        if ws:
            out.append(" ".join(ws))
    return out

# ---------------------------------------------------------------- backends

PROMPT_FR = """Rewrite this English line as COHERENT FRENCH that, read aloud
by a French speaker, SOUNDS as close as possible to the English. Real French
words only (no anglicisms). Meaning may drift; sound may not. Give {k}
different attempts, one per line, no numbering, no commentary.

English line: {line}"""

PROMPT_EN = """Rewrite this English line as a surreal English mondegreen:
nearly identical SOUND, different words and meaning. Change at least 60% of
the words. Give {k} different attempts, one per line, no numbering,
no commentary.

English line: {line}"""


def call_anthropic(prompt: str, model: str, max_tokens: int = 600) -> str:
    import _load_env
    _load_env.load_keys()
    key = os.environ["ANTHROPIC_API_KEY"]
    body = json.dumps({
        "model": model, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=120))
    return out["content"][0]["text"]


def call_ollama(prompt: str, model: str) -> str:
    body = json.dumps({
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.8},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))
    return out["message"]["content"]


def propose(lines: list[str], args) -> dict[int, list[str]]:
    props: dict[int, list[str]] = {i: [] for i in range(len(lines))}
    if args.backend == "file":
        for raw in open(args.proposals, encoding="utf-8"):
            raw = raw.rstrip("\n")
            if not raw or raw.startswith("#"):
                continue
            idx, cand = raw.split("\t", 1)
            props[int(idx)].append(cand.strip())
        return props
    tmpl = PROMPT_FR if args.mode == "fr" else PROMPT_EN
    call = call_anthropic if args.backend == "anthropic" else call_ollama
    if args.one_shot:
        whole = tmpl.format(k=1, line=" / ".join(lines))
        text = call(whole, args.model)
        for i, seg in enumerate(re.split(r"\s*/\s*", text.strip())[: len(lines)]):
            props[i].append(seg.strip())
        return props
    for i, line in enumerate(lines):
        text = call(tmpl.format(k=args.k, line=line), args.model)
        for cand in text.strip().splitlines():
            cand = cand.strip().strip("-• ").strip()
            if cand:
                props[i].append(cand)
        print(f"  proposed {len(props[i])} for line {i}: {line}", file=sys.stderr)
    return props

# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fable", nargs="?", help="path to fable .txt (or --demo)")
    ap.add_argument("--demo", action="store_true", help="use the built-in PD Aesop demo fable")
    ap.add_argument("--mode", choices=["fr", "en"], default="fr")
    ap.add_argument("--backend", choices=["anthropic", "ollama", "file"], default="file")
    ap.add_argument("--model", default="claude-sonnet-5", help="anthropic model id or ollama model name")
    ap.add_argument("--k", type=int, default=5, help="candidates per line")
    ap.add_argument("--proposals", default="proposals.tsv", help="file backend: idx<TAB>candidate")
    ap.add_argument("--one-shot", action="store_true", help="whole-fable single call (benchmark vs per-line)")
    ap.add_argument("--emit-prompts", action="store_true", help="print per-line prompts for an in-the-loop proposer, then exit")
    ap.add_argument("--out", default="fable-out")
    args = ap.parse_args()

    text = DEMO_FABLE if args.demo else open(args.fable, encoding="utf-8").read()
    lines = breath_lines(text)

    if args.emit_prompts:
        tmpl = PROMPT_FR if args.mode == "fr" else PROMPT_EN
        for i, line in enumerate(lines):
            print(f"### line {i}\n{tmpl.format(k=args.k, line=line)}\n")
        return

    props = propose(lines, args)

    rows, assembled = [], []
    for i, line in enumerate(lines):
        scored = sorted((judge(line, c, args.mode) for c in props[i]),
                        key=lambda r: -r["combo"])
        best = next((r for r in scored if r["tier"] in ("STRONG", "VERIFIED")), None)
        for r in scored:
            rows.append({"idx": i, "src": line, **r, "kept": r is best})
        assembled.append(best["cand"] if best else f"[unverified: {line}]")
        tag = best["tier"] if best else "NONE"
        top = scored[0] if scored else None
        print(f"[{i}] {tag:<8} {line!r} -> {top['cand'] if top else '(no proposals)'!r}"
              + (f"  combo={top['combo']}" if top else ""))

    with open(f"{args.out}.tsv", "w", encoding="utf-8") as f:
        cols = ["idx", "src", "cand", "combo", "lexique", "meaning", "change", "tier", "kept"]
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    with open(f"{args.out}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(assembled) + "\n")

    kept = sum(1 for a in assembled if not a.startswith("[unverified"))
    strong = sum(1 for r in rows if r["kept"] and r["tier"] == "STRONG")
    print(f"\n{kept}/{len(lines)} lines verified ({strong} STRONG) "
          f"-> {args.out}.txt / {args.out}.tsv")
    if kept < len(lines):
        print("unverified lines stay bracketed — propose again for those indices only")


if __name__ == "__main__":
    main()
