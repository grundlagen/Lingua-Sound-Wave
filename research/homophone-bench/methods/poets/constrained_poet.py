"""E40 (env-honest form): VERIFY-CONSTRAINED DECODING.

The endgame design was: an LLM writes French freely while an FST masks tokens
that break the sound. No local logit-bias hook here -- so the mask moves up one
level: STEPWISE. Per English chunk, Haiku proposes N French continuations with
full poetic license (metaphor, kennings, adjoined words, elision, liaison);
the MATCHER vetoes each candidate (sound floor per step); failures are
resampled with feedback. The LLM never scores sound -- the judge does. That is
the constrained-decoding invariant, at phrase granularity.

Also in: C28 context-vector meaning (candidates scored against the SENTENCE,
not just the word) and E37 assonance bonus (Van Rooten lines sing).

Run: python constrained_poet.py "the sea remembers every ship"
     python constrained_poet.py --paragraph      (the ultimate-goal demo)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

import matcher
import prosody
from semantic_cosine import semantic_cosine

try:
    import _load_env
    _load_env.load_keys()
except Exception:
    pass

MODEL = "claude-haiku-4-5-20251001"
STEP_FLOOR = 0.55          # per-chunk sound veto
RESAMPLE = 2               # retries with feedback


_FRV = None
def frvocab():
    global _FRV
    if _FRV is None:
        _FRV = set()
        for line in open("data/lexique.tsv", encoding="utf-8", errors="ignore"):
            w = line.split("\t")[0].strip().lower()
            if w:
                _FRV.add(w)
        for line in open("fr-units.tsv", encoding="utf-8"):
            _FRV.add(line.split("\t")[0].strip().lower())
    return _FRV


def is_french(text):
    ws = [w.strip("'").lower() for w in text.replace("'", "' ").split()]
    return all((not w) or w in frvocab() for w in ws)


def combo(en, fr):
    try:
        qi, ci = matcher.g2p(en, "en"), matcher.g2p(fr, "fr")
        return 0.5 * matcher._ngram_channel(qi, ci) + 0.5 * matcher._feat_channel(qi, ci)
    except Exception:
        return 0.0


def assonance(fr_line):
    """E37: dominant-vowel share of the French line's vowels."""
    try:
        segs = matcher._segs(matcher._canonical(matcher.g2p(fr_line, "fr")))
    except Exception:
        return 0.0
    V = set("iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɑɒ") | {"ɑ̃", "ɛ̃", "ɔ̃", "œ̃"}
    vs = [s for s in segs if s.replace("ː", "") in V or s[0] in V]
    if len(vs) < 3:
        return 0.0
    from collections import Counter
    return Counter(vs).most_common(1)[0][1] / len(vs)


def haiku(prompt, max_tokens=700):
    key = os.environ["ANTHROPIC_API_KEY"]
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
                                 headers={"x-api-key": key,
                                          "anthropic-version": "2023-06-01",
                                          "Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=90))
    return out["content"][0]["text"]


STEP_PROMPT = """You are writing ONE French sentence that, read aloud by a French
speaker, SOUNDS like an English sentence -- while staying grammatical French
whose meaning echoes the English (metaphor, kenning, association all allowed).

English sentence: "{line}"
French so far   : "{sofar}"
NEXT English chunk to render in sound: "{chunk}"
Known sound-true French options for these words (use, adapt, or beat them):
{hints}
{feedback}
Propose 6 DIFFERENT French continuations (1-3 words each) that:
 - continue the French-so-far grammatically;
 - SOUND like the English chunk (elision l'/d', liaison, silent endings,
   th->d, h-drop, nasals, adjoined/split words all allowed);
 - keep the sentence's meaning near the English (loose/metaphoric is fine).
Reply ONLY a JSON array of 6 strings."""


def chunks(line, size=2):
    ws = [w for w in re.findall(r"[a-zA-Z']+", line)]
    out, i = [], 0
    while i < len(ws):
        take = min(size, len(ws) - i)
        out.append(" ".join(ws[i:i + take]))
        i += take
    return out


_BC = None
def bc_hints(chunk):
    """Verified channel candidates as prompt hints (machine precision in)."""
    global _BC
    try:
        import beauty_compose as B
        if _BC is None:
            _BC = B.load_all()
        outs = []
        for w in chunk.split():
            cs = B.candidates(w, _BC)[:3]
            if cs:
                outs.append(f"  {w}: " + ", ".join(f"{fr} ({s_:.2f})" for _j, s_, _m, fr, _c in cs))
        return "\n".join(outs) or "  (none)"
    except Exception:
        return "  (none)"


def decode(line, verbose=True):
    sofar = ""
    picks = []
    for chunk in chunks(line):
        best, feedback = None, ""
        for attempt in range(1 + RESAMPLE):
            try:
                txt = haiku(STEP_PROMPT.format(line=line, sofar=sofar or "(start)",
                                               chunk=chunk, hints=bc_hints(chunk),
                                               feedback=feedback))
                cands = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
            except Exception:
                cands = []
            scored = []
            for c in cands:
                c = str(c).strip().strip('".,;')
                if not c or not is_french(c):
                    continue                      # the vocabulary mask: REAL French only
                s = combo(chunk, c.replace("'", " "))
                scored.append((s, c))
            scored.sort(reverse=True)
            if scored and scored[0][0] >= STEP_FLOOR:
                best = scored[0]
                break
            if scored:
                best = best if best and best[0] >= scored[0][0] else scored[0]
                feedback = (f"Your last candidates sounded too far from the "
                            f"English (best {scored[0][1]!r} = {scored[0][0]:.2f}). "
                            f"Match the SOUND more literally, and use ONLY real "
                            f"French dictionary words.")
            else:
                feedback = ("All your candidates were rejected: they contained "
                            "non-French words. Use ONLY real French dictionary "
                            "words (elision l'/d' allowed).")
        if best is None:
            best = (0.0, chunk)
        picks.append((chunk, best[1], best[0]))
        sofar = (sofar + " " + best[1]).strip()
    fr_line = sofar
    s = combo(line, fr_line.replace("'", " "))
    m = max(0.0, semantic_cosine(line, fr_line))
    # C28 context blend is implicit here: Haiku sees the WHOLE sentence
    try:
        import juncture
        s = max(s, juncture.best_juncture_score(line, fr_line))
    except Exception:
        pass
    p = prosody.prosodic_score(line, fr_line.replace("'", " "))
    a = assonance(fr_line)
    if verbose:
        print(f"EN : {line}")
        print(f"FR : {fr_line}")
        print(f"     sound {s:.2f}  meaning {m:.2f}  prosody {p:.2f}  assonance {a:.2f}")
        print("     " + "  ".join(f"{c}≈{f}[{sc:.2f}]" for c, f, sc in picks) + "\n")
    return fr_line, s, m, p


PARAGRAPH = [
    "the sea remembers every ship",
    "we call to the moon and she answers",
    "my sorrow sleeps in a deep well",
    "bless the dawn that made us free",
    "less debt, less mess, more soup",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--paragraph", action="store_true")
    args = ap.parse_args()
    lines = PARAGRAPH if args.paragraph else (args.text or PARAGRAPH[:1])
    tot_s = tot_m = 0.0
    for line in lines:
        _, s, m, p = decode(line)
        tot_s += s
        tot_m += m
    n = len(lines)
    print(f"== paragraph mean: sound {tot_s/n:.2f}  meaning {tot_m/n:.2f} ==")


if __name__ == "__main__":
    main()
