"""DUAL POET -- the developed composer: beam over candidate LISTS (not greedy
top-1), conjugation families (French silent morphology = free grammar), trigram
L2 Viterbi, optional Haiku grammar-fixer whose edits must PRESERVE sound.

Routes per word come from beauty_compose.candidates (dual / ladder GOLD / zipf
glue / transitive synonym chains / Haiku bridges / metaphor). New here:

  CONJUGATION FAMILY  every French candidate expands to its sound-identical
      morphology family (donne/donnes/donnent, petit/petits) -- Lexique words
      grouped by identical espeak IPA. The L2 picks the grammatical member;
      sound cannot change by construction.
  BEAM + TRIGRAM      Viterbi over top-K candidates per word scored by
      joint = candidate_joint x P_trigram(word | prev two) -- grammar improves
      mechanically, no LLM needed.
  HAIKU FIXER (--fix) one pass: "repair agreement/articles/elision ONLY via
      sound-preserving edits"; verified -- kept only if combo drops < 0.05 and
      meaning does not fall.

Run: python dual_poet.py "the dog at the door made me cry" [--fix]
     python dual_poet.py --bench 40
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from collections import defaultdict
from functools import lru_cache

import matcher
import trigram_lm
import beauty_compose as BC
from semantic_cosine import semantic_cosine

try:
    import _load_env
    _load_env.load_keys()
except Exception:
    pass


# --------------------------------------------------- conjugation family index
@lru_cache(maxsize=1)
def family_index():
    """Lexique word -> sound-identical siblings (grouped by espeak IPA)."""
    by_ipa = defaultdict(set)
    words = set()
    for line in open("data/lexique.tsv", encoding="utf-8", errors="ignore"):
        w = line.split("\t")[0].strip().lower()
        if w and w.isalpha() and len(w) < 14:
            words.add(w)
    # espeak per word is slow; only families for words we actually meet (lazy)
    return words, by_ipa


def family(fr_word):
    words, by_ipa = family_index()
    sibs = {fr_word}
    # cheap morphology family: same stem +- French silent endings
    SIL = ["", "s", "e", "es", "ent", "x", "t", "d"]
    for base_cut in range(0, 4):
        stem = fr_word[: len(fr_word) - base_cut] if base_cut else fr_word
        if len(stem) < 3:
            break
        for suf in SIL:
            cand = stem + suf
            if cand in words:
                try:
                    if matcher.g2p(cand, "fr") == matcher.g2p(fr_word, "fr"):
                        sibs.add(cand)
                except Exception:
                    pass
    return sibs


# --------------------------------------------------------------- Viterbi beam
def compose(line, D, K=4, beam=24):
    ws = [w.lower().strip(".,;:!?'") for w in line.split() if w.strip(".,;:!?'")]
    FR = trigram_lm.load("fr")
    cand_lists = []
    for w in ws:
        cs = BC.candidates(w, D)[:K]
        opts = []
        seen = set()
        for j, s, m, fr, ch in cs:
            for sib in family(fr.split()[-1]) if " " not in fr else {fr}:
                fr2 = fr if " " in fr else sib
                if fr2 in seen:
                    continue
                seen.add(fr2)
                opts.append((j, s, m, fr2, ch))
        if not opts:
            opts = [(0.05, 0.0, 0.0, w, "MISS")]
        good = [o for o in opts if o[1] >= 0.45]     # sound floor: grammar may
        cand_lists.append((good or opts)[: K * 3])   # not be bought with sound

    # beam over (prev2, prev1) with trigram
    beams = [((["<s>", "<s>"]), [], 0.0)]
    for opts in cand_lists:
        nxt = []
        for ctx, picks, lp in beams:
            for j, s, m, fr, ch in opts:
                toks = fr.split()
                lp2, c = lp, ctx[:]
                for t in toks:
                    lp2 += math.log(FR.cond(c[-2], c[-1], t) + 1e-10)
                    c = [c[-1], t]
                lp2 += 3.2 * math.log(j + 1e-9)     # candidate joint dominates
                nxt.append((c, picks + [(fr, ch, s, m)], lp2))
        nxt.sort(key=lambda x: -x[2])
        beams = nxt[:beam]
    _, picks, _ = beams[0]
    fr_line = " ".join(p[0] for p in picks)
    return ws, picks, fr_line


def verify(en_line, fr_line):
    s = BC.combo(en_line, fr_line.replace("'", " "))
    m = max(0.0, semantic_cosine(en_line, fr_line))
    return s, m


# ------------------------------------------------------------- haiku fixer
def haiku_fix(en_line, fr_line):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    prompt = (
        "Repair ONLY grammar (article gender, agreement, elision like le->l', "
        "verb agreement using SILENT endings) in this French line. You may NOT "
        "change how it sounds when read aloud -- only silent-letter or "
        "article/elision repairs. Keep every content word. Reply ONLY the "
        f"fixed French line.\n\nFrench: {fr_line}")
    body = json.dumps({"model": "claude-haiku-4-5-20251001", "max_tokens": 200,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
                                 headers={"x-api-key": key,
                                          "anthropic-version": "2023-06-01",
                                          "Content-Type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=60))
        fixed = out["content"][0]["text"].strip().strip('"')
        return fixed.split("\n")[0]
    except Exception:
        return None


def run_line(line, D, fix=False, show=True):
    ws, picks, fr_line = compose(line, D)
    s, m = verify(line, fr_line)
    fixed_note = ""
    if fix:
        fixed = haiku_fix(line, fr_line)
        if fixed and fixed != fr_line:
            s2, m2 = verify(line, fixed)
            if s2 >= s - 0.05 and m2 >= m - 0.05:      # sound-preserving only
                fr_line, s, m = fixed, s2, m2
                fixed_note = "  (haiku-fixed)"
    if show:
        print(f"EN : {line}")
        print(f"FR : {fr_line}{fixed_note}")
        print(f"     sound {s:.2f}  meaning {m:.2f}")
        print("     " + "  ".join(f"{w}≈{p[0]}[{p[1]};{p[2]:.2f}]"
                                  for w, p in zip(ws, picks)) + "\n")
    return s, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()
    D = BC.load_all()
    if args.bench:
        band = n = 0
        for i, line in enumerate(open("corpus-carves.tsv", encoding="utf-8")):
            if i == 0 or n >= args.bench:
                continue
            en = line.split("\t")[0]
            s, m = run_line(en, D, fix=args.fix, show=(n < 5))
            band += (s >= 0.55 and m >= 0.45)
            n += 1
        print(f"DUAL POET bench: {band}/{n} = {band/max(1,n):.0%} in the Rooten band")
        return
    for line in (args.text or ["the dog at the door made me cry"]):
        run_line(line, D, fix=args.fix)


if __name__ == "__main__":
    main()
