"""multilang.py — the button. One language-pair config in, the whole
homophonic+semantic loop out.

    python multilang.py en es              # build + mine + translate, en->es
    python multilang.py en es "the sea is cold and the moon is bright"

Everything language-specific lives in PAIRS below; the engine is generic.
The matcher core (matcher.py: panphon featural NW + sharpened costs + ngram
Dice) is already language-agnostic — it scores any two IPA strings — so a
new pair needs only: espeak voices, wordfreq codes, and a MUSE bilingual
dict for the meaning edges. Per-pair artifacts land in pairs/<src>-<tgt>/.

The hand-tuned EN<->FR equivalence table in matcher.py is a *floor* under
the panphon baseline, so other pairs still work at panphon quality out of
the box; learned-cost overlays (learn_costs.py) specialize each pair later.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from wordfreq import top_n_list, zipf_frequency

import matcher
from matcher import _canonical, _ngram_channel, nw_sim_ipa, _segs


@dataclass
class LangPair:
    src: str            # espeak voice + wordfreq code for source
    tgt: str
    espeak_src: str
    espeak_tgt: str
    wf_src: str
    wf_tgt: str
    muse_url: str
    rtl: bool = False
    notes: str = ""
    tgt_wordlist: str = ""   # optional path if wordfreq lacks the language


PAIRS = {
    "en-fr": LangPair("en", "fr", "en-us", "fr", "en", "fr",
                      "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-fr.txt",
                      notes="reference pair; hand-tuned equivalence table applies"),
    "en-es": LangPair("en", "es", "en-us", "es", "en", "es",
                      "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-es.txt",
                      notes="Spanish: shallow orthography, espeak G2P excellent"),
    "en-ga": LangPair("en", "ga", "en-us", "ga", "en", "en",
                      "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ga.txt",
                      notes="Irish: wordfreq has no 'ga' — needs a real Irish "
                            "wordlist (tgt_wordlist); espeak 'ga' G2P works"),
    "en-he": LangPair("en", "he", "en-us", "he", "en", "he",
                      "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-he.txt",
                      rtl=True,
                      notes="Hebrew: RTL; espeak needs native-script input "
                            "(works on Hebrew words, not transliterations)"),
    "ar-he": LangPair("ar", "he", "ar", "he", "ar", "he",
                      "",
                      rtl=True,
                      notes="Arabic<->Hebrew: both RTL, both Semitic; no MUSE "
                            "pair shipped — meaning edges need another source"),
}


def workdir(lp: LangPair) -> str:
    d = os.path.join("pairs", f"{lp.src}-{lp.tgt}")
    os.makedirs(d, exist_ok=True)
    return d


def batch_g2p(words: list[str], voice: str) -> dict[str, str]:
    """One espeak process per 800-word chunk; '.' terminator keeps words
    as separate utterances (blocks liaison/sandhi across the list)."""
    out: dict[str, str] = {}
    CH = 800
    for i in range(0, len(words), CH):
        chunk = words[i:i + CH]
        text = "".join(w + ".\n" for w in chunk)
        r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, "--stdin"],
                           input=text, capture_output=True, text=True)
        lines = [ln for ln in r.stdout.split("\n") if ln.strip()]
        if len(lines) != len(chunk):
            continue
        for w, ln in zip(chunk, lines):
            ln = ln.replace("(en)", "").replace(f"({voice})", "")
            out[w] = matcher._normalize_ipa(ln)
    return out


def build_lexicon(lp: LangPair, side: str, n: int) -> dict[str, str]:
    """word -> IPA via wordfreq top-N + espeak. Cached per pair/side."""
    wf = lp.wf_src if side == "src" else lp.wf_tgt
    voice = lp.espeak_src if side == "src" else lp.espeak_tgt
    cache = os.path.join(workdir(lp), f"lex-{side}-{n}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    words = [w for w in top_n_list(wf, n) if w.isalpha() and len(w) > 1]
    lex = batch_g2p(words, voice)
    json.dump(lex, open(cache, "w"), ensure_ascii=False)
    return lex


def combo(qi: str, ci: str) -> float:
    return 0.5 * _ngram_channel(qi, ci) + 0.5 * _feat(qi, ci)


def _feat(a: str, b: str) -> float:
    return nw_sim_ipa(_canonical(a), _canonical(b))


def mine_dictionary(lp: LangPair, src_lex, tgt_lex, bar=0.62):
    """Block by shared phoneme bigram, rank by combo. Generic homophone mine."""
    def bigrams(ipa):
        s = ("#",) + _segs(_canonical(ipa)) + ("#",)
        return frozenset(s[i] + s[i + 1] for i in range(len(s) - 1))

    tgt_items = list(tgt_lex.items())
    tgt_bg = [bigrams(p) for _w, p in tgt_items]
    index = defaultdict(list)
    for j, bg in enumerate(tgt_bg):
        for b in bg:
            index[b].append(j)

    entries = []
    for sw, si in src_lex.items():
        qb = bigrams(si)
        if not qb:
            continue
        cnt = defaultdict(int)
        for b in qb:
            for j in index[b]:
                cnt[j] += 1
        cands = sorted(cnt, key=lambda j: -cnt[j])[:15]
        best = None
        for j in cands:
            tw, ti = tgt_items[j]
            if tw == sw:
                continue
            s = combo(si, ti)
            if s >= bar and (best is None or s > best[0]):
                best = (s, tw, ti)
        if best:
            entries.append({"src": sw, "tgt": best[1], "score": round(best[0], 3),
                            "src_ipa": si, "tgt_ipa": best[2]})
    return entries


def main():
    if len(sys.argv) < 3:
        print("usage: python multilang.py <src> <tgt> [sentence]")
        print("registered pairs:", ", ".join(PAIRS))
        return
    key = f"{sys.argv[1]}-{sys.argv[2]}"
    if key not in PAIRS:
        print(f"unregistered pair {key}; add to PAIRS. known: {list(PAIRS)}")
        return
    lp = PAIRS[key]
    wd = workdir(lp)
    print(f"=== {key} === {lp.notes}", flush=True)
    if lp.wf_tgt == "en" and lp.tgt != "en":
        print(f"  WARNING: wordfreq has no '{lp.tgt}' list; target lexicon "
              f"would be English. Supply tgt_wordlist before mining.", flush=True)

    src_lex = build_lexicon(lp, "src", 1500)
    tgt_lex = build_lexicon(lp, "tgt", 8000)
    print(f"  lexicons: {len(src_lex)} src, {len(tgt_lex)} tgt", flush=True)

    entries = mine_dictionary(lp, src_lex, tgt_lex)
    json.dump(entries, open(os.path.join(wd, "dictionary.json"), "w"),
              ensure_ascii=False, indent=0)
    s_tier = [e for e in entries if e["score"] >= 0.9]
    print(f"  mined {len(entries)} homophone pairs ({len(s_tier)} S-tier)",
          flush=True)
    print("  samples:", flush=True)
    for e in sorted(entries, key=lambda x: -x["score"])[:12]:
        print(f"    {e['score']:.2f}  {e['src']:12s} ~ {e['tgt']:14s}"
              f"  [{e['src_ipa']} | {e['tgt_ipa']}]", flush=True)


if __name__ == "__main__":
    main()
