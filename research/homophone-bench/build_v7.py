"""Dictionary v7 = best of v5 + v6.

v5's method is the better base: RETRIEVAL (block the whole French lexicon, rank by
the rule-rich combo), exhaustive, multiple candidates per word, less corner-
cutting. v6's value: the single-phoneme decoder (fine grain + fillers) for words
retrieval misses and for the line/filler carves retrieval cannot make.

v7 unions them ("the fix"):
  RETRIEVAL leg (v5)  -- block + rank, keep top-K per word (multiple, like v5);
                         ALLOPHONE-AWARE: blocking indexes each FR word's rule
                         variants (matcher._variants: schwa-drop, nasal-split,
                         rhotic ʁ~ɹ, diphthong-smooth), so allophonic realizations
                         are reachable, and combo ranks with the same variants.
  DECODER leg (v6)    -- single-phoneme poetry trie for words where retrieval is
                         weak (< A) and for filler/line carves.
  keep the better per (en,fr) by combo; v5 re-mining gates set the `gold` flag.

Single-phoneme is the decoder inventory; single-ALLOPHONE enters via the variant
expansion on both sides of the match (the accent-independent way -- allophones as
match-time rules, per REPRESENTATION.md, not stored data).

Run: python build_v7.py [--n-en N] [--n-fr M]
Outputs: dictionary-v7.json, dictionary-v7.tsv
"""
from __future__ import annotations

import json
import sys
import time

from wordfreq import top_n_list

import matcher
from matcher import g2p, _segs, _canonical, _ngram_channel, _feat_channel, _variants
import phonetic_decoder as pd
import poetry_mode as pm

try:
    import bigram_lm
    LM = bigram_lm.load("fr")
except Exception:
    LM = None

S, A, B = 0.90, 0.78, 0.62
SHORTLIST = 30
KEEP_PER_WORD = 2          # v5-style: multiple candidates, less cutting


def variant_bigrams(ipa: str) -> set:
    """Allophone-aware bigram set: union over the rule variants of the IPA."""
    out = set()
    for v in _variants(ipa):
        s = ("#",) + _segs(_canonical(v)) + ("#",)
        out |= {s[i] + s[i + 1] for i in range(len(s) - 1)}
    return out


def combo_ipa(qi: str, ci: str) -> float:
    return 0.5 * _ngram_channel(qi, ci) + 0.5 * _feat_channel(qi, ci)  # feat uses _variants


def tier(s):
    return "S" if s >= S else "A" if s >= A else "B" if s >= B else None


def coherence(fr):
    toks = [t.lower() for t in fr.replace("'", " ").split() if t]
    return LM.fluency(toks) if (LM and toks) else 0.0


def build_fr_index(n_fr):
    out = []
    for w in top_n_list("fr", n_fr):
        if not (w.isalpha() and len(w) > 1):
            continue
        ipa = g2p(w, "fr")
        out.append((w, ipa, variant_bigrams(ipa)))   # allophone-aware blocking key
    return out


def retrieve(w, qi, qb, fr_index):
    short = sorted(fr_index,
                   key=lambda c: (len(qb & c[2]) / (len(qb) + len(c[2])) if c[2] else 0),
                   reverse=True)[:SHORTLIST]
    scored = []
    for cw, ci, _ in short:
        if cw == w:
            continue
        scored.append((combo_ipa(qi, ci), cw))
    scored.sort(reverse=True)
    return scored[:KEEP_PER_WORD]


def decode_leg(w, qi, root):
    cands = pd.decode(qi, root, top_n=8, max_words=4,
                      lm=LM, lm_weight=pd.LM_BEAM_WEIGHT if LM else 0.0)
    best = None
    for c in cands:
        if c["coverage"] < 0.6:
            continue
        s = matcher.homophone_score(w, "en", c["fr"], "fr")["score"]
        strict = (c["expensive_deletions"] == 0 and c.get("max_substitution", 1) <= 0.45
                  and c["coverage"] >= 0.85)
        if best is None or s > best[0]:
            best = (s, c["fr"], any(x in pm.FILLER for x in c["fr"].split()),
                    c["words"] > 1, strict)
    return best


def main():
    n_en = int(sys.argv[sys.argv.index("--n-en") + 1]) if "--n-en" in sys.argv else 2000
    n_fr = int(sys.argv[sys.argv.index("--n-fr") + 1]) if "--n-fr" in sys.argv else 7000
    pd.MIN_WORD_SEGS = 1; pd.WORD_PENALTY = 0.04; pd.BEAM = 300

    print(f"phonemizing {n_fr} FR words (allophone-aware index)...", file=sys.stderr)
    fr_index = build_fr_index(n_fr)
    root = pm.build_poetry_trie(min_zipf=2.0)
    try:
        v5 = json.load(open("dictionary-v5.json"))
        known = {e["en"] for e in v5 if e.get("direction", "en_fr") == "en_fr"}
        v5_cog = {e["en"] for e in v5 if e.get("cognate")}
    except FileNotFoundError:
        known, v5_cog = set(), set()

    targets = [w for w in top_n_list("en", n_en + 2000)
               if w.isalpha() and len(w) >= 2][:n_en]
    rows, seen = [], set()
    t0 = time.time()
    n_ret = n_dec = 0
    for i, w in enumerate(targets):
        qi = g2p(w, "en")
        qb = variant_bigrams(qi)
        if not qb:
            continue
        # retrieval leg (v5), multiple candidates
        best_ret = 0.0
        for s, fr in retrieve(w, qi, qb, fr_index):
            t = tier(s)
            if not t or (w, fr) in seen:
                continue
            seen.add((w, fr)); n_ret += 1
            best_ret = max(best_ret, s)
            rows.append({"en": w, "fr": fr, "score": round(s, 3), "tier": t,
                         "via": "retrieve", "cognate": w in v5_cog,
                         "uses_filler": False, "multiword": " " in fr,
                         "coherence": round(coherence(fr), 3),
                         "gold": s >= S, "novel": w not in known})
        # decoder leg (v6): only when retrieval is weak, OR to add a filler/line carve
        if best_ret < A:
            d = decode_leg(w, qi, root)
            if d:
                s, fr, uf, mw, strict = d
                t = tier(s)
                if t and (w, fr) not in seen:
                    seen.add((w, fr)); n_dec += 1
                    rows.append({"en": w, "fr": fr, "score": round(s, 3), "tier": t,
                                 "via": "decode", "cognate": w in v5_cog,
                                 "uses_filler": uf, "multiword": mw,
                                 "coherence": round(coherence(fr), 3),
                                 "gold": strict and s >= S, "novel": w not in known})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(targets)}  rows {len(rows)} "
                  f"(ret {n_ret} dec {n_dec})  {time.time()-t0:.0f}s", file=sys.stderr)

    rows.sort(key=lambda r: -r["score"])

    def _nat(v):                                   # numpy -> native for JSON
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        return v
    rows = [{k: _nat(v) for k, v in r.items()} for r in rows]
    json.dump([{**r, "direction": "en_fr", "usable_for_composition": True,
                "loanword": False, "source_stage": "v7_retrieve_union_decode"}
               for r in rows],
              open("dictionary-v7.json", "w"), ensure_ascii=False, indent=0)
    with open("dictionary-v7.tsv", "w", encoding="utf-8") as f:
        f.write("tier\tscore\tvia\tgold\tfiller\tcognate\tnovel\ten\tfr\n")
        for r in rows:
            f.write(f"{r['tier']}\t{r['score']}\t{r['via']}\t{int(r['gold'])}"
                    f"\t{int(r['uses_filler'])}\t{int(r['cognate'])}\t{int(r['novel'])}"
                    f"\t{r['en']}\t{r['fr']}\n")
    from collections import Counter
    tc = Counter(r["tier"] for r in rows)
    print(f"\nv7: {len(rows)} entries from {len(targets)} EN words  "
          f"(retrieve {n_ret} + decode {n_dec})")
    print(f"  tiers {dict(tc)}; gold {sum(1 for r in rows if r['gold'])}; "
          f"filler {sum(1 for r in rows if r['uses_filler'])}; "
          f"decode-only wins {n_dec}")
    print(f"  [{time.time()-t0:.0f}s] wrote dictionary-v7.json, dictionary-v7.tsv")


if __name__ == "__main__":
    main()
