"""v7, efficiently: v5 IS the retrieval leg (richer than a re-run -- multiword,
pair-bank, overrides), so DON'T redo retrieval. Run ONLY the single-phoneme
decoder and keep what it ADDS to v5.

  additions = decoder entries (single-phoneme poetry trie) that are
              (a) absent from v5, or
              (b) S-tier where v5's best for that English word is < A.
  v7 = v5  ∪  additions     (no retrieval recomputation)

Run: python build_v7_decode.py [--limit N]
Outputs: v7-decode-additions.json/tsv, dictionary-v7-integrated.json
"""
from __future__ import annotations

import json
import sys
import time

from wordfreq import top_n_list

import matcher
from matcher import g2p
import phonetic_decoder as pd
import poetry_mode as pm

try:
    import bigram_lm
    LM = bigram_lm.load("fr")
except Exception:
    LM = None

S, A, B = 0.90, 0.78, 0.62


def coherence(fr):
    toks = [t.lower() for t in fr.replace("'", " ").split() if t]
    return LM.fluency(toks) if (LM and toks) else 0.0


def decode_best(w, root):
    try:
        qi = g2p(w, "en")
    except Exception:
        return None
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
                    c["words"] > 1, bool(strict))
    return best


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    pd.MIN_WORD_SEGS = 1; pd.WORD_PENALTY = 0.04; pd.BEAM = 300

    v5 = json.load(open("dictionary-v5.json"))
    v5_pairs = {(e["en"], e["fr"]) for e in v5 if e.get("direction", "en_fr") == "en_fr"}
    v5_best = {}
    for e in v5:
        if e.get("direction", "en_fr") != "en_fr":
            continue
        v5_best[e["en"]] = max(v5_best.get(e["en"], 0.0), float(e.get("score", 0)))
    v5_cog = {e["en"] for e in v5 if e.get("cognate")}

    root = pm.build_poetry_trie(min_zipf=2.0)
    # word set: all v5 headwords + frequent EN words (so v7 >= v5 coverage)
    words = list(dict.fromkeys(list(v5_best) + [
        w for w in top_n_list("en", 9000) if w.isalpha() and len(w) >= 2]))
    if limit:
        words = words[:limit]

    adds, t0 = [], time.time()
    for i, w in enumerate(words):
        d = decode_best(w, root)
        if not d:
            continue
        s, fr, uf, mw, strict = d
        if s < B:
            continue
        novel_pair = (w, fr) not in v5_pairs
        improves = s >= S and v5_best.get(w, 0.0) < A
        if novel_pair or improves:
            adds.append({"en": w, "fr": fr, "score": round(float(s), 3),
                         "tier": "S" if s >= S else "A" if s >= A else "B",
                         "via": "decode", "uses_filler": bool(uf),
                         "multiword": bool(mw), "gold": bool(strict and s >= S),
                         "cognate": w in v5_cog, "coherence": round(coherence(fr), 3),
                         "novel_vs_v5": novel_pair,
                         "direction": "en_fr", "usable_for_composition": True,
                         "loanword": False, "source_stage": "v7_decode_addition"})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(words)}  adds {len(adds)}  {time.time()-t0:.0f}s",
                  file=sys.stderr)

    json.dump(adds, open("v7-decode-additions.json", "w"), ensure_ascii=False, indent=0)
    with open("v7-decode-additions.tsv", "w", encoding="utf-8") as f:
        f.write("tier\tscore\tgold\tfiller\tnovel\ten\tfr\n")
        for r in sorted(adds, key=lambda r: -r["score"]):
            f.write(f"{r['tier']}\t{r['score']}\t{int(r['gold'])}\t{int(r['uses_filler'])}"
                    f"\t{int(r['novel_vs_v5'])}\t{r['en']}\t{r['fr']}\n")
    merged = list(v5) + adds
    json.dump(merged, open("dictionary-v7-integrated.json", "w"),
              ensure_ascii=False, indent=0)
    from collections import Counter
    print(f"\nv7 = v5 ({len(v5)}) + {len(adds)} decoder additions -> {len(merged)}")
    print(f"  additions tiers {dict(Counter(r['tier'] for r in adds))}; "
          f"gold {sum(1 for r in adds if r['gold'])}; "
          f"filler {sum(1 for r in adds if r['uses_filler'])}")
    print(f"  [{time.time()-t0:.0f}s] wrote v7-decode-additions.*, dictionary-v7-integrated.json")


if __name__ == "__main__":
    main()
