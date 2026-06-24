"""Build dictionary v6: v5's re-mining ethos + 1-segment filler words + the
coherence/ranking fixes established this session.

v5 ethos kept verbatim: real lexicon -> decode -> matcher-rank -> tier by
quality, with composition-ready fields. What v6 changes (all motivated by the
Humpty/Mother-Goose work):

  1. POETRY TRIE -- admit van Rooten's 1-segment filler words (un /œ̃/, et /e/,
     on /ɔ̃/, a, aux, y, eau, le, la, de...); content words still need >=2 segs
     (no confetti). These are "the extra words" v5 structurally lacked, the
     scaffolding homophonic verse is built from.
  2. ARBITER RANKING -- rank candidates by the COVERAGE-AWARE matcher combo, not
     the decoder's coverage-blind path-normalised similarity (the bug that rated
     a partial 'épidémie' above a fuller 'un petit' carve).
  3. COHERENCE -- score the FR side with the bigram LM and keep it per entry, so
     composition can prefer carves that read as French, not salad.
  4. RE-MINE -- words v5 could not carve because they needed a filler now get
     entries (the honest-negative lesson: growth from re-mining, not re-labelling).

Outputs (v5-compatible schema + new fields):
  dictionary-v6.tsv   re-mined entries
  v6-fillers.tsv      the 1-segment filler primitives (composition building blocks)

Run: python build_v6.py [--limit N]
"""
from __future__ import annotations

import json
import sys
import time

from wordfreq import top_n_list, zipf_frequency

import matcher
import phonetic_decoder as pd
import poetry_mode as pm
from lexicon_g2p import load_en

try:
    import bigram_lm
    LM = bigram_lm.load("fr")
except Exception:
    LM = None

S, A, B = 0.90, 0.78, 0.62
COVER_MIN = 0.60


def tier(combo):
    return "S" if combo >= S else "A" if combo >= A else "B" if combo >= B else None


def coherence(fr):
    toks = [t.lower() for t in fr.replace("'", " ").split() if t]
    return LM.fluency(toks) if (LM and toks) else 0.0


def v5_headwords():
    try:
        d = json.load(open("dictionary-v5.json"))
        return {e["en"] for e in d if e.get("direction", "en_fr") == "en_fr"}
    except FileNotFoundError:
        return set()


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 500
    pd.BEAM = 300
    pd.MIN_WORD_SEGS = 1          # allow the 1-seg fillers to emit...
    pd.WORD_PENALTY = 0.05        # ...and let short carves compete
    root = pm.build_poetry_trie(min_zipf=2.0)
    lex_en = load_en()
    known = v5_headwords()
    print(f"v6 mine: poetry trie + arbiter rank. v5 had {len(known)} headwords.",
          file=sys.stderr)

    # candidate EN words: frequent, real, decodable (v5's source discipline)
    targets = [w for w in top_n_list("en", 6000)
               if w.isalpha() and len(w) >= 3 and w in lex_en][:limit]

    rows, t0 = [], time.time()
    for i, w in enumerate(targets):
        ipa = (lex_en.get(w) or [None])[0]
        if not ipa:
            continue
        cands = pd.decode(matcher._canonical(ipa), root, top_n=10, max_words=5,
                          lm=LM, lm_weight=pd.LM_BEAM_WEIGHT if LM else 0.0)
        best = None
        for c in cands:
            if c["coverage"] < COVER_MIN:
                continue
            combo = matcher.homophone_score(w, "en", c["fr"], "fr")["score"]  # arbiter
            t = tier(combo)
            if not t:
                continue
            coh = coherence(c["fr"])
            uses_filler = any(x in pm.FILLER for x in c["fr"].split())
            cand = (combo, c["fr"], t, c["coverage"], coh, uses_filler, c["words"])
            if best is None or combo > best[0]:
                best = cand
        if best:
            combo, fr, t, cov, coh, uf, nw = best
            rows.append({"en": w, "fr": fr, "combo": round(combo, 3), "tier": t,
                         "coverage": round(cov, 2), "coherence": round(coh, 3),
                         "uses_filler": uf, "multiword": nw > 1,
                         "novel": w not in known})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(targets)}  kept {len(rows)}  "
                  f"{time.time()-t0:.0f}s", file=sys.stderr)

    rows.sort(key=lambda r: -r["combo"])
    with open("dictionary-v6.tsv", "w", encoding="utf-8") as f:
        f.write("tier\tcombo\tcoverage\tcoherence\tuses_filler\tnovel\ten\tfr\n")
        for r in rows:
            f.write(f"{r['tier']}\t{r['combo']}\t{r['coverage']}\t{r['coherence']}"
                    f"\t{int(r['uses_filler'])}\t{int(r['novel'])}\t{r['en']}\t{r['fr']}\n")

    # the filler primitives layer (composition building blocks)
    with open("v6-fillers.tsv", "w", encoding="utf-8") as f:
        f.write("filler\tzipf\n")
        for w in sorted(pm.FILLER):
            f.write(f"{w}\t{zipf_frequency(w, 'fr'):.2f}\n")

    n_filler = sum(1 for r in rows if r["uses_filler"])
    n_novel = sum(1 for r in rows if r["novel"])
    print(f"\nv6: {len(rows)} entries from {len(targets)} EN words "
          f"({100*len(rows)//max(1,len(targets))}%); "
          f"{n_filler} use a filler ({100*n_filler//max(1,len(rows))}%), "
          f"{n_novel} novel vs v5.")
    print("top filler-carry entries (NEW capability vs v5):")
    shown = 0
    for r in rows:
        if r["uses_filler"] and r["multiword"]:
            print(f"  [{r['tier']} {r['combo']:.2f} coh {r['coherence']:.2f}] "
                  f"{r['en']:12s} -> {r['fr']}")
            shown += 1
            if shown >= 15:
                break
    print(f"\nwrote dictionary-v6.tsv, v6-fillers.tsv  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
