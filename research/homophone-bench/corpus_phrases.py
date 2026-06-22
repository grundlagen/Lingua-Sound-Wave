"""Extract frequent bigram phrases from the trained bigram LMs, convert them
to canonical IPA, and save as seed blocks for fragment_weave.

Instead of seeding the weave from uniform random IPA blocks (which produces
word salad), we seed it from the IPA of real fluent bigrams found in the corpus.
This makes the generator start life near grammatical phrase structures.

Usage:
  python corpus_phrases.py               # build corpus-phrases-{en,fr}.tsv
  python corpus_phrases.py --top 400     # extract top-N bigrams per language
"""
from __future__ import annotations

import re
import subprocess
import sys
from functools import lru_cache

from wordfreq import zipf_frequency

import bigram_lm as B
from matcher import _canonical, _segs

MIN_ZIPF = 3.2        # both words in a bigram must be at least this common
MIN_BIGRAM = 3        # minimum corpus count for a bigram to be included
TOP_N = 300           # how many bigrams to keep per language


@lru_cache(maxsize=50000)
def _espeak_ipa(word: str, lang: str) -> str:
    lang_code = "en-us" if lang == "en" else "fr"
    try:
        r = subprocess.run(
            ["espeak-ng", "-q", "--ipa", f"-v{lang_code}", word],
            capture_output=True, text=True, timeout=5,
        )
        raw = r.stdout.strip().replace("\n", " ")
        # reject if espeak fell back to a different language (leaves "(xx)" tags)
        if re.search(r"\([a-z]{2,4}\)", raw):
            return ""
        # strip stress/tone marks espeak adds
        raw = re.sub(r"[ˈˌ‿ ]+", "", raw)
        return _canonical(raw)
    except Exception:
        return ""


def bigram_ipa(w1: str, w2: str, lang: str) -> str:
    """IPA for the concatenated bigram (no space — single phoneme stream)."""
    p1 = _espeak_ipa(w1, lang)
    p2 = _espeak_ipa(w2, lang)
    if not p1 or not p2:
        return ""
    return p1 + p2


def extract(lang: str, top_n: int = TOP_N) -> list[dict]:
    lm = B.load(lang)
    candidates = []
    for (a, b), count in lm.bi.items():
        if count < MIN_BIGRAM:
            continue
        if zipf_frequency(a, lang) < MIN_ZIPF or zipf_frequency(b, lang) < MIN_ZIPF:
            continue
        # score = bigram probability × geometric mean zipf
        prob = lm.cond(a, b)
        za, zb = zipf_frequency(a, lang), zipf_frequency(b, lang)
        score = prob * ((za * zb) ** 0.5) / 6.0
        candidates.append((score, count, a, b))
    candidates.sort(reverse=True)

    results = []
    seen_ipa = set()
    for score, count, a, b in candidates:
        if len(results) >= top_n:
            break
        ipa = bigram_ipa(a, b, lang)
        if not ipa or ipa in seen_ipa:
            continue
        segs = _segs(ipa)
        if not (3 <= len(segs) <= 14):   # usable IPA length for fragment_weave
            continue
        seen_ipa.add(ipa)
        results.append({
            "lang": lang, "w1": a, "w2": b, "phrase": f"{a} {b}",
            "count": count, "score": round(score, 4), "ipa": ipa,
            "segs": len(segs),
        })
    return results


def main():
    top_n = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else TOP_N
    for lang in ("en", "fr"):
        print(f"\nExtracting {lang} bigrams (top {top_n})...", flush=True)
        rows = extract(lang, top_n=top_n)
        out = f"corpus-phrases-{lang}.tsv"
        with open(out, "w", encoding="utf-8") as f:
            f.write("phrase\tipa\tsegs\tcount\tscore\n")
            for r in rows:
                f.write(f"{r['phrase']}\t{r['ipa']}\t{r['segs']}\t{r['count']}\t{r['score']}\n")
        print(f"  {len(rows)} phrases -> {out}")
        if rows:
            print(f"  top 5: {[r['phrase'] for r in rows[:5]]}")


if __name__ == "__main__":
    main()
