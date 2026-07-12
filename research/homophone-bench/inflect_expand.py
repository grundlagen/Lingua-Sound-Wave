"""Stage 3 deterministic + stage 4 filter: inflection expansion of gold pairs.

PIPELINE.md build order #2-#3. No GPU, no LLM. For every gold pair at the
trusted tiers, expand BOTH sides through full inflection tables — French from
Lexique383 (lemme column), English from UniMorph eng — Zipf-weighted, then
judge every new cross-product with the matcher combo (judge of record).
"If *rives* works, try *rive*, *rivait* first."

Sound is scored exactly as matcher.g2p does (espeak-ng, same normalization),
but batched: one espeak process per chunk of unique words, verified against
matcher.g2p on a sample each run.

Tiers kept (word scope, dual_mine convention):
  DUAL-S  combo >= 0.75      DUAL-A  combo >= 0.60
Cognate surface look-alikes (SequenceMatcher >= 0.75) are flagged, not kept
silently — the franglais leak is a measured artifact.

Inputs (fetch if absent):
  /tmp/lexique383.tsv   http://www.lexique.org/databases/Lexique383/Lexique383.tsv
  /tmp/unimorph-eng.tsv https://raw.githubusercontent.com/unimorph/eng/master/eng

Output: inflection-pairs.tsv (en, fr, combo, tier, cognate, src_en, src_fr,
        src_tier, zipf_en, zipf_fr), sorted by combo, deduped against the
        tier ladder. Survivors are stage-1 candidates (new provenance column).

Usage:
    python inflect_expand.py [--tiers DUAL-S] [--k 6] [--limit 0] [--out inflection-pairs.tsv]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import defaultdict
from difflib import SequenceMatcher

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matcher  # noqa: E402
from wordfreq import zipf_frequency  # noqa: E402

LEXIQUE383 = "/tmp/lexique383.tsv"
UNIMORPH = "/tmp/unimorph-eng.tsv"
WORD = re.compile(r"^[a-zàâäçéèêëîïôöùûüÿœæ-]+$")


def fetch_inputs():
    import urllib.request
    if not os.path.exists(LEXIQUE383):
        urllib.request.urlretrieve(
            "http://www.lexique.org/databases/Lexique383/Lexique383.tsv", LEXIQUE383)
    if not os.path.exists(UNIMORPH):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/unimorph/eng/master/eng", UNIMORPH)


def load_fr_morph():
    """ortho -> set(all orthos sharing a lemme+cgram)."""
    lemma_forms = defaultdict(set)
    form_lemmas = defaultdict(set)
    for i, line in enumerate(open(LEXIQUE383, encoding="utf-8", errors="ignore")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        ortho, lemme, cgram = p[0].lower(), p[2].lower(), p[3]
        if not WORD.match(ortho) or not lemme:
            continue
        key = (lemme, cgram)
        lemma_forms[key].add(ortho)
        form_lemmas[ortho].add(key)
    return lemma_forms, form_lemmas


def load_en_morph():
    lemma_forms = defaultdict(set)
    form_lemmas = defaultdict(set)
    for line in open(UNIMORPH, encoding="utf-8", errors="ignore"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        lemma, form = p[0].lower(), p[1].lower()
        if not WORD.match(lemma) or not WORD.match(form):
            continue
        pos = p[2].split(";", 1)[0]
        key = (lemma, pos)
        lemma_forms[key].add(form)
        lemma_forms[key].add(lemma)
        form_lemmas[form].add(key)
        form_lemmas[lemma].add(key)
    return lemma_forms, form_lemmas


def variants(word, lemma_forms, form_lemmas, lang, k):
    """Zipf-top k inflected siblings of word (word itself excluded)."""
    out = set()
    for key in form_lemmas.get(word, ()):
        out.update(lemma_forms[key])
    out.discard(word)
    ranked = sorted(out, key=lambda w: -zipf_frequency(w, lang))
    return ranked[:k]


def batch_g2p(words, lang, cache):
    """espeak-ng one process per chunk; identical normalization to matcher.g2p."""
    todo = [w for w in words if (lang, w) not in cache]
    voice = {"en": "en-us", "fr": "fr"}[lang]
    CH = 400
    for i in range(0, len(todo), CH):
        chunk = todo[i:i + CH]
        out = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", voice],
            input="\n".join(chunk), capture_output=True, text=True, check=True)
        lines = [l for l in out.stdout.split("\n") if l.strip()]
        if len(lines) != len(chunk):   # espeak hiccup: fall back word-by-word
            lines = [matcher.g2p(w, lang) for w in chunk]
            for w, ipa in zip(chunk, lines):
                cache[(lang, w)] = ipa
            continue
        for w, ipa in zip(chunk, lines):
            cache[(lang, w)] = matcher._normalize_ipa(ipa.strip())
    # sample agreement check with the judge of record
    import random
    for w in random.Random(0).sample(list(words), min(5, len(words))):
        assert cache[(lang, w)] == matcher.g2p(w, lang), \
            f"batch g2p diverges from matcher.g2p on {w!r}"


def combo_ipa(qi, ci):
    return 0.5 * matcher._ngram_channel(qi, ci) + 0.5 * matcher._feat_channel(qi, ci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="DUAL-S", help="csv of source tiers (prefix match for LOOP)")
    ap.add_argument("--k", type=int, default=6, help="zipf-top variants per side")
    ap.add_argument("--limit", type=int, default=0, help="cap gold pairs (0 = all)")
    ap.add_argument("--min-zipf", type=float, default=1.5, help="drop variants rarer than this")
    ap.add_argument("--out", default="inflection-pairs.tsv")
    args = ap.parse_args()
    tiers = set(args.tiers.split(","))

    fetch_inputs()
    print("loading morphology ...")
    fr_lf, fr_fl = load_fr_morph()
    en_lf, en_fl = load_en_morph()

    gold, known = [], set()
    for i, line in enumerate(open("tier-ladder.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        en, fr, tier = p[1].lower(), p[2].lower(), p[3]
        known.add((en, fr))
        if tier in tiers and WORD.match(en) and WORD.match(fr) and en != fr \
                and en in en_fl:   # EN side must be English (MUSE FR-noise guard)
            gold.append((en, fr, tier))
    if args.limit:
        gold = gold[:args.limit]
    print(f"{len(gold)} gold pairs at tiers {sorted(tiers)}; {len(known)} known pairs for dedupe")

    # build the candidate cross-products
    cands = []            # (en', fr', src_en, src_fr, src_tier)
    en_words, fr_words = set(), set()
    for en, fr, tier in gold:
        evs = [en] + [v for v in variants(en, en_lf, en_fl, "en", args.k)
                      if zipf_frequency(v, "en") >= args.min_zipf]
        fvs = [fr] + [v for v in variants(fr, fr_lf, fr_fl, "fr", args.k)
                      if zipf_frequency(v, "fr") >= args.min_zipf]
        for e in evs:
            for f in fvs:
                if (e, f) == (en, fr) or (e, f) in known or e == f:
                    continue
                cands.append((e, f, en, fr, tier))
                en_words.add(e)
                fr_words.add(f)
    print(f"{len(cands)} new cross-products to judge "
          f"({len(en_words)} EN / {len(fr_words)} FR unique words)")

    cache: dict = {}
    batch_g2p(en_words, "en", cache)
    batch_g2p(fr_words, "fr", cache)
    print("g2p cached; scoring ...")

    rows, seen = [], set()
    for n, (e, f, sen, sfr, stier) in enumerate(cands):
        if (e, f) in seen:
            continue
        seen.add((e, f))
        s = combo_ipa(cache[("en", e)], cache[("fr", f)])
        if s < 0.60:
            continue
        tier = "DUAL-S" if s >= 0.75 else "DUAL-A"
        cog = int(SequenceMatcher(None, e, f).ratio() >= 0.75)
        rows.append((e, f, round(s, 3), tier, cog, sen, sfr, stier,
                     round(zipf_frequency(e, "en"), 2), round(zipf_frequency(f, "fr"), 2)))
        if (n + 1) % 20000 == 0:
            print(f"  scored {n + 1}/{len(cands)}, kept {len(rows)}")

    rows.sort(key=lambda r: -r[2])
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("en\tfr\tcombo\ttier\tcognate\tsrc_en\tsrc_fr\tsrc_tier\tzipf_en\tzipf_fr\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    ds = sum(1 for r in rows if r[3] == "DUAL-S")
    noncog = sum(1 for r in rows if r[3] == "DUAL-S" and not r[4])
    print(f"\nkept {len(rows)} new pairs -> {args.out}")
    print(f"  DUAL-S: {ds} ({noncog} non-cognate)   DUAL-A: {len(rows) - ds}")


if __name__ == "__main__":
    main()
