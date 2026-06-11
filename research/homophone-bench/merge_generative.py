"""Merge vetted fragment-generated matches into dictionary-v5.

The fragment probe writes candidates with a generator score. This pass turns
them into audited dictionary rows by filling French IPA, recomputing an
independent v5 score, enriching alignments/junctions, and preserving provenance
instead of treating the rows as hand-built gold.

Run before finalize.py:
    python merge_generative.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from functools import lru_cache

import matcher
from enrich import enrich
from lexicon_g2p import load_fr

matcher._vecs = lru_cache(maxsize=None)(matcher._vecs.__wrapped__)
matcher._segs = lru_cache(maxsize=None)(matcher._segs.__wrapped__)

S, A, B = 0.90, 0.78, 0.62


def phrase_ipa(phrase: str, lex_fr: dict[str, list[str]]) -> str:
    prons: list[str] = []
    for raw in phrase.split():
        word = raw.strip(".,!?;:()[]{}\"'").lower()
        if not word:
            continue
        if word in lex_fr:
            prons.append(lex_fr[word][0])
        else:
            prons.append(matcher.g2p(word, "fr"))
    return " ".join(prons)


def independent_score(en_ipa: str, fr_ipa: str) -> float:
    fr_flat = fr_ipa.replace(" ", "")
    ngram = matcher._ngram_channel(en_ipa, fr_flat)
    feat = matcher._feat_channel(en_ipa, fr_flat)
    return round(0.5 * ngram + 0.5 * feat, 3)


def tier_for(score: float) -> str:
    if score >= S:
        return "S"
    if score >= A:
        return "A"
    if score >= B:
        return "B"
    return "B"


def main() -> None:
    entries = json.load(open("dictionary-v5.json", encoding="utf-8"))
    rows = list(csv.DictReader(open("generative-matches.tsv", encoding="utf-8"), delimiter="\t"))
    lex_fr = load_fr()
    have = {(e.get("en"), e.get("fr")) for e in entries}

    added: list[dict] = []
    skipped_existing = 0
    for row in rows:
        en = row["en"].strip().lower()
        fr = row["fr"].strip().lower()
        if (en, fr) in have:
            skipped_existing += 1
            continue
        generator_score = round(float(row["score"]), 3)
        fr_ipa = phrase_ipa(fr, lex_fr)
        if not fr_ipa:
            continue
        score = independent_score(row["en_ipa"], fr_ipa)
        entry = {
            "en": en,
            "fr": fr,
            "score": score,
            "generator_score": generator_score,
            "independent_score": score,
            "tier": tier_for(score),
            "rank": 0,
            "multiword": " " in fr,
            "generated": True,
            "decoder": True,
            "fragment": True,
            "cognate": False,
            "loanword": False,
            "direction": "en_fr",
            "coverage": None,
            "max_substitution": None,
            "en_ipa": row["en_ipa"],
            "fr_ipa": fr_ipa,
            "en_freq_rank": None,
            "chunk_recipe": row["chunk_recipe"],
            "source_stage": "v5.2_generated_validated",
            "provenance": "fragment_chain_decoder_arbiter",
            "accepted_by": (
                "fragments.py decode gates; independent combo score recomputed "
                "from en_ipa/fr_ipa during merge"
            ),
        }
        added.append(enrich(entry))
        have.add((en, fr))

    entries.extend(added)
    entries.sort(key=lambda e: (-float(e.get("score", 0.0)), e.get("en", ""), e.get("fr", "")))
    with open("dictionary-v5.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=0)

    counts = Counter(e["tier"] for e in added)
    usable_floor = sum(1 for e in added if e["tier"] in {"S", "A"})
    print(f"merged generated entries: {len(added)}")
    print(f"skipped existing pairs: {skipped_existing}")
    print(f"generated tier pre-finalize: {dict(counts)}")
    print(f"generated S/A before final gates: {usable_floor}/{len(added)}")


if __name__ == "__main__":
    main()
