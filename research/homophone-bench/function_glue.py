"""Targeted composition-only rescue for core function-word glue.

The strict function-word decoder intentionally rejects very short weak forms
when they do not clear the normal dictionary threshold. Composition still
needs a tiny, auditable glue layer for words such as "the". These rows remain
marked as composition_only and keep their low independent scores.
"""
from __future__ import annotations

import json
from functools import lru_cache

import matcher
from enrich import enrich

matcher._vecs = lru_cache(maxsize=None)(matcher._vecs.__wrapped__)
matcher._segs = lru_cache(maxsize=None)(matcher._segs.__wrapped__)

GLUE_ROWS = [
    {
        "en": "the",
        "fr": "de",
        "en_ipa": "ðə",
        "fr_ipa": "də",
        "funcword_band": "funcword_core",
    },
    {
        "en": "and",
        "fr": "end",
        "funcword_band": "funcword_core",
    },
]


def score_from_ipa(en_ipa: str, fr_ipa: str) -> float:
    return round(0.5 * matcher._ngram_channel(en_ipa, fr_ipa)
                 + 0.5 * matcher._feat_channel(en_ipa, fr_ipa), 3)


def mark_glue(entry: dict, row: dict) -> dict:
    e = dict(entry)
    e["funcword"] = True
    e["funcword_band"] = row["funcword_band"]
    e["composition_only"] = True
    e["source_stage"] = e.get("source_stage", "v5.1_funcword_glue")
    e["provenance"] = e.get("provenance", "explicit_weak_function_glue")
    e["accepted_by"] = (
        "composition-only core function glue; keeps independent score and "
        "does not upgrade gold tier"
    )
    return e


def main() -> None:
    entries = json.load(open("dictionary-v5.json", encoding="utf-8"))
    by_pair = {(e.get("en"), e.get("fr")): i for i, e in enumerate(entries)}
    added = 0
    updated = 0

    for row in GLUE_ROWS:
        pair = (row["en"], row["fr"])
        if pair in by_pair:
            entries[by_pair[pair]] = mark_glue(entries[by_pair[pair]], row)
            updated += 1
            continue

        score = score_from_ipa(row["en_ipa"], row["fr_ipa"])
        entry = {
            "en": row["en"],
            "fr": row["fr"],
            "score": score,
            "tier": "B",
            "rank": 0,
            "multiword": False,
            "decoder": False,
            "cognate": False,
            "loanword": False,
            "direction": "en_fr",
            "en_ipa": row["en_ipa"],
            "fr_ipa": row["fr_ipa"],
            "en_freq_rank": None,
        }
        entries.append(enrich(mark_glue(entry, row)))
        added += 1

    with open("dictionary-v5.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=0)
    print(f"function glue added={added} updated={updated}")


if __name__ == "__main__":
    main()
