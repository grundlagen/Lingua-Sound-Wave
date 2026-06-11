"""Rescue pass for English function words.

The composition smoke test showed function words (the/is/of/and...) are the
coverage holes: they are 1-3 segments long, so the multiword decoder never
fires (needs >= 2 words x 2 segments), and the pairwise pipeline's frequency
slice missed some short French matches (is ~ hisse). But lines are mostly
function words, so these matter more than anything for composition.

Fix: decode each function word as a SINGLE word over the full-Lexique trie
(max_words=1), keep matches >= 0.80 with no expensive deletions, merge into
dictionary-v5.json with provenance funcword, then re-run finalize.py.
"""
from __future__ import annotations

import json

import phonetic_decoder as pd
from lexicon_g2p import load_en, load_fr

FUNCTION_WORDS = [
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "and",
    "or", "but", "if", "in", "on", "at", "by", "to", "for", "with", "from",
    "as", "so", "no", "not", "all", "any", "can", "will", "would", "could",
    "should", "do", "did", "done", "has", "had", "have", "this", "that",
    "these", "those", "there", "here", "when", "then", "than", "what", "who",
    "how", "why", "more", "most", "some", "such", "very", "just", "now",
    "out", "up", "down", "over", "under", "into", "about",
]


def main():
    lex_en = load_en()
    lex_fr = load_fr()
    root = pd.build_trie(min_zipf=1.5)
    entries = json.load(open("dictionary-v5.json"))
    have = {(e["en"], e["fr"]) for e in entries}

    added = []
    for w in FUNCTION_WORDS:
        prons = lex_en.get(w)
        if not prons:
            continue
        best_existing = max((e["score"] for e in entries if e["en"] == w
                             and e.get("usable_for_composition")), default=0.0)
        if best_existing >= 0.95:
            continue
        for c in pd.decode(prons[0], root, top_n=6, max_words=1):
            if (c["similarity"] >= 0.80 and c["expensive_deletions"] == 0
                    and c["max_substitution"] <= 0.30 and (w, c["fr"]) not in have):
                tier = "S" if c["similarity"] >= 0.90 else "A"
                fr_ipa = (lex_fr.get(c["fr"]) or [""])[0]
                added.append({
                    "en": w, "fr": c["fr"], "score": c["similarity"],
                    "tier": tier, "rank": 0, "multiword": False,
                    "funcword": True, "decoder": True, "cognate": False,
                    "loanword": False, "coverage": c["coverage"],
                    "max_substitution": c["max_substitution"],
                    "en_ipa": prons[0], "fr_ipa": fr_ipa,
                    "direction": "en_fr", "en_freq_rank": None,
                })
                have.add((w, c["fr"]))
                if sum(1 for a in added if a["en"] == w) >= 2:
                    break

    # enrich the new entries with alignment/junction fields, then merge
    from enrich import enrich
    added = [enrich(a) for a in added]
    entries += added
    with open("dictionary-v5.json", "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=0)
    for a in sorted(added, key=lambda x: -x["score"]):
        print(f"  {a['tier']} {a['score']:.2f} {a['en']:8s} ~ {a['fr']:12s} [{a['fr_ipa']}]")
    print(f"\nmerged {len(added)} function-word entries; now re-run finalize.py")


if __name__ == "__main__":
    main()
