"""Finalize v5 into the composition-ready release the review specified:

  1. junction columns surfaced in the TSV (they existed in JSON only);
  2. gap/deletion ratios per row — split into cheap (linguistically
     licensed: offglide/schwa/h) vs expensive gaps, because a flat gap
     ratio would wrongly punish the dough/dos class whose deletions are
     all licensed;
  3. syllable_delta and a huge_deletion flag (the variation->verne
     collapse class: 3+ syllables crushed to <=1, or delta >= 3);
  4. B split into B_safe / B_reservoir;
  5. usable_for_composition implementing the agreed acceptance rule;
  6. composition-index.json: row indexes by pivot, first/final pivot
     class, syllable count, direction, tier.

Run: python finalize.py
Outputs: dictionary-v5.tsv (extended), composition-index.json,
         updates dictionary-v5.json in place with the new fields.
"""
from __future__ import annotations

import json
from collections import defaultdict

CHEAP_GAP_SEGS = {"ʊ", "ɪ", "j", "w", "ə", "ɚ", "h"}


def derive(e: dict) -> dict:
    align = e.get("align") or []
    n = len(align)
    gaps = [(x, y, c) for x, y, c in align if x == "·" or y == "·"]
    cheap = sum(1 for x, y, c in gaps
                if (x if x != "·" else y).replace("ː", "") in CHEAP_GAP_SEGS)
    e["gap_ratio"] = round(len(gaps) / n, 3) if n else 1.0
    e["expensive_gap_ratio"] = round((len(gaps) - cheap) / n, 3) if n else 1.0
    es, fs = e.get("en_syll"), e.get("fr_syll")
    if es is not None and fs is not None:
        e["syllable_delta"] = abs(es - fs)
        e["huge_deletion"] = (es >= 3 and fs <= 1) or (fs >= 3 and es <= 1) \
            or abs(es - fs) >= 3
    else:
        e["syllable_delta"] = 99
        e["huge_deletion"] = True

    tier = e["tier"]
    if tier == "B":
        b_safe = (e["score"] >= 0.72 and e["syllable_delta"] <= 1
                  and e["gap_ratio"] <= 0.20 and not e["huge_deletion"])
        e["tier"] = "B_safe" if b_safe else "B_reservoir"

    t = e["tier"]
    e["usable_for_composition"] = (
        t == "S"
        or (t == "A" and e["syllable_delta"] <= 1 and e["gap_ratio"] <= 0.30)
        or t == "B_safe"
    )
    return e


def main():
    entries = json.load(open("dictionary-v5.json"))
    entries = [derive(dict(x)) for x in entries]

    with open("dictionary-v5.json", "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=0)

    cols = ["tier", "score", "direction", "en", "fr", "flags", "en_ipa", "fr_ipa",
            "pivot", "en_syll", "fr_syll", "syllable_delta", "gap_ratio",
            "expensive_gap_ratio", "huge_deletion", "usable_for_composition",
            "en_onset", "en_coda", "fr_onset", "fr_coda", "alignment"]
    with open("dictionary-v5.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for x in entries:
            flags = ",".join(k for k in
                             ["multiword", "cognate", "loanword", "pairbank", "decoder"]
                             if x.get(k))
            row = [x.get("tier", ""), x.get("score", ""),
                   x.get("direction", "en_fr"), x.get("en", ""), x.get("fr", ""),
                   flags, x.get("en_ipa", ""), x.get("fr_ipa", ""),
                   x.get("pivot", ""), x.get("en_syll", ""), x.get("fr_syll", ""),
                   x.get("syllable_delta", ""), x.get("gap_ratio", ""),
                   x.get("expensive_gap_ratio", ""),
                   int(bool(x.get("huge_deletion"))),
                   int(bool(x.get("usable_for_composition"))),
                   x.get("en_onset", ""), x.get("en_coda", ""),
                   x.get("fr_onset", ""), x.get("fr_coda", ""),
                   x.get("alignment", "")]
            f.write("\t".join(str(v) for v in row) + "\n")

    # ---- composition indexes (row index -> entries list position) ----
    idx = {"pivot": defaultdict(list), "first_class": defaultdict(list),
           "final_class": defaultdict(list), "en_syll": defaultdict(list),
           "direction": defaultdict(list), "tier": defaultdict(list)}
    for i, x in enumerate(entries):
        p = x.get("pivot", "")
        idx["pivot"][p].append(i)
        if p:
            idx["first_class"][p[0]].append(i)
            idx["final_class"][p[-1]].append(i)
        idx["en_syll"][str(x.get("en_syll", ""))].append(i)
        idx["direction"][x.get("direction", "en_fr")].append(i)
        idx["tier"][x["tier"]].append(i)
    with open("composition-index.json", "w") as f:
        json.dump({k: dict(v) for k, v in idx.items()}, f)

    from collections import Counter
    tc = Counter(x["tier"] for x in entries)
    uc = sum(1 for x in entries if x["usable_for_composition"])
    print(f"tiers: {dict(tc)}")
    print(f"usable_for_composition: {uc}/{len(entries)}")
    print("wrote dictionary-v5.tsv (extended), composition-index.json")


if __name__ == "__main__":
    main()
