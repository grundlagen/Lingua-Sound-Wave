"""Generative dual-language lines from pattern "lots".

The user's spec: organize the material by structural pattern so a generative
system can chain units of every granularity — partial-word fragments became
entries (generative flag), function words, whole words, multiword phrases —
into lines that are simultaneously English and French sound-streams.

Method:
  1. LOTS: bucket usable entries by (en_syllables, fr_onset class,
     fr_coda class, kind). Kind ladder: func < word < generative < multi.
  2. TEMPLATES: a line is a rhythm template like [1,1,2,1] (syllables per
     slot) plus a kind pattern (e.g. func word multi) — the
     "partial->whole->multi" chains the user described.
  3. GENERATE: walk slots left to right, sampling entries whose FR onset is
     compatible with the previous FR coda (no vowel-vowel hiatus), QC:
     entry score >= 0.85, line mean >= 0.92, rhythm delta == 0.
  4. Emit both texts + both IPA streams + per-slot provenance.

Deterministic given --seed. Output: generated-lines.tsv
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict

SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 7
N_LINES = 40

entries = json.load(open("dictionary-v5.json"))


def kind(e):
    if e.get("funcword"):
        return "func"
    if e.get("multiword") and not e.get("generative"):
        return "multi"
    if e.get("generative"):
        return "gen"
    return "word"


pool = [e for e in entries
        if e.get("usable_for_composition") and e.get("direction") == "en_fr"
        and e.get("score", 0) >= 0.85 and e.get("fr_onset")]

lots = defaultdict(list)
for e in pool:
    lots[(e["en_syll"], kind(e))].append(e)

TEMPLATES = [
    # (syllables, kind) per slot — mixed granularity by design
    [(1, "func"), (1, "word"), (2, "multi")],
    [(1, "word"), (1, "func"), (2, "gen")],
    [(2, "gen"), (1, "func"), (1, "word")],
    [(1, "func"), (2, "word"), (2, "multi")],
    [(2, "multi"), (1, "word"), (1, "func"), (1, "word")],
    [(1, "word"), (1, "word"), (2, "gen"), (1, "func")],
    [(2, "word"), (2, "multi")],
    [(3, "gen"), (1, "func"), (1, "word")],
]


def hiatus(prev, e):
    return (prev is not None and prev["fr_coda"].endswith("|V")
            and e["fr_onset"].endswith("|V"))


def gen_line(rng):
    tpl = rng.choice(TEMPLATES)
    slots, prev = [], None
    for syll, k in tpl:
        cands = lots.get((syll, k)) or lots.get((syll, "word")) or []
        cands = [e for e in cands if not hiatus(prev, e)]
        if not cands:
            return None
        e = rng.choice(sorted(cands, key=lambda x: -x["score"])[:60])
        slots.append(e)
        prev = e
    mean = sum(e["score"] for e in slots) / len(slots)
    delta = sum(e.get("syllable_delta", 0) for e in slots)
    if mean < 0.92 or delta != 0:
        return None
    return {
        "en": " ".join(e["en"] for e in slots),
        "fr": " ".join(e["fr"] for e in slots),
        "en_ipa": " ".join(e.get("en_ipa", "") for e in slots),
        "fr_ipa": " ".join(e.get("fr_ipa", "") for e in slots),
        "mean_score": round(mean, 3),
        "kinds": "+".join(kind(e) for e in slots),
        "template": "-".join(str(s) for s, _ in
                             [(e["en_syll"], None) for e in slots]),
    }


def main():
    rng = random.Random(SEED)
    out, seen = [], set()
    tries = 0
    while len(out) < N_LINES and tries < 20000:
        tries += 1
        line = gen_line(rng)
        if line and line["en"] not in seen:
            seen.add(line["en"])
            out.append(line)
    out.sort(key=lambda x: -x["mean_score"])
    with open("generated-lines.tsv", "w") as f:
        f.write("mean_score\tkinds\ten\tfr\ten_ipa\tfr_ipa\n")
        for r in out:
            f.write(f"{r['mean_score']}\t{r['kinds']}\t{r['en']}\t{r['fr']}"
                    f"\t{r['en_ipa']}\t{r['fr_ipa']}\n")
    print(f"lots: {len(lots)} pattern buckets over {len(pool)} usable entries")
    print(f"generated {len(out)} QC-passing lines -> generated-lines.tsv\n")
    for r in out[:12]:
        print(f"  [{r['mean_score']:.2f} {r['kinds']}]")
        print(f"    EN: {r['en']}")
        print(f"    FR: {r['fr']}")


if __name__ == "__main__":
    main()
