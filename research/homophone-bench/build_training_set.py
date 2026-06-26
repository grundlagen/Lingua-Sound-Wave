"""Consolidate every legitimately-generated homophonic artifact into ONE
instruction-style training set for the restructuring task (input English ->
output French that sounds like it). This is the dataset you'd fine-tune an LLM on;
it is built only from our own generated data + public-domain English (no
copyrighted homophonic text is used).

Sources folded in:
  dictionary-v7-integrated.json   word-level homophone pairs (carve atoms)
  loop-certified-pairs-v7u-aug    dual atoms: EN sounds-like AND means the FR
  phrase-bank-balanced.tsv        EN phrase -> fluent FR homophone
  corpus-carves.tsv               PD nursery-rhyme line -> FR homophone

Output: train-homophonic.jsonl  (task, prompt, completion, + metadata)
Run: python build_training_set.py
"""
from __future__ import annotations

import json

CARVE_INSTR = "Rewrite the English so it sounds the same when read aloud in French:"
ATOM_INSTR = "Give a French word that sounds like this English word and means the same:"


def main():
    rows = []

    # 1. word-level homophone pairs (usable_for_composition)
    try:
        d = json.load(open("dictionary-v7-integrated.json", encoding="utf-8"))
        for e in d:
            if e.get("usable_for_composition") and e.get("direction", "en_fr") == "en_fr" \
                    and float(e.get("score", 0)) >= 0.85:
                rows.append({"task": "word_carve", "prompt": f"{CARVE_INSTR} {e['en']}",
                             "completion": e["fr"], "combo": round(float(e["score"]), 3),
                             "source": "dict-v7"})
    except FileNotFoundError:
        pass

    # 2. dual atoms (sound AND meaning) -- the gold restructuring targets
    try:
        for i, line in enumerate(open("loop-certified-pairs-v7u-aug.tsv", encoding="utf-8")):
            if i == 0:
                continue
            en, fr, cert, *_ = line.rstrip("\n").split("\t")
            rows.append({"task": "dual_atom", "prompt": f"{ATOM_INSTR} {en}",
                         "completion": fr, "certifications": int(cert),
                         "source": "loop-certified"})
    except FileNotFoundError:
        pass

    # 3 + 4. phrase / line carves (with quality so a trainer can filter)
    for path, src in [("phrase-bank-balanced.tsv", "phrase-bank"),
                      ("corpus-carves.tsv", "pd-corpus")]:
        try:
            for i, line in enumerate(open(path, encoding="utf-8")):
                if i == 0:
                    continue
                en, fr, combo, cov, flu = line.rstrip("\n").split("\t")
                rows.append({"task": "phrase_carve", "prompt": f"{CARVE_INSTR} {en}",
                             "completion": fr, "combo": float(combo),
                             "coverage": float(cov), "fluency": float(flu),
                             "source": src})
        except FileNotFoundError:
            pass

    with open("train-homophonic.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_task = Counter(r["task"] for r in rows)
    by_src = Counter(r["source"] for r in rows)
    print(f"train-homophonic.jsonl: {len(rows)} examples")
    print("  by task:  ", dict(by_task))
    print("  by source:", dict(by_src))
    print("\n  samples:")
    seen = set()
    for r in rows:
        if r["task"] not in seen:
            seen.add(r["task"])
            print(f"    [{r['task']}] {r['prompt']!r} -> {r['completion']!r}")


if __name__ == "__main__":
    main()
