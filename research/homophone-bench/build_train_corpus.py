"""Build train-dual-v1.jsonl -- the TRAINING corpus for dual (literal ∧ sound)
translation, from everything this project has verified.

The user's point is right: the lexicon is now training-grade and retrieval-only
use is weak. This emits one instruction-tuning file, quality-tagged, both
directions, multiple task forms:

  dual_word        EN word -> FR that sounds like it AND relates in meaning
                   (tier-ladder 118k, scores attached)
  dual_word_rev    FR -> EN (dual-pairs-fr2en)
  glue             function-word sound-mappings (zipf-glue)
  bridge           Haiku-verified cross-scope renderings (llm-bridge)
  phrase_unit      EN phrase -> FR carve unit (phrase-bank)
  line_carve       EN line -> FR line (corpus-carves + the verified verse)
  rate_pair        (EN, FR) -> predict sound score  (teaches the judge INTO
                   the model -- reward-model data)

Every row: {task, prompt, completion, sound, meaning, tier, source}.

Run: python build_train_corpus.py
"""
from __future__ import annotations

import json
import os
import random


OUT = "train-dual-v1.jsonl"
rows = []


def add(task, prompt, completion, sound="", meaning="", tier="", source=""):
    rows.append({"task": task, "prompt": prompt, "completion": completion,
                 "sound": sound, "meaning": meaning, "tier": tier,
                 "source": source})


P_DUAL = ("Give a French rendering of the English word '{en}' that SOUNDS like "
          "it when read aloud in French and echoes its meaning.")
P_REV = ("Give an English rendering of the French word '{fr}' that SOUNDS like "
         "it when read aloud in English and echoes its meaning.")
P_LINE = ("Rewrite this English line as French that sounds the same when read "
          "aloud and stays coherent French: {en}")
P_RATE = ("Rate 0-100 how much the English '{en}' sounds like the French "
          "'{fr}' when both are read aloud.")


def main():
    # 1. tier-ladder words (the spine)
    n = 0
    for i, line in enumerate(open("tier-ladder.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 12 and int(p[0]) <= 8 and p[1].isalpha():
            snd = p[10] or ""
            add("dual_word", P_DUAL.format(en=p[1]), p[2],
                sound=snd, meaning=p[11], tier=p[3], source="tier-ladder")
            n += 1
    print(f"dual_word: {n}")

    # 1b. inflection-expansion survivors (inflect_expand.py, judge-verified)
    n = 0
    try:
        for i, line in enumerate(open("inflection-pairs.tsv", encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 5 and p[3] == "DUAL-S" and p[4] == "0":
                add("dual_word", P_DUAL.format(en=p[0]), p[1],
                    sound=p[2], meaning="", tier=p[3], source="inflect-expand")
                n += 1
    except FileNotFoundError:
        pass
    print(f"dual_word (inflection survivors): {n}")

    # 2. reverse direction
    n = 0
    for i, line in enumerate(open("dual-pairs-fr2en.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6 and p[5] in ("DUAL-S", "DUAL-A"):
            add("dual_word_rev", P_REV.format(fr=p[0]), p[1],
                sound=p[2], meaning=p[3], tier=p[5], source="dual-fr2en")
            n += 1
    print(f"dual_word_rev: {n}")

    # 3. glue + bridges
    for path, task, src in (("zipf-glue.tsv", "glue", "zipf-glue"),
                            ("llm-bridge.tsv", "bridge", "llm-bridge")):
        n = 0
        for i, line in enumerate(open(path, encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                add(task, P_DUAL.format(en=p[0]), p[1], sound=p[2],
                    meaning=p[3] if len(p) > 3 else "", source=src)
                n += 1
        print(f"{task}: {n}")

    # 4. phrase units + line carves
    n = 0
    for i, line in enumerate(open("phrase-bank-balanced.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and float(p[2]) >= 0.55:
            add("phrase_unit", P_LINE.format(en=p[0]), p[1], sound=p[2],
                source="phrase-bank")
            n += 1
    print(f"phrase_unit: {n}")
    n = 0
    for i, line in enumerate(open("corpus-carves.tsv", encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and float(p[2]) >= 0.45:
            add("line_carve", P_LINE.format(en=p[0]), p[1], sound=p[2],
                source="corpus-carves")
            n += 1
    # the verified verse gallery + paragraph (small, gold)
    GALLERY = [
        ("less debt, less mess", "laisse dette, laisse messe", "1.00"),
        ("do tell me, who said less?", "doux, tel mie, où cède laisse ?", "0.95"),
        ("we knew the sea", "oui, nous, ceci", "0.87"),
        ("my movie, my mess", "mes mous vies, mes messes", "0.86"),
        ("bless the chef, bless the soup", "blesse le chef, blesse la soupe", "0.85"),
        ("sell the soup, seize the seat", "selle la soupe, sise le site", "0.84"),
        ("moo, said the moose; boo, said the fool",
         "mou, cède la mousse ; boue, cède la foule", "0.83"),
        ("less and less, the bell said: dawn",
         "laisse en laisse, label cède, donne", "0.81"),
        ("humpty dumpty sat on a wall; humpty dumpty had a great fall",
         "un petit, un petit, cette âme au vol ; un petit, un petit, a des regrets, folle", "0.57"),
        ("less debt, less mess, more soup",
         "laisses de tête, d'messe, maures soupe", "0.86"),
    ]
    for en, fr, s in GALLERY:
        add("line_carve", P_LINE.format(en=en), fr, sound=s, tier="VERIFIED",
            source="final-verse")
        n += 1
    print(f"line_carve: {n}")

    # 5. rate_pair (reward-model data): positives from ladder + shuffled negatives
    rng = random.Random(0)
    lads = [r for r in rows if r["task"] == "dual_word" and r["sound"]]
    sample = rng.sample(lads, min(20000, len(lads)))
    frs = [r["completion"] for r in sample]
    n = 0
    for r in sample:
        en = r["prompt"].split("'")[1]
        sc = int(float(r["sound"]) * 100)
        add("rate_pair", P_RATE.format(en=en, fr=r["completion"]), str(sc),
            sound=r["sound"], source="ladder-pos")
        neg = rng.choice(frs)
        if neg != r["completion"]:
            add("rate_pair", P_RATE.format(en=en, fr=neg), "15",
                source="ladder-neg")
        n += 2
    print(f"rate_pair: {n}")

    rng.shuffle(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{OUT}: {len(rows)} instruction rows "
          f"({os.path.getsize(OUT)//1024//1024} MB)")
    print("Enough for LoRA SFT of a 1-8B model (needs ~10k-100k rows; we have "
          "more) + a reward head (rate_pair). Recipe: selflearn/train_selflearn.py "
          "--data train-dual-v1.jsonl on any single GPU.")


if __name__ == "__main__":
    main()
