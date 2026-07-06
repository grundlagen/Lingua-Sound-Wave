#!/usr/bin/env python3
"""Add function words from zipf-glue.tsv to Stage 1, rebuild with general word frequencies."""
import json, os

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")
print("STAGE 1+: ADDING FUNCTION WORDS + GENERAL FREQUENCIES")
print("=" * 55)

# Load existing Stage 1
pairs = {}
for line in open("stage1_homophones.jsonl",encoding="utf-8"):
    r = json.loads(line)
    pairs[(r["en"], r["fr"])] = r

# Add function words from zipf-glue.tsv (Van Rooten style)
added = 0
for i,line in enumerate(open("zipf-glue.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=4:
        en, fr, sound = p[0], p[1], float(p[2])
        if (en,fr) not in pairs:
            pairs[(en,fr)] = {"en":en,"fr":fr,"sound":sound,"meaning":0.5,"source":"zipf-glue","tier":"F"}
            added += 1

# Add high-frequency function words with their common homophones
FUNC = {
    "the":"le","a":"un","an":"un","and":"et","or":"ou","but":"mais",
    "in":"en","on":"sur","at":"à","to":"à","of":"de","for":"pour",
    "is":"est","are":"sont","was":"était","were":"étaient","be":"être",
    "he":"il","she":"elle","it":"il","we":"nous","they":"ils",
    "my":"mon","your":"votre","his":"son","her":"sa","its":"son",
    "this":"ce","that":"ce","these":"ces","those":"ces",
    "not":"pas","no":"non","yes":"oui","do":"faire","does":"fait",
    "have":"avoir","has":"a","had":"avait","can":"peut","will":"va",
}
for en,fr in FUNC.items():
    if (en,fr) not in pairs:
        pairs[(en,fr)] = {"en":en,"fr":fr,"sound":0.7,"meaning":0.8,"source":"van-rooten","tier":"F"}

print(f"  Added: {added} zipf-glue + {sum(1 for (e,f) in pairs if pairs[(e,f)]['source']=='van-rooten')} Van Rooten function words")
print(f"  Total: {len(pairs)} pairs")

# Save
with open("stage1_homophones.jsonl","w") as f:
    for (en,fr), d in sorted(pairs.items()):
        f.write(json.dumps(d,ensure_ascii=False)+"\n")

# Build general frequency list from bigram LM
import bigram_lm as blm
lm_en = blm.load("en")
lm_fr = blm.load("fr")

# Most common WORDS (not bigrams) from the LM unigram counts
en_common = [(w,lm_en.uni[w]) for w,_ in lm_en.uni.most_common(5000)]
fr_common = [(w,lm_fr.uni[w]) for w,_ in lm_fr.uni.most_common(5000)]

# Also build general bigram frequency
en_top_bigrams = [(f"{a} {b}",c) for (a,b),c in lm_en.bi.most_common(10000)]
fr_top_bigrams = [(f"{a} {b}",c) for (a,b),c in lm_fr.bi.most_common(10000)]

# Save general frequency data
with open("stage3_general_freq.json","w") as f:
    json.dump({
        "en_common_words": en_common[:1000],
        "fr_common_words": fr_common[:1000],
        "en_top_bigrams": en_top_bigrams[:2000],
        "fr_top_bigrams": fr_top_bigrams[:2000],
    }, f, ensure_ascii=False)

print(f"  EN common words: {len(en_common)}, FR: {len(fr_common)}")
print(f"  EN top bigrams: {len(en_top_bigrams)}, FR: {len(fr_top_bigrams)}")
print(f"  Saved: stage3_general_freq.json")

# Sample
for lang, words in [("EN",en_common[:10]),("FR",fr_common[:10])]:
    print(f"  {lang} most common: {', '.join(w for w,_ in words)}")
