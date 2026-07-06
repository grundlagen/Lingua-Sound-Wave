#!/usr/bin/env python3
"""STAGE 3+4: Extract common phrases from both corpora, find homophone equivalents in the other language."""
import json, os, pickle
from collections import defaultdict, Counter

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")
print("STAGE 3+4: CROSS-LANGUAGE PHRASE MATCHING")
print("=" * 55)

# Load Stage 1 mapping both ways
en_to_fr = {}; fr_to_en = defaultdict(set)
for line in open("stage1_homophones.jsonl",encoding="utf-8"):
    r = json.loads(line)
    en_to_fr[r["en"]] = r["fr"]
    fr_to_en[r["fr"].split()[0]].add(r["en"])

# Load bigram LMs
import bigram_lm as blm
lm_en = blm.load("en"); lm_fr = blm.load("fr")
print(f"  EN: {len(lm_en.bi):,} bigrams, FR: {len(lm_fr.bi):,} bigrams")
print(f"  Stage 1: {len(en_to_fr):,} EN→FR mappings")

# Get most common bigrams from both languages
en_common = lm_en.bi.most_common(20000)
fr_common = lm_fr.bi.most_common(20000)

# For each EN bigram, can we homophone both words to FR?
matches = []
for (a,b), count in en_common:
    fa = en_to_fr.get(a, "")
    fb = en_to_fr.get(b, "")
    if fa and fb:  # both words have homophones
        phrase_en = f"{a} {b}"
        phrase_fr = f"{fa} {fb}"
        # Check FR fluency
        fr_flu = lm_fr.fluency(phrase_fr.split()) if hasattr(lm_fr,'fluency') else 0.3
        matches.append((count, a, b, fa, fb, phrase_en, phrase_fr, fr_flu))

# For each FR bigram, reverse lookup
fr_matches = []
for (a,b), count in fr_common:
    ea_list = fr_to_en.get(a, set())
    eb_list = fr_to_en.get(b, set())
    if ea_list and eb_list:
        for ea in list(ea_list)[:3]:
            for eb in list(eb_list)[:3]:
                phrase_fr = f"{a} {b}"
                phrase_en = f"{ea} {eb}"
                en_flu = lm_en.fluency(phrase_en.split()) if hasattr(lm_en,'fluency') else 0.3
                fr_matches.append((count, a, b, ea, eb, phrase_fr, phrase_en, en_flu))

matches.sort(reverse=True)
fr_matches.sort(reverse=True)

print(f"  EN→FR matches: {len(matches)}")
print(f"  FR→EN matches: {len(fr_matches)}")

# Save top matches
with open("stage4_en_to_fr_phrases.jsonl","w") as f:
    for c,a,b,fa,fb,enp,frp,flu in matches[:5000]:
        f.write(json.dumps({"en":enp,"fr":frp,"count":c,"fr_flu":round(flu,3)},ensure_ascii=False)+"\n")

with open("stage4_fr_to_en_phrases.jsonl","w") as f:
    for c,a,b,ea,eb,frp,enp,flu in fr_matches[:5000]:
        f.write(json.dumps({"fr":frp,"en":enp,"count":c,"en_flu":round(flu,3)},ensure_ascii=False)+"\n")

# Cross-reference: phrases that appear in BOTH directions
en_phrases = {m[5] for m in matches[:5000]}
fr_phrases = {m[4] for m in fr_matches[:5000]}
bidirectional = []
for c,a,b,fa,fb,enp,frp,flu in matches[:3000]:
    for fc,fa2,fb2,ea,eb,frp2,enp2,flu2 in fr_matches[:3000]:
        if enp == enp2 and frp == frp2:
            bidirectional.append((enp, frp, c+fc))
            break

print(f"  Bidirectional matches: {len(bidirectional)}")

with open("stage4_bidirectional.jsonl","w") as f:
    for enp,frp,score in sorted(bidirectional,key=lambda x:-x[2])[:2000]:
        f.write(json.dumps({"en":enp,"fr":frp,"score":score},ensure_ascii=False)+"\n")

# Show samples
print(f"\n  Top EN→FR phrases:")
for c,a,b,fa,fb,enp,frp,flu in matches[:10]:
    print(f"    [{c:5d}] {enp:25s} → {frp:25s} FR-flu={flu:.2f}")

print(f"\n  Top FR→EN phrases:") 
for c,a,b,ea,eb,frp,enp,flu in fr_matches[:10]:
    print(f"    [{c:5d}] {frp:25s} → {enp:25s} EN-flu={flu:.2f}")

print(f"\n  Top bidirectional:")
for enp,frp,score in sorted(bidirectional,key=lambda x:-x[2])[:10]:
    print(f"    [{score:5d}] {enp:25s} ↔ {frp:25s}")
