"""Targeted glue miner — fills the 60 missing function words using en-word-ipa.tsv
to skip espeak for the EN side. Appends to zipf-glue.tsv."""
import csv, unicodedata, sys
sys.path.insert(0, ".")
import matcher

en_ipa = {}
for l in open("en-word-ipa.tsv"):
    if l.startswith("word"): continue
    w, i = l.rstrip().split("\t", 1)
    en_ipa[w] = unicodedata.normalize("NFD", i.replace("\u02c8","").replace("\u02cc","").replace(" ","").replace(".",""))

fr_words = []
with open("data/lexique.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        fr = r.get("ortho","")
        fq = float(r.get("freqlemfilms2","0") or "0")
        if fr and fq > 1:
            fr_words.append((fr, fq))
fr_words.sort(key=lambda x: -x[1])
fr_words = [w for w,_ in fr_words[:3000]]
print(f"EN dict: {len(en_ipa)}  FR pool: {len(fr_words)}")

missing = "would could should been had did does were each every any more most only even still also much many few other own same such without within over under before after between both either neither while until since once now then there here where how what who when why than very too quite almost already always never often well might may must".split()
print(f"Mining {len(missing)} missing function words...")

results = []
for en in missing:
    e_ipa = en_ipa.get(en)
    if not e_ipa:
        e_ipa = matcher.g2p(en, "en")
    scored = []
    for fr in fr_words:
        s = matcher.homophone_score(en, "en", fr, "fr")["score"]
        if s >= 0.50:
            scored.append((s, fr))
    scored.sort(reverse=True)
    for s, fr in scored[:5]:
        results.append((en, fr, s))
    if scored:
        top = "  ".join(f"{fr}({s:.2f})" for s,fr in scored[:3])
        print(f"  {en:10s} -> {top}")
    else:
        print(f"  {en:10s} -> (none)")

with open("zipf-glue.tsv", "a") as f:
    for en, fr, s in results:
        f.write(f"{en}\t{fr}\t{s:.3f}\t0.0\n")
print(f"\nAdded {len(results)} new glue rows")
