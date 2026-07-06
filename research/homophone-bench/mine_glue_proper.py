"""Mine the 28 missing function words using pipeline methods:
lexique IPA + allophone_layer + matcher. Zero espeak calls."""
import sys; sys.path.insert(0,".")
import matcher
from lexicon_g2p import load_fr, clean_ipa
import allophone_layer

# FR lexicon with IPA
fr_lex = load_fr()
fr_ipa = {w: p[0] for w,p in fr_lex.items() if len(w)>=2 and len(w)<=10}

# EN IPA dictionary
en_ipa = {}
for l in open("en-word-ipa.tsv"):
    if l.startswith("word"): continue
    w,i = l.rstrip().split("\t",1)
    en_ipa[w] = clean_ipa(i)

# 28 missing function words (no inventory, weak/no glue)
missing = "a after almost already also always and as between each few have he here his how often once one only other quite that the while".split()

# Filter long FR words by segment length window for speed
def seg_len(ipa): return len(matcher._segs(ipa))
fr_by_len = {}
for w,p in fr_ipa.items():
    fr_by_len.setdefault(seg_len(p), []).append((w,p))

print(f"FR lexicon: {len(fr_ipa)} words indexed by segment length")
print(f"Mining {len(missing)} words with allophone-aware scoring...\n")

results = []
for en in missing:
    e_ipa = en_ipa.get(en)
    if not e_ipa: continue
    n = seg_len(e_ipa)
    if n < 1: continue
    # Only compare FR words with similar segment count (+-3)
    cand_pool = []
    for L in range(max(1,n-3), n+4):
        cand_pool.extend(fr_by_len.get(L, []))
    
    scored = []
    for fr, f_ipa in cand_pool[:2000]:  # limit to speed up
        _, best = allophone_layer.allophone_score(en, fr)
        if best >= 0.55:
            scored.append((best, fr))
    scored.sort(reverse=True)
    for s, fr in scored[:4]:
        results.append((en, fr, round(s,3)))
    
    if scored:
        top = "  ".join(f"{fr}({s:.2f})" for s,fr in scored[:3])
        print(f"  {en:10s} -> {top}")
    else:
        print(f"  {en:10s} -> (none >=0.55)")

# Append to glue table
with open("zipf-glue.tsv", "a") as f:
    for en,fr,s in results:
        f.write(f"{en}\t{fr}\t{s:.3f}\t0.0\n")
print(f"\nAdded {len(results)} new glue rows for {len(set(r[0] for r in results))} words")
