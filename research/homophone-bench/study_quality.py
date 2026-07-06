"""Study: compare glue pairs vs S-tier quality, find better matches."""
import sys; sys.path.insert(0,".")
import matcher, csv

# S-tier non-identity pairs
s_pairs = []
for l in open("../pipeline/tier-ladder-cycle3.tsv"):
    p = l.rstrip().split("\t")
    if len(p)>=11 and p[3]=="S" and p[1]!=p[2] and p[1].isalpha() and p[2].isalpha():
        s_pairs.append((float(p[9]) if p[9] else 0, p[1], p[2]))
s_pairs.sort(reverse=True)

# Glue pairs
glue_pairs = []
for l in open("zipf-glue.tsv"):
    if l.startswith("en"): continue
    p = l.rstrip().split("\t")
    glue_pairs.append((float(p[2]), p[0], p[1]))
glue_pairs.sort(reverse=True)

# Score top S-tier with matcher
print("=== S-TIER (non-identity) top 12 ===")
for s,en,fr in s_pairs[:12]:
    r = matcher.homophone_score(en,"en",fr,"fr")
    print(f"  {s:.3f}  matcher={r['score']:.3f}  {en:14s} -> {fr}")
print(f"  ... {len(s_pairs)} pairs, mean lad={sum(x[0] for x in s_pairs)/len(s_pairs):.3f}")

# Score top glue
print("\n=== GLUE top 12 ===")
for s,en,fr in glue_pairs[:12]:
    r = matcher.homophone_score(en,"en",fr,"fr")
    print(f"  {s:.3f}  matcher={r['score']:.3f}  {en:14s} -> {fr}")
print(f"  ... {len(glue_pairs)} pairs, mean={sum(x[0] for x in glue_pairs)/len(glue_pairs):.3f}")

# Find S-tier matches for glue words — could glue use better inventory matches?
print("\n=== GLUE words with S-tier inventory matches ===")
glue_words = set(p[1] for p in glue_pairs)
for s,en,fr in s_pairs:
    if en in glue_words:
        r = matcher.homophone_score(en,"en",fr,"fr")
        # find current best glue match
        best_glue = max((g for g in glue_pairs if g[1]==en), key=lambda x:x[0], default=(0,"",""))
        print(f"  {en:14s}  GLUE best: {best_glue[2]:14s} ({best_glue[0]:.3f})  S-tier: {fr:14s} ({s:.3f})  matcher: {r['score']:.3f}")
