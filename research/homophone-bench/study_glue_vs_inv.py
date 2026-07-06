"""Quantify: how many glue words have BETTER S-tier inventory matches?"""
import sys; sys.path.insert(0,".")
import matcher

# Load full S/GOLD/STRICT-GOLD non-identity inventory
inventory = {}  # en -> [(fr, score)]
for l in open("../pipeline/tier-ladder-cycle3.tsv"):
    p = l.rstrip().split("\t")
    if len(p)>=11 and p[3] in ("S","GOLD","STRICT-GOLD","DUAL-S","LOOP1","LOOP2"):
        en, fr = p[1].strip(), p[2].strip()
        if en and fr and en != fr and en.isalpha() and fr.isalpha():
            s = float(p[9]) if p[9] else 0.75
            inventory.setdefault(en, []).append((fr, s))

# Load glue
glue = {}  # en -> [(fr, score)]
for l in open("zipf-glue.tsv"):
    if l.startswith("en"): continue
    p = l.rstrip().split("\t")
    glue.setdefault(p[0], []).append((p[1], float(p[2])))

# For each glue word, find best S-tier match and compare
better = 0
worse = 0
no_inv = 0
upgrades = []

for en, g_pairs in glue.items():
    best_g = max(g_pairs, key=lambda x: x[1])
    if en not in inventory:
        no_inv += 1
        continue
    best_i = max(inventory[en], key=lambda x: x[1])
    # Score both with matcher
    gs = matcher.homophone_score(en, "en", best_g[0], "fr")["score"]
    is_ = matcher.homophone_score(en, "en", best_i[0], "fr")["score"]
    if is_ > gs + 0.02:  # meaningful improvement
        better += 1
        upgrades.append((en, best_g[0], gs, best_i[0], is_, is_-gs))
    elif gs > is_ + 0.02:
        worse += 1

print(f"Glue words: {len(glue)}, with S-tier inventory: {len(glue)-no_inv}, without: {no_inv}")
print(f"Inventory BETTER: {better}, glue BETTER: {worse}, tie: {len(glue)-no_inv-better-worse}")
print(f"\nTop upgrades (S-tier > glue by 0.02+ matcher):")
upgrades.sort(key=lambda x: -x[5])
for en, gfr, gs, ifr, is_, delta in upgrades[:20]:
    print(f"  {en:12s}  glue: {gfr:16s} {gs:.3f}  ->  S-tier: {ifr:16s} {is_:.3f}  (+{delta:.3f})")
