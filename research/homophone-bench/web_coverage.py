"""Coverage of EN/FR dictionaries against the FULL 195k web node set
(homophone pairs + MUSE meaning layer), not just the homophone dataset side."""
import pickle
from wordfreq import top_n_list
g = pickle.load(open("graph-v7u.pkl","rb"))
nodes = g["edges"].keys()
en_web = {n[3:] for n in nodes if n.startswith("en:")}
fr_web = {n[3:] for n in nodes if n.startswith("fr:")}
print(f"full web: {len(en_web)} en-nodes, {len(fr_web)} fr-nodes")
for lang, side in (("en", en_web), ("fr", fr_web)):
    top = top_n_list(lang, 60000)
    print(f"  {lang.upper()} dictionary vs FULL WEB {lang}-nodes:")
    for n in (1000, 5000, 10000, 30000, 60000):
        words = [w for w in top[:n] if w.isalpha()]
        hit = sum(1 for w in words if w in side)
        print(f"    top {n:6d}: {hit:6d}/{len(words):6d} ({hit/len(words)*100:5.1f}%)")
