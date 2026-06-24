"""Resolve a 'gap' word by HOPPING the chain until a useful FR landing.
No word is a dead-end (giant component): walk ≈snd/=trans/~syn until you reach a
fluent French word reached via >=1 sound hop. Prints the path.
Run: python chain_hop.py cold quiet trees guy
"""
import collections,heapq,sys
from wordfreq import zipf_frequency
G=collections.defaultdict(list)
for i,l in enumerate(open("hops-all.tsv",encoding="utf-8")):
    if i==0: continue
    a,b,t,w=l.rstrip("\n").split("\t")
    try: w=float(w)
    except: w=0.7
    G[a].append((b,t,w)); G[b].append((a,t,w))
def hop(start,max_hops=6):
    pq=[(-1.0,start,[start],0)]; best={start:1.0}
    while pq:
        negp,n,path,ns=heapq.heappop(pq); p=-negp
        if n.startswith("fr:") and ns>=1 and len(n[3:])>1 and zipf_frequency(n[3:],"fr")>=4.0 and len(path)>=3:
            return path,p
        if len(path)>max_hops: continue
        for m,t,w in G.get(n,[])[:60]:
            np_=p*w
            if np_>best.get(m,0)+1e-9:
                best[m]=np_; heapq.heappush(pq,(-np_,m,path+[m],ns+(1 if t=="≈snd" else 0)))
    return None,0
sym={"≈snd":"≈","=trans":"=","=cog":"=","~syn":"~"}
def fmt(path):
    s=path[0]
    for i in range(1,len(path)):
        typ=next((t for m,t,w in G[path[i-1]] if m==path[i]),"?")
        s+=f" {sym.get(typ,'?')} {path[i]}"
    return s
for w in (sys.argv[1:] or ["cold","quiet","trees","guy"]):
    p,q=hop(f"en:{w}")
    print(f"{w:9s} -> {p[-1][3:]:12s} | {fmt(p)}" if p else f"{w:9s} -> none (not in web; add to dict)")
