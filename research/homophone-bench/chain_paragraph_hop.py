"""Long-chain paragraph translation over hops-all.tsv (June-12 chain_paragraph
method, no embeddings). Per content word: best-product walk up to 40 hops from
the word + its synonyms, land on a fluent FR word via >=1 sound hop. Shows hops
(inflation). Stop words pass. Run: python chain_paragraph_hop.py "your sentence"
"""
import collections,heapq,sys
from wordfreq import zipf_frequency
G=collections.defaultdict(list); syn=collections.defaultdict(set)
for i,l in enumerate(open("hops-all.tsv",encoding="utf-8")):
    if i==0: continue
    a,b,t,w=l.rstrip("\n").split("\t")
    try:w=float(w)
    except:w=0.7
    G[a].append((b,t,w)); G[b].append((a,t,w))
    if t=="~syn": syn[a].add(b); syn[b].add(a)
sym={"≈snd":"≈","=trans":"=","=cog":"=","~syn":"~"}
def et(u,v): return next((t for m,t,w in G[u] if m==v),"?")
def fmt(p):
    s=p[0]
    for i in range(1,len(p)): s+=f" {sym.get(et(p[i-1],p[i]),'?')} {p[i]}"
    return s
def transfer(word,max_hops=40):
    seeds=[f"en:{word}"]+list(syn.get(f"en:{word}",()))[:5]
    pq=[]; best={}
    for s in seeds:
        if s in G: heapq.heappush(pq,(-1.0,s,[s],0)); best[s]=1.0
    while pq:
        negp,n,path,ns=heapq.heappop(pq); p=-negp
        if n.startswith("fr:") and ns>=1 and len(n[3:])>1 and zipf_frequency(n[3:],"fr")>=4.3 and len(path)>=2:
            return n[3:],path
        if len(path)>max_hops: continue
        for m,t,w in G.get(n,[])[:80]:
            if p*w>best.get(m,0)+1e-9:
                best[m]=p*w; heapq.heappush(pq,(-p*w,m,path+[m],ns+(1 if t=="≈snd" else 0)))
    return None,None
STOP={"the","a","an","is","of","and","to","in","on","with","that","it","for"}
src=(sys.argv[1] if len(sys.argv)>1 else "cold quiet forest falling leaves").split()
fr=[]
for w in src:
    if w in STOP: fr.append(w); continue
    t,path=transfer(w)
    if t: fr.append(t); print(f"  {w:9s} -> {t:10s} [{len(path)-1}h]  {fmt(path)}")
    else: fr.append(f"[{w}]"); print(f"  {w:9s} -> [disconnected: 0 path -- needs denser synonym layer]")
print(f"\nEN: {' '.join(src)}\nFR: {' '.join(fr)}")
