#!/usr/bin/env python3
"""
HOMOPHONE WRITER v5 — Cross-accent generation + LLM training corpus.

NEW METHOD: cross-accent homophone generation.
  Instead of matching FR_IPA ≈ EN_IPA directly (which struggles with cross-lingual
  phonology differences), we match through TWO ears:

    EN_ear:  EN_voice(candidate) ≈ EN_voice(target)
    FR_ear:  FR_voice(candidate) ≈ EN_voice(target)

  The EN ear test: "what does this French text sound like to an English speaker?"
  The FR ear test: "is this actually pronounceable as French?"

  Score = min(EN_ear, FR_ear) — BOTH must work. This is the proper dual constraint.

  This discovers homophones that the IPA matcher misses because cross-lingual
  phonology is applied by the speech synthesizer, not approximated by edit distance.

ALSO INCLUDES:
  - Training corpus build: emits train-dual-v2.jsonl from v7 + strict_gold + chain-web
  - Ladder+chain routing (no cognates)
  - Three-iteration self-improve
  - Strict judging (never loosen)

Run: python homophone_writer_v5.py "the sea remembers every ship"
     python homophone_writer_v5.py --bench 4 --iter 3
     python homophone_writer_v5.py --build-train
"""

from __future__ import annotations

import argparse, json, os, re, subprocess, sys, unicodedata
from collections import defaultdict

import numpy as np
import panphon

# ═══════════════════════════════════════════════════════════════════
FT = panphon.FeatureTable()
N_FEATURES, SHARPEN, GAP = 24, 0.35, 0.42
VOWELS = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɑɒ"
PRI, SEC, UNST = 1.0, 0.6, 0.3

_EQUIV_RAW = [
    (["l","ɫ","w"],.20),(["θ","s","f","t"],.25),(["ð","z","d"],.25),
    (["ŋ","n"],.15),(["ŋ","ɲ"],.20),(["ɲ","n"],.15),(["p","b"],.20),
    (["t","d"],.20),(["k","ɡ"],.20),(["s","z"],.20),(["f","v"],.20),
    (["ʃ","ʒ"],.20),(["i","ɪ"],.10),(["e","ɛ"],.10),(["u","ʊ"],.10),
    (["o","ɔ"],.10),(["ɔ","ɒ"],.10),(["ɑ","ɒ"],.10),(["a","ɑ","ɐ","æ"],.15),
    (["ə","ɐ","ɜ","ʌ","ɪ","ʊ","ɛ"],.15),(["ɚ","ə"],.05),(["ɚ","œ"],.20),
    (["œ","ʌ"],.15),(["ø","œ"],.10),(["ø","e"],.20),(["y","i"],.20),
    (["y","u"],.20),(["ɥ","y"],.10),(["ɥ","w"],.15),(["j","i"],.20),
    (["w","u"],.20),(["v","w"],.20),
]
EQUIV = {}
for g,c in _EQUIV_RAW:
    for i in range(len(g)):
        for j in range(i+1,len(g)):
            EQUIV[tuple(sorted((g[i],g[j])))] = min(c, EQUIV.get(tuple(sorted((g[i],g[j]))),1.0))

CHEAP_GAP = {"h":.08,"ə":.12,"ɚ":.12,"ʔ":.08,"ʲ":.08,"ʷ":.08,"j":.18,"w":.18}

# ═══════════════════════════════════════════════════════════════════
_ECACHE = {}
def _espeak(text, lang):
    k = (text, lang)
    if k in _ECACHE: return _ECACHE[k]
    try:
        r = subprocess.run(["espeak-ng","-q","--ipa","-v",
            {"en":"en-us","fr":"fr","en-us":"en-us"}.get(lang,lang), text],
            capture_output=True, text=True, check=True)
        v = unicodedata.normalize("NFD", r.stdout.strip())
    except: v = ""
    _ECACHE[k] = v; return v

def g2p(text, lang):
    r = _espeak(text, lang)
    for c in "ˈˌ .‿|‖ˑː«»": r = r.replace(c,"")
    return r

def _segs(ipa):
    s,i,n = [],0,len(ipa)
    while i<n:
        if i+1<n and ipa[i:i+2] in {"ɑ̃","ɛ̃","ɔ̃","œ̃","t͡ʃ","d͡ʒ","t͡s"}:
            s.append(ipa[i:i+2]); i+=2
        else: s.append(ipa[i]); i+=1
    return tuple(s)

_FCACHE = {}
def _feat(seg):
    if seg in _FCACHE: return _FCACHE[seg]
    try:
        v=FT.word_to_vector_list(seg, numeric=True)
        a=np.clip(np.array(v,dtype=np.float32),-1,1) if v else np.zeros(N_FEATURES,dtype=np.float32)
        if a.ndim==2: a=a[0]
        if a.shape[0]!=N_FEATURES:
            p=np.zeros(N_FEATURES,dtype=np.float32); p[:min(a.shape[0],N_FEATURES)]=a[:min(a.shape[0],N_FEATURES)]; a=p
    except: a=np.zeros(N_FEATURES,dtype=np.float32)
    _FCACHE[seg]=a; return a

def _ecost(a,b):
    if a==b: return 0.0
    f=EQUIV.get(tuple(sorted((a,b))),0.90)
    d=min(1.0,float(np.tanh(np.sqrt(np.sum((_feat(a)-_feat(b))**2))/np.sqrt(N_FEATURES)/SHARPEN)))
    return min(f,d)

def ndice(ia,ib,n=2):
    def ng(s): return {s[i:i+n] for i in range(len(s)-n+1)} if len(s)>=n else {s}
    A,B=ng(ia),ng(ib); return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

def fnw(ia,ib):
    sa,sb=_segs(ia),_segs(ib); lx,ly=len(sa),len(sb)
    if lx==0 or ly==0: return 0.0
    D=np.full((lx+1,ly+1),np.inf,dtype=np.float32)
    D[0,:]=[sum(CHEAP_GAP.get(s,GAP) for s in sb[:j]) for j in range(ly+1)]
    D[:,0]=[sum(CHEAP_GAP.get(s,GAP) for s in sa[:i]) for i in range(lx+1)]
    for i in range(1,lx+1):
        for j in range(1,ly+1):
            D[i,j]=min(D[i-1,j-1]+_ecost(sa[i-1],sb[j-1]),
                       D[i,j-1]+CHEAP_GAP.get(sb[j-1],GAP),
                       D[i-1,j]+CHEAP_GAP.get(sa[i-1],GAP))
    return max(0.0,1.0-D[lx,ly]/max(lx,ly))

# ═══════════════════════════════════════════════════════════════════
# CROSS-ACCENT SCORING — the new primary method
# ═══════════════════════════════════════════════════════════════════
def cross_accent_score(en_text, fr_text):
    """
    Score a French candidate through TWO ears:
      EN_ear: how does it sound to an English speaker? (EN voice reading FR text)
      FR_ear: how does it sound to a French speaker? (FR voice reading FR text)
    
    Returns: cross_score = min(en_match, fr_match), plus detailed IPA.
    
    This is the proper dual constraint: a good homophone must satisfy BOTH
    the English ear (sounds like the target) AND the French ear (is actually
    pronounceable as French). The espeak synthesizer applies real phonology
    rules (vowel reduction, liaison, stress) that the IPA matcher misses.
    """
    target = g2p(en_text, "en")
    if not target: return 0.0, {"target":"","en_read":"","fr_read":"","en_match":0,"fr_match":0}

    # English speaker reading the French text
    en_read = g2p(fr_text, "en")
    # French speaker reading the French text
    fr_read = g2p(fr_text, "fr")

    en_match = 0.5*ndice(target, en_read) + 0.5*fnw(target, en_read) if en_read else 0.0
    fr_match = 0.5*ndice(target, fr_read) + 0.5*fnw(target, fr_read) if fr_read else 0.0

    # Cross score: BOTH must work (min)
    cross = min(en_match, fr_match)

    return cross, {
        "target": target, "en_read": en_read, "fr_read": fr_read,
        "en_match": en_match, "fr_match": fr_match, "cross": cross,
    }

# ═══════════════════════════════════════════════════════════════════
# PROSODIC + JOKER (sound × meaning — secondary judge)
# ═══════════════════════════════════════════════════════════════════
def prosodic_score(en,fr):
    def _stressed(text,lang):
        sa,wa=[],[]
        raw=_espeak(text,lang)
        if not raw: return [],[]
        for word in raw.split():
            dm,stops="",{}
            for ch in word:
                if ch=="ˈ": stops[len(dm)]=PRI; continue
                if ch=="ˌ": stops[len(dm)]=SEC; continue
                if ch in ".‿|‖ˑː": continue
                dm+=ch
            segs=list(_segs(dm)); sw=[UNST]*len(segs)
            off=0
            for idx,s in enumerate(segs):
                for o,swv in stops.items():
                    if off<=o<=off+len(s): sw[idx]=max(sw[idx],swv)
                off+=len(s)
            vi=[j for j,s in enumerate(segs) if any(c in VOWELS for c in s)]
            for j,s in enumerate(segs):
                if j in vi: continue
                near=[sw[k] for k in vi if abs(k-j)<=1]
                if near: sw[j]=max(sw[j],max(near)*(1.0 if any(k>j for k in vi if abs(k-j)==1) else 0.85))
            sa+=segs; wa+=sw
        return sa,wa
    sa,wa=_stressed(en,"en"); sb,wb=_stressed(fr,"fr")
    if not sa or not sb: return 0.0
    n,m=len(sa),len(sb)
    D=[[0.0]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1): D[i][0]=D[i-1][0]+GAP*wa[i-1]
    for j in range(1,m+1): D[0][j]=D[0][j-1]+GAP*wb[j-1]
    for i in range(1,n+1):
        for j in range(1,m+1):
            w=(wa[i-1]+wb[j-1])/2.0
            D[i][j]=min(D[i-1][j-1]+_ecost(sa[i-1],sb[j-1])*w,
                       D[i-1][j]+GAP*wa[i-1],D[i][j-1]+GAP*wb[j-1])
    tot=sum(wa)+sum(wb)
    match=max(0.0,1.0-2.0*D[n][m]/tot if tot else 1.0)
    sya=[wa[i] for i,s in enumerate(sa) if any(c in VOWELS for c in s)]
    syb=[wb[i] for i,s in enumerate(sb) if any(c in VOWELS for c in s)]
    rhy=1.0-abs(len(sya)-len(syb))/max(1,len(sya)+len(syb))
    if syb:
        frf=syb[-1]/max(syb) if max(syb) else 0
        fre=max(0.0,1.0-(np.std(syb)/(np.mean(syb)+1e-9)))
        frn=max(0.0,min(1.0,0.6*frf+0.4*fre))
    else: frn=0.5
    return 0.6*match+0.2*frn+0.2*rhy

_SEM=None
def semcos(en,fr):
    global _SEM
    if _SEM is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SEM=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except: _SEM=False
    if _SEM is False: return 0.5
    try:
        v=_SEM.encode([en,fr],normalize_embeddings=True); return float(v[0]@v[1])
    except: return 0.5

def joker_judge(en, fr):
    s = prosodic_score(en, fr.replace("'"," "))
    m = semcos(en, fr)
    return s*m, s, m

# ═══════════════════════════════════════════════════════════════════
# UNIFIED DATABASE
# ═══════════════════════════════════════════════════════════════════
_DB=None
def load_db(b="."):
    global _DB
    if _DB is not None: return _DB
    db={
        "strict_gold":defaultdict(list),"v7_gold":defaultdict(list),
        "dual":defaultdict(list),"ladder":defaultdict(list),"glue":defaultdict(list),
        "chain":defaultdict(lambda:defaultdict(list)),
        "fr_class":{},"en_class":{},
        "syn_en":defaultdict(set),"syn_fr":defaultdict(set),
        "trans":defaultdict(set),"bridge":defaultdict(list),
        "fr_vocab":set(),
        "fr_idx":{},"fr_units":[],"fr_bylen":None,"unit_bylen":None,
    }
    # strict-gold
    try:
        for i,line in enumerate(open(f"{b}/strict-gold.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=2: db["strict_gold"][p[0]].append((p[1],1.0,0.9))
    except: pass
    # v7 gold
    try:
        for i,line in enumerate(open(f"{b}/dictionary-v7.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=9 and p[3]=="1":
                try: db["v7_gold"][p[7]].append((p[8],float(p[1]),1.0,p[0]))
                except: pass
    except: pass
    # tier-ladder
    try:
        for i,line in enumerate(open(f"{b}/tier-ladder.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=12 and p[10]:
                try: db["ladder"][p[1]].append((float(p[10]),float(p[11]) if p[11] else 0.5,p[2]))
                except: continue
    except: pass
    # chain-web
    try:
        for i,line in enumerate(open(f"{b}/chain-web-full-v7u.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=5:
                a,b,hops,q = p[0],p[1],int(p[2]),float(p[3])
                if ":" in a and ":" in b:
                    sl,sw = a.split(":",1); tl,tw = b.split(":",1)
                    if sl=="en" and tl=="fr": db["chain"][sw][tw].append((hops,q))
                    elif sl=="fr" and tl=="en": db["chain"][tw][sw].append((hops,q))
    except: pass
    # dual-pairs
    for fp in [f"{b}/dual-pairs.tsv","dual-pairs.tsv"]:
        try:
            for i,line in enumerate(open(fp,encoding="utf-8")):
                if i==0: continue
                p=line.rstrip("\n").split("\t")
                if len(p)>=6:
                    en_w, fr_w = p[0], p[1]
                    # Filter: skip identity pairs where en==fr and not real French
                    if en_w.lower() == fr_w.lower():
                        continue  # these are polluted (moon=moon, ship=ship)
                    db["dual"][en_w].append((float(p[2]),fr_w))
            break
        except: continue
    # zipf-glue
    try:
        for i,line in enumerate(open(f"{b}/zipf-glue.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=3: db["glue"][p[0]].append((float(p[2]),p[1]))
    except: pass
    # homophone classes
    for path,target in [("fr-homophone-classes-lexique.tsv","fr_class"),
                        ("fr-homophone-classes.tsv","fr_class"),
                        ("en-homophone-classes.tsv","en_class")]:
        try:
            for i,line in enumerate(open(f"{b}/{path}",encoding="utf-8")):
                if i==0: continue
                ms=line.rstrip("\n").split("\t")[1].split()
                for m in ms: db[target][m]=ms
        except: pass
    # synonyms
    try:
        for line in open(f"{b}/muse-pivot-syn.tsv",encoding="utf-8"):
            a,b,_=line.rstrip("\n").split("\t")
            if a.startswith("en:") and b.startswith("en:"):
                db["syn_en"][a[3:]].add(b[3:]); db["syn_en"][b[3:]].add(a[3:])
            elif a.startswith("fr:") and b.startswith("fr:"):
                db["syn_fr"][a[3:]].add(b[3:]); db["syn_fr"][b[3:]].add(a[3:])
    except: pass
    # MUSE
    for mp in [f"{b}/muse-en-fr.txt","/tmp/muse-en-fr.txt"]:
        try:
            for line in open(mp,encoding="utf-8"):
                p=line.split()
                if len(p)==2: db["trans"][p[0]].add(p[1])
            break
        except: continue
    # bridges
    try:
        for i,line in enumerate(open(f"{b}/llm-bridge.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=4: db["bridge"][p[0]].append((float(p[2]),float(p[3]),p[1]))
    except: pass
    # babel
    for fp in [f"{b}/fr-word-ipa.tsv","fr-word-ipa.tsv"]:
        try:
            for i,line in enumerate(open(fp,encoding="utf-8")):
                if i==0: continue
                p=line.rstrip("\n").split("\t")
                if len(p)>=2 and p[1]: db["fr_idx"][p[0]]=p[1]
            break
        except: continue
    for fp in [f"{b}/fr-units.tsv","fr-units.tsv"]:
        try:
            for i,line in enumerate(open(fp,encoding="utf-8")):
                if i==0: continue
                p=line.rstrip("\n").split("\t")
                if len(p)>=3: db["fr_units"].append((p[0],p[1],p[2]))
            break
        except: continue
    # vocab
    for w in db["fr_idx"]:
        if "(en)" not in w: db["fr_vocab"].add(w)
    for cls in db["fr_class"].values():
        for w in cls:
            if w: db["fr_vocab"].add(w)
    # babel indices
    fr_bl=defaultdict(list)
    for w,p in db["fr_idx"].items(): fr_bl[len(_segs(p))].append((w,p))
    db["fr_bylen"]=dict(fr_bl)
    u_bl=defaultdict(list)
    for u,p,k in db["fr_units"]: u_bl[len(_segs(p))].append((f"{u}〔{k}〕",p))
    db["unit_bylen"]=dict(u_bl)
    for k in ("dual","glue","ladder","bridge"):
        for v in db[k].values(): v.sort(reverse=True)

    _DB=db; return db

# ═══════════════════════════════════════════════════════════════════
# CROSS-ACCENT CANDIDATE SELECTION
# ═══════════════════════════════════════════════════════════════════
def cross_accent_candidates(en_word, db, top=12):
    """Generate candidates, score by cross-accent (min EN_ear, FR_ear)."""
    results = []
    seen = set()

    def add(fr, channel, pre_score=None):
        fr = fr.lower().strip(".,;:!?'\"«»")
        if fr in seen or not fr: return
        if " " not in fr and db["fr_vocab"] and fr not in db["fr_vocab"]: return
        seen.add(fr)
        if pre_score is not None:
            cross = pre_score
            _, ca = cross_accent_score(en_word, fr) if pre_score is None else (pre_score, {})
        else:
            cross, ca = cross_accent_score(en_word, fr)
        sem = semcos(en_word, fr)
        # Combined: cross-accent × semantic (product, strict)
        combined = cross * sem
        results.append((combined, cross, sem, fr, channel, ca))

    # 1. STRICT GOLD
    for fr, s, m in db["strict_gold"].get(en_word, [])[:5]:
        add(fr, "S_gold", s * m if s and m else None)
    # 2. V7 GOLD
    for fr, s, m, tier in db["v7_gold"].get(en_word, [])[:5]:
        add(fr, f"v7_{tier}", s * m if s and m else None)
    # 3. LADDER
    for s, m, fr in db["ladder"].get(en_word, [])[:8]:
        cross, ca = cross_accent_score(en_word, fr)
        combined = cross * semcos(en_word, fr)
        results.append((combined, cross, semcos(en_word, fr), fr, "ladder", ca))
    # 4. CHAIN-WEB
    for fr, edges in db["chain"].get(en_word, {}).items():
        if edges:
            best = min(edges, key=lambda e: e[0])
            cross, ca = cross_accent_score(en_word, fr)
            sem = semcos(en_word, fr)
            trust = 1.0 - 0.15*(best[0]-1)
            results.append((cross*sem*trust, cross, sem, fr, f"chain_h{best[0]}", ca))
    # 5. DUAL (identity-filtered)
    for s, fr in db["dual"].get(en_word, [])[:6]:
        cross, ca = cross_accent_score(en_word, fr)
        sem = semcos(en_word, fr)
        results.append((cross*sem, cross, sem, fr, "dual", ca))
    # 6. GLUE
    ART = {"the":{"le","la","les","de","des","du"},"a":{"un","une","à","et"},"an":{"un","une"}}
    for s, fr in db["glue"].get(en_word, [])[:4]:
        if en_word in ART and fr not in ART[en_word]: continue
        cross, ca = cross_accent_score(en_word, fr)
        results.append((cross*0.6, cross, 0.6, fr, "glue", ca))
    # 7. EN HOMOPHONE CLASS pivot
    for sib in db["en_class"].get(en_word, [])[:5]:
        if sib==en_word: continue
        for _s, fr in db["dual"].get(sib, [])[:2]:
            cross, ca = cross_accent_score(en_word, fr)
            results.append((cross*0.6, cross, 0.6, fr, f"enclass:{sib}", ca))
    # 8. FR HOMOPHONE CLASS meaning-max
    if results:
        best = results[0]
        if " " not in best[3]:
            for sib in db["fr_class"].get(best[3],[])[:6]:
                if sib not in seen and (not db["fr_vocab"] or sib in db["fr_vocab"]):
                    seen.add(sib)
                    m = semcos(en_word, sib)
                    if m > best[2]:
                        cross, ca = cross_accent_score(en_word, sib)
                        results.append((cross*m, cross, m, sib, "frclass", ca))

    results.sort(reverse=True)
    return results[:top]

# ═══════════════════════════════════════════════════════════════════
# BABEL WINDOWS
# ═══════════════════════════════════════════════════════════════════
def wmatch(ipa, bylen, top=5, tol=2, max_cands=2000):
    n=len(_segs(ipa)); cands=[]
    for L in range(max(1,n-tol),n+tol+1):
        bucket=bylen.get(L,[])
        if len(cands)+len(bucket)<=max_cands: cands.extend(bucket)
        else: cands.extend(bucket[:max_cands-len(cands)]); break
    scored=[]
    for w,p in cands:
        s=0.5*ndice(ipa,p)+0.5*fnw(ipa,p)
        if s>=0.55: scored.append((s,w))
    scored.sort(reverse=True); return scored[:top]

def babel_many(gram, db, top=4):
    ipa=g2p(gram,"en")
    if not ipa: return []
    hits=wmatch(ipa,db["fr_bylen"],top=top)
    uhits=wmatch(ipa,db["unit_bylen"],top=2)
    return sorted(hits+uhits,reverse=True)[:top]

# ═══════════════════════════════════════════════════════════════════
# DECOMPOSE
# ═══════════════════════════════════════════════════════════════════
_MANUAL = {"remembers":"remember","remembered":"remember","ships":"ship",
    "sleeps":"sleep","slept":"sleep","calls":"call","blesses":"bless",
    "blessed":"bless","made":"make","makes":"make","walks":"walk",
    "says":"say","said":"say","goes":"go","went":"go","gone":"go",
    "has":"have","had":"have","does":"do","done":"do","did":"do",
    "sees":"see","saw":"see","seen":"see","answers":"answer","answered":"answer"}
_SUFFIXES = [("ingly",""),("ings",""),("ment",""),("ness",""),("tion","t"),
    ("able",""),("ible",""),("ally","al"),("izes","ize"),("ised","ise"),
    ("izing","ise"),("ized","ize"),("ers","er"),("est",""),("er",""),
    ("ed",""),("ing",""),("s",""),("'s",""),("'d",""),("'ll",""),("n't","")]

def decompose(w):
    wl=w.lower()
    if wl in _MANUAL: return wl, _MANUAL[wl], w[len(_MANUAL[wl]):]
    for suf,rep in _SUFFIXES:
        if wl.endswith(suf) and len(wl)-len(suf)>=3:
            return wl, wl[:len(wl)-len(suf)]+rep, suf
    return wl, wl, ""

# ═══════════════════════════════════════════════════════════════════
# CLIMB
# ═══════════════════════════════════════════════════════════════════
def climb(en, fr, db, passes=3):
    cross, ca = cross_accent_score(en, fr)
    best_c, best_m = cross, semcos(en, fr)
    words = fr.split()
    for _ in range(passes):
        improved = False
        for i,w in enumerate(words):
            key = w.strip(",.;:!?'\"«»").lower()
            for alt in db["fr_class"].get(key,[])[:6]:
                if alt==key: continue
                cand = " ".join(words[:i]+[alt]+words[i+1:])
                c2, _ = cross_accent_score(en, cand)
                if c2<=best_c: continue
                m2 = semcos(en, cand)
                if m2>=best_m-0.05:
                    words[i]=alt; best_c,best_m=c2,m2; improved=True; break
            if improved: break
        if not improved: break
    return " ".join(words)

# ═══════════════════════════════════════════════════════════════════
# WRITE
# ═══════════════════════════════════════════════════════════════════
def write(line, db=None, verbose=True):
    if db is None: db = load_db()
    ws_raw = [w.lower().strip(".,;:!?'\"") for w in line.split() if w.strip(".,;:!?'\"")]
    if not ws_raw: return None

    decomposed = [(w, decompose(w)[1], decompose(w)[2]) for w in ws_raw]

    # ── 1. CROSS-ACCENT CANDIDATES ──
    picks = []
    for orig, stem, suffix in decomposed:
        cands = cross_accent_candidates(stem, db, top=10)
        if not cands and stem != orig:
            cands = cross_accent_candidates(orig, db, top=10)
        if cands:
            _, cross, sem, fr, ch, ca = cands[0]
            picks.append((orig, fr, cross, sem, ch, ca))
        else:
            picks.append((orig, orig, 0.0, 0.0, "miss", {}))

    if verbose:
        print(f"Phase 1 (cross-accent, {len(picks)} words):")
        for w,fr,cross,sem,ch,ca in picks:
            en_e = ca.get("en_match",0); fr_e = ca.get("fr_match",0)
            gold = " ★" if ch in ("S_gold","v7_S","v7_A") else ""
            chain = " ◈" if "chain" in ch else ""
            print(f"  {w:15s} → {fr:18s} [{ch:12s}; cross={cross:.3f} "
                  f"EN={en_e:.2f} FR={fr_e:.2f} sem={sem:.2f}]{gold}{chain}")

    raw_fr = " ".join(fr for _,fr,_,_,_,_ in picks)

    # ── 2. BABEL WINDOW ──
    merged = list(picks)
    i=0
    while i < len(ws_raw)-1:
        for span in (3,2):
            if i+span>len(ws_raw): continue
            gram=" ".join(ws_raw[i:i+span])
            hits=babel_many(gram,db,top=3)
            if hits and hits[0][0]>=0.72:
                fr_unit=hits[0][1].split("〔")[0]
                cross, ca = cross_accent_score(gram, fr_unit)
                sem = semcos(gram, fr_unit)
                merged[i]=(gram,fr_unit,cross,sem,"babel_m1",ca)
                for j in range(i+1,i+span): merged[j]=None
                i+=span; break
        else: i+=1
    picks=[p for p in merged if p is not None]
    merged_fr=" ".join(fr for _,fr,_,_,_,_ in picks)

    if verbose and merged_fr!=raw_fr:
        print(f"\nPhase 2 (babel window):")
        for w,fr,cross,sem,ch,ca in picks:
            en_e = ca.get("en_match",0); fr_e = ca.get("fr_match",0)
            print(f"  {w:15s} → {fr:18s} [cross={cross:.3f} EN={en_e:.2f} FR={fr_e:.2f}]")

    # ── 3. CLIMB ──
    climbed = climb(line, merged_fr, db)
    if verbose and climbed!=merged_fr:
        print(f"\nPhase 3 (climb): {merged_fr} → {climbed}")

    # ── 4. FINAL SCORES ──
    cross, ca = cross_accent_score(line, climbed)
    _, s_sound, s_sem = joker_judge(line, climbed)

    in_rooten = s_sound >= 0.55 and s_sem >= 0.45
    band = "✓ ROOTEN" if in_rooten else ("~ EDGE" if cross >= 0.25 else "  below")

    if verbose:
        print(f"\nPhase 4 (cross-accent + JOKER):")
        print(f"  cross-accent = {cross:.3f}  (EN-ear={ca['en_match']:.3f}  FR-ear={ca['fr_match']:.3f})")
        print(f"  JOKER        = {s_sound:.3f} × {s_sem:.3f} = {s_sound*s_sem:.3f}")
        print(f"  EN hears: [{ca['en_read']}]")
        print(f"  FR hears: [{ca['fr_read']}]")
        print(f"  TARGET:   [{ca['target']}]")
        print(f"\n{'='*60}")
        print(f"EN  : {line}")
        print(f"FR  : {climbed}")
        print(f"BAND: {band}  cross={cross:.3f}  JOKER={s_sound*s_sem:.3f}")

    return {"en":line,"fr":climbed,"cross":cross,"sound":s_sound,"semantic":s_sem,
            "en_ear":ca["en_match"],"fr_ear":ca["fr_match"],
            "en_ipa":ca["en_read"],"fr_ipa":ca["fr_read"],"target_ipa":ca["target"],
            "rooten":in_rooten}

# ═══════════════════════════════════════════════════════════════════
# SELF-IMPROVE (3 iterations)
# ═══════════════════════════════════════════════════════════════════
def self_improve(line, db, iterations=3, verbose=True):
    if verbose: print(f"\n{'#'*60}\n# SELF-IMPROVE x{iterations}: '{line}'\n{'#'*60}")
    best = write(line, db, verbose=verbose)
    if not best: return None

    for it in range(1, iterations):
        words = best["fr"].split()
        ws_en = [w.lower().strip(".,;:!?'\"") for w in line.split() if w.strip(".,;:!?'\"")]
        if len(words) != len(ws_en): break

        if verbose: print(f"\n--- ITERATION {it+1}/{iterations} ---")
        # Score each word pair by cross-accent
        word_scores = []
        for en_w, fr_w in zip(ws_en, words):
            c, ca = cross_accent_score(en_w, fr_w)
            word_scores.append((c, ca["en_match"], ca["fr_match"], en_w, fr_w))

        weakest = min(word_scores, key=lambda x: x[0])
        if verbose:
            for c,en_e,fr_e,ew,fw in word_scores:
                mark = " ← WEAKEST" if (ew,fw)==(weakest[3],weakest[4]) else ""
                print(f"  {ew:15s} → {fw:15s} cross={c:.3f} EN={en_e:.2f} FR={fr_e:.2f}{mark}")

        # Try alternatives for weakest
        alt_cands = cross_accent_candidates(weakest[3], db, top=8)
        improved = False
        for _, cross, sem, alt_fr, ch, ca in alt_cands[1:]:
            new_words = [fw if (ew,fw)!=(weakest[3],weakest[4]) else alt_fr
                        for ew,fw in zip(ws_en, words)]
            new_fr = " ".join(new_words)
            c2, _ = cross_accent_score(line, new_fr)
            if c2 > best["cross"]:
                best["fr"] = new_fr; best["cross"] = c2; improved = True
                if verbose:
                    print(f"  IMPROVE: {weakest[4]} → {alt_fr} [{ch}; cross {best['cross']:.3f} → {c2:.3f}]")
                break
        if not improved and verbose:
            print(f"  No improvement for '{weakest[4]}'.")
            if it>=2: break

    # Final scores
    cross, ca = cross_accent_score(line, best["fr"])
    best["cross"] = cross
    best["en_ear"] = ca["en_match"]
    best["fr_ear"] = ca["fr_match"]
    best["en_ipa"] = ca["en_read"]
    best["fr_ipa"] = ca["fr_read"]
    best["target_ipa"] = ca["target"]
    _, s_s, s_m = joker_judge(line, best["fr"])
    best["sound"] = s_s; best["semantic"] = s_m
    best["rooten"] = s_s>=0.55 and s_m>=0.45

    if verbose:
        print(f"\nFINAL: {best['fr']}")
        print(f"  cross={best['cross']:.3f}  JOKER={s_s*s_m:.3f}")
        print(f"  EN-ear={ca['en_match']:.3f}  FR-ear={ca['fr_match']:.3f}")
        print(f"  BAND: {'✓ ROOTEN' if best['rooten'] else '~ EDGE'}")
    return best

# ═══════════════════════════════════════════════════════════════════
# BUILD TRAINING CORPUS
# ═══════════════════════════════════════════════════════════════════
def build_training_corpus(db=None):
    if db is None: db = load_db()
    rows = []

    def add(en, fr, sound, meaning, tier, source):
        rows.append(json.dumps({
            "prompt": f"Rewrite so it sounds the same in French: {en}",
            "completion": fr,
            "sound": sound, "meaning": meaning, "tier": tier, "source": source,
        }, ensure_ascii=False))

    # v7 gold pairs
    for en_word, entries in db["v7_gold"].items():
        for fr, s, m, tier in entries[:3]:
            add(en_word, fr, s, m, tier, "v7_gold")
    # strict-gold pairs
    for en_word, entries in db["strict_gold"].items():
        for fr, s, m in entries[:3]:
            add(en_word, fr, s, m, "S", "strict_gold")
    # ladder pairs (top by sound)
    for en_word, entries in db["ladder"].items():
        for s, m, fr in sorted(entries, reverse=True)[:2]:
            if s >= 0.70:
                add(en_word, fr, s, m, "ladder", "ladder")
    # dual pairs (top by sound, identity-filtered)
    for en_word, entries in db["dual"].items():
        for s, fr in sorted(entries, reverse=True)[:2]:
            if s >= 0.75 and en_word.lower() != fr.lower():
                add(en_word, fr, s, 0.8, "dual", "dual")
    # glue
    for en_word, entries in db["glue"].items():
        for s, fr in entries[:2]:
            if s >= 0.50:
                add(en_word, fr, s, 0.6, "glue", "glue")

    path = "train-dual-v2.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(r + "\n")
    print(f"Training corpus: {len(rows)} rows → {path}")
    print(f"GPU training: python selflearn/train_selflearn.py --base Qwen/Qwen2.5-1.5B-Instruct --data {path} --rounds 4")
    return path

# ═══════════════════════════════════════════════════════════════════
# BENCH
# ═══════════════════════════════════════════════════════════════════
CORPUS = [
    "we see the moon at dawn","mary had a little lamb",
    "the sea remembers every ship","we call to the moon and she answers",
    "less debt less mess more soup","my sorrow sleeps in a deep well",
    "bless the dawn that made us free","the cat sat on the mat",
]

def bench(n=4, iters=1, db=None):
    if db is None: db=load_db()
    results=[]
    for line in CORPUS[:n]:
        print(f"\n{'─'*60}")
        r = self_improve(line, db, iters, verbose=True) if iters>1 else write(line, db, verbose=True)
        if r: results.append(r)
    if results:
        crosses=[r["cross"] for r in results]
        sounds=[r["sound"] for r in results]
        sems=[r["semantic"] for r in results]
        en_ears=[r["en_ear"] for r in results]
        fr_ears=[r["fr_ear"] for r in results]
        rooten=sum(1 for r in results if r["rooten"])
        print(f"\n{'='*60}")
        print(f"BENCH ({len(results)} lines, {iters} iter):")
        print(f"  cross-accent: {np.mean(crosses):.3f}±{np.std(crosses):.3f}")
        print(f"  EN-ear: {np.mean(en_ears):.3f}  FR-ear: {np.mean(fr_ears):.3f}")
        print(f"  JOKER sound: {np.mean(sounds):.3f}  semantic: {np.mean(sems):.3f}")
        print(f"  Rooten: {rooten}/{len(results)} ({100*rooten/len(results):.0f}%)")
        for r in sorted(results, key=lambda x:-x["cross"])[:3]:
            print(f"  cross={r['cross']:.3f}  {r['en']} → {r['fr']}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("text",nargs="*")
    ap.add_argument("--bench",type=int,default=0)
    ap.add_argument("--iter",type=int,default=1)
    ap.add_argument("--build-train",action="store_true")
    ap.add_argument("--db-only",action="store_true")
    ap.add_argument("--base-dir",default=".")
    args=ap.parse_args()
    os.chdir(args.base_dir)
    db=load_db(args.base_dir)
    print(f"DB: {sum(len(v) for v in db['dual'].values())} dual, "
          f"{len(db['strict_gold'])} strict-gold, {len(db['v7_gold'])} v7-gold, "
          f"{len(db['ladder'])} ladder, {sum(len(v) for v in db['chain'].values())} chain")
    print(f"     NO COGNATES. Dual identity-filtered.\n")
    if args.db_only: return
    if args.build_train: build_training_corpus(db); return
    iters = min(args.iter, 3)
    if args.bench: bench(args.bench, iters, db); return
    if args.text:
        for t in args.text:
            if iters>1: self_improve(t, db, iters, verbose=True)
            else: write(t, db, verbose=True)
    else: bench(3, iters, db)

if __name__=="__main__": main()
