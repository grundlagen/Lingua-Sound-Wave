#!/usr/bin/env python3
"""
HOMOPHONE WRITER v4 — Ladder+chain routing, cross-accent testing, self-improve.

ORBIT-LEVEL CHANGES:
  - NO cognates. Cognates are polluted (English words like "moon" incorrectly
    catalogued as French). Removed entirely.
  - ROUTING: strict_gold → v7_gold → ladder → chain-web → dual → window → glue
  - CROSS-ACCENT TEST: "picture the phonetics" — synthesize how the French output
    sounds to an English ear (EN voice reading FR text) AND to a French ear
    (FR voice reading FR text). Match both against the English source IPA.
  - JOKER JUDGING: reward = sound × meaning. Strict. No heuristic blending.
    No loosening.
  - THREE-ITERATION SELF-IMPROVE: generate v1 → cross-accent judge → swap bad
    words → re-judge → swap worst → re-judge. Never loosen the judge.

Run: python homophone_writer_v4.py "the sea remembers every ship"
     python homophone_writer_v4.py --bench 4
     python homophone_writer_v4.py --iter 3 "mary had a little lamb"
"""

from __future__ import annotations

import argparse, os, re, subprocess, sys, unicodedata
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
    for c in "ˈˌ .‿|‖ˑː": r = r.replace(c,"")
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

# ═══════════════════════════════════════════════════════════════════
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

def combo(en,fr):
    qi,ci=g2p(en,"en"),g2p(fr,"fr")
    return 0.5*ndice(qi,ci)+0.5*fnw(qi,ci) if qi and ci else 0.0

# ═══════════════════════════════════════════════════════════════════
# CROSS-ACCENT PHONETIC TEST ("picture the phonetics")
# ═══════════════════════════════════════════════════════════════════
def cross_accent_test(en_text, fr_text):
    """Test how FR text sounds to EN ear AND FR ear. Return both scores + IPA."""
    target = g2p(en_text, "en")
    # English speaker reading the French text (applying EN phonology)
    en_read = g2p(fr_text, "en")
    # French speaker reading the French text (native)
    fr_read = g2p(fr_text, "fr")

    en_score = 0.5*ndice(target, en_read) + 0.5*fnw(target, en_read) if en_read else 0.0
    fr_score = 0.5*ndice(target, fr_read) + 0.5*fnw(target, fr_read) if fr_read else 0.0

    return {
        "target_ipa": target,
        "en_reading": en_read,
        "fr_reading": fr_read,
        "en_ear_score": round(en_score, 3),
        "fr_ear_score": round(fr_score, 3),
        "cross_min": round(min(en_score, fr_score), 3),
        "cross_mean": round((en_score + fr_score) / 2, 3),
    }

# ═══════════════════════════════════════════════════════════════════
# JOKER JUDGING (strict)
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
    """Strict JOKER judging. reward = sound × meaning. No heuristic. No loosening."""
    s = prosodic_score(en, fr.replace("'"," "))
    m = semcos(en, fr)
    return s*m, s, m

# ═══════════════════════════════════════════════════════════════════
# UNIFIED DATABASE (ladder + chain + strict_gold — NO cognates)
# ═══════════════════════════════════════════════════════════════════
_DB=None
def load_db(b="."):
    global _DB
    if _DB is not None: return _DB
    db={
        "strict_gold":defaultdict(list),
        "v7_gold":defaultdict(list),
        "dual":defaultdict(list),
        "ladder":defaultdict(list),
        "glue":defaultdict(list),
        "chain":defaultdict(lambda:defaultdict(list)),  # en→fr→[(hops,quality)]
        "fr_class":{}, "en_class":{},
        "syn_en":defaultdict(set), "syn_fr":defaultdict(set),
        "trans":defaultdict(set), "bridge":defaultdict(list),
        "fr_vocab":set(),
        "fr_idx":{},"fr_units":[],"fr_bylen":None,"unit_bylen":None,
    }

    # ── strict-gold ──
    try:
        for i,line in enumerate(open(f"{b}/strict-gold.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=2: db["strict_gold"][p[0]].append((p[1],1.0,0.9))
    except FileNotFoundError: pass

    # ── v7 gold ──
    try:
        for i,line in enumerate(open(f"{b}/dictionary-v7.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=9 and p[3]=="1":  # gold column
                try: db["v7_gold"][p[7]].append((p[8],float(p[1]),1.0,p[0]))
                except: pass
    except FileNotFoundError: pass

    # ── tier-ladder (gold homophone one-for-ones) ──
    try:
        for i,line in enumerate(open(f"{b}/tier-ladder.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=12 and p[10]:
                try:
                    db["ladder"][p[1]].append((float(p[10]),float(p[11]) if p[11] else 0.5,p[2]))
                except: continue
    except FileNotFoundError: pass

    # ── chain-web (transitive chains) ──
    try:
        for i,line in enumerate(open(f"{b}/chain-web-full-v7u.tsv",encoding="utf-8")):
            if i==0: continue  # skip header
            p=line.rstrip("\n").split("\t")
            if len(p)>=5:
                a,b,hops,q,sub=p[0],p[1],int(p[2]),float(p[3]),p[4]
                if ":" in a and ":" in b:
                    src_lang,src_word = a.split(":",1)
                    tgt_lang,tgt_word = b.split(":",1)
                    if src_lang=="en" and tgt_lang=="fr":
                        db["chain"][src_word][tgt_word].append((hops,q,sub))
                    elif src_lang=="fr" and tgt_lang=="en":
                        db["chain"][tgt_word][src_word].append((hops,q,sub))
    except FileNotFoundError: pass

    # ── dual-pairs ──
    for fp in [f"{b}/dual-pairs.tsv","dual-pairs.tsv"]:
        try:
            for i,line in enumerate(open(fp,encoding="utf-8")):
                if i==0: continue
                p=line.rstrip("\n").split("\t")
                if len(p)>=6: db["dual"][p[0]].append((float(p[2]),p[1]))
            break
        except: continue

    # ── zipf-glue ──
    try:
        for i,line in enumerate(open(f"{b}/zipf-glue.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=3: db["glue"][p[0]].append((float(p[2]),p[1]))
    except FileNotFoundError: pass

    # ── homophone classes ──
    for path,target in [("fr-homophone-classes-lexique.tsv","fr_class"),
                        ("fr-homophone-classes.tsv","fr_class"),
                        ("en-homophone-classes.tsv","en_class")]:
        try:
            for i,line in enumerate(open(f"{b}/{path}",encoding="utf-8")):
                if i==0: continue
                ms=line.rstrip("\n").split("\t")[1].split()
                for m in ms: db[target][m]=ms
        except FileNotFoundError: pass

    # ── synonyms ──
    try:
        for line in open(f"{b}/muse-pivot-syn.tsv",encoding="utf-8"):
            a,b,_=line.rstrip("\n").split("\t")
            if a.startswith("en:") and b.startswith("en:"):
                db["syn_en"][a[3:]].add(b[3:]); db["syn_en"][b[3:]].add(a[3:])
            elif a.startswith("fr:") and b.startswith("fr:"):
                db["syn_fr"][a[3:]].add(b[3:]); db["syn_fr"][b[3:]].add(a[3:])
    except FileNotFoundError: pass

    # ── MUSE translations ──
    for mp in [f"{b}/muse-en-fr.txt","/tmp/muse-en-fr.txt"]:
        try:
            for line in open(mp,encoding="utf-8"):
                p=line.split()
                if len(p)==2: db["trans"][p[0]].add(p[1])
            break
        except FileNotFoundError: continue

    # ── bridges ──
    try:
        for i,line in enumerate(open(f"{b}/llm-bridge.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=4: db["bridge"][p[0]].append((float(p[2]),float(p[3]),p[1]))
    except FileNotFoundError: pass

    # ── babel windows ──
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

    # ── French vocab ──
    for w in db["fr_idx"]:
        if "(en)" not in w: db["fr_vocab"].add(w)
    for cls in db["fr_class"].values():
        for w in cls:
            if w: db["fr_vocab"].add(w)

    # Pre-build length indices
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
# LADDER + CHAIN CANDIDATE SELECTION (no cognates)
# ═══════════════════════════════════════════════════════════════════
def ladder_chain_candidates(en_word, db, top=12):
    """Candidates from strict_gold → v7_gold → ladder → chain-web → dual → glue."""
    results = []
    seen = set()

    def add(fr, channel, pre_s=None, pre_m=None):
        fr = fr.lower().strip(".,;:!?'\"«»")
        if fr in seen or not fr: return
        if " " not in fr and db["fr_vocab"] and fr not in db["fr_vocab"]: return
        seen.add(fr)
        if pre_s is not None and pre_m is not None:
            reward, s, m = pre_s*pre_m, pre_s, pre_m
        else:
            reward, s, m = joker_judge(en_word, fr)
        results.append((reward, s, m, fr, channel))

    # 1. STRICT GOLD (judge-verified — highest trust)
    for fr, s, m in db["strict_gold"].get(en_word, [])[:5]:
        add(fr, "S_gold", s, m)

    # 2. V7 GOLD (curated dictionary)
    for fr, s, m, tier in db["v7_gold"].get(en_word, [])[:5]:
        add(fr, f"v7_{tier}", s, m)

    # 3. LADDER (GOLD homophone one-for-ones from tier-ladder)
    for s, m, fr in db["ladder"].get(en_word, [])[:8]:
        add(fr, "ladder", s, m)

    # 4. CHAIN-WEB (transitive chains through graph)
    for fr, edges in db["chain"].get(en_word, {}).items():
        if edges:
            best = min(edges, key=lambda e: e[0])  # shortest hop
            hops, quality = best[0], best[1]
            m = semcos(en_word, fr)
            # Trust declines with hop distance
            trust = 1.0 - 0.15 * (hops - 1)
            s_est = quality * trust
            add(fr, f"chain_h{hops}", s_est, m * trust)

    # 5. DUAL (translation∧homophone pairs)
    for s, fr in db["dual"].get(en_word, [])[:6]:
        m = semcos(en_word, fr)
        add(fr, "dual", s, m)

    # 6. GLUE (function word matches)
    ART = {"the":{"le","la","les","de","des","du"},"a":{"un","une","à","et"},"an":{"un","une"}}
    for s, fr in db["glue"].get(en_word, [])[:4]:
        if en_word in ART and fr not in ART[en_word]: continue
        add(fr, "glue", s, 0.6)

    # 7. BRIDGE (Haiku-verified cross-scope)
    for s, m, fr in db["bridge"].get(en_word, [])[:4]:
        add(fr, "haiku", s, m)

    # 8. EN HOMOPHONE CLASS pivot → translation
    for sib in db["en_class"].get(en_word, [])[:5]:
        if sib==en_word: continue
        for _s, fr in db["dual"].get(sib, [])[:2]:
            m = semcos(en_word, fr)
            add(fr, f"enclass:{sib}", _s, 0.6*m if m>0 else 0.6)
        for _s, _m, fr in db["ladder"].get(sib, [])[:2]:
            add(fr, f"enclass:{sib}", _s, 0.5*_m)

    # 9. FR HOMOPHONE CLASS meaning-max on best
    if results:
        best = results[0]
        if " " not in best[3]:
            for sib in db["fr_class"].get(best[3],[])[:6]:
                if sib not in seen and (not db["fr_vocab"] or sib in db["fr_vocab"]):
                    seen.add(sib)
                    m = semcos(en_word, sib)
                    if m > best[2]:
                        results.append((best[1]*m, best[1], m, sib, "frclass"))

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
# HILL-CLIMB (through FR homophone classes)
# ═══════════════════════════════════════════════════════════════════
def climb(en, fr, db, passes=3):
    r,_,_ = joker_judge(en, fr)
    best_r, best_m = r, semcos(en, fr)
    words = fr.split()
    for _ in range(passes):
        improved = False
        for i,w in enumerate(words):
            key = w.strip(",.;:!?'\"«»").lower()
            for alt in db["fr_class"].get(key,[])[:6]:
                if alt==key: continue
                cand = " ".join(words[:i]+[alt]+words[i+1:])
                r2,_,_ = joker_judge(en, cand)
                if r2<=best_r: continue
                m2 = semcos(en, cand)
                if m2>=best_m-0.05:
                    words[i]=alt; best_r,best_m=r2,m2; improved=True; break
            if improved: break
        if not improved: break
    return " ".join(words), best_r

# ═══════════════════════════════════════════════════════════════════
# WRITE — main pipeline
# ═══════════════════════════════════════════════════════════════════
def write(line, db=None, verbose=True):
    if db is None: db = load_db()
    ws_raw = [w.lower().strip(".,;:!?'\"") for w in line.split() if w.strip(".,;:!?'\"")]
    if not ws_raw: return None

    decomposed = [(w, decompose(w)[1], decompose(w)[2]) for w in ws_raw]

    # ── 1. LADDER+CHAIN CANDIDATES ──
    picks = []
    for orig, stem, suffix in decomposed:
        cands = ladder_chain_candidates(stem, db, top=8)
        if not cands and stem != orig:
            cands = ladder_chain_candidates(orig, db, top=8)
        if cands:
            _, s, m, fr, ch = cands[0]
            picks.append((orig, fr, s, m, ch))
        else:
            picks.append((orig, orig, 0.0, 0.0, "miss"))
    raw = " ".join(fr for _,fr,_,_,_ in picks)

    if verbose:
        print(f"Phase 1 (ladder+chain, {len(picks)} words):")
        for w,fr,s,m,ch in picks:
            gold_mark = " ★" if ch in ("S_gold","v7_S","v7_A") else ""
            chain_mark = " ◈" if ch.startswith("chain") else ""
            print(f"  {w:15s} → {fr:18s} [{ch:12s}; r={s*m:.3f} s={s:.2f} m={m:.2f}]{gold_mark}{chain_mark}")

    # ── 2. BABEL WINDOW MERGES ──
    merged = list(picks)
    i=0
    while i < len(ws_raw)-1:
        for span in (3,2):
            if i+span>len(ws_raw): continue
            gram=" ".join(ws_raw[i:i+span])
            hits=babel_many(gram,db,top=3)
            if hits and hits[0][0]>=0.72:
                fr_unit=hits[0][1].split("〔")[0]
                r,s,m=joker_judge(gram, fr_unit)
                merged[i]=(gram,fr_unit,s,m,"babel_m1")
                for j in range(i+1,i+span): merged[j]=None
                i+=span; break
        else: i+=1
    picks=[p for p in merged if p is not None]
    merged_fr=" ".join(fr for _,fr,_,_,_ in picks)

    if verbose and merged_fr!=raw:
        print(f"\nPhase 2 (babel window):")
        for w,fr,s,m,ch in picks:
            print(f"  {w:15s} → {fr:18s} [{ch:12s}; r={s*m:.3f}]")

    # ── 3. HILL-CLIMB ──
    climbed, _ = climb(line, merged_fr, db)
    if verbose and climbed!=merged_fr:
        print(f"\nPhase 3 (climb): {merged_fr} → {climbed}")

    # ── 4. JOKER JUDGE ──
    reward, s_sound, s_sem = joker_judge(line, climbed)
    # ── 5. CROSS-ACCENT TEST ──
    ca = cross_accent_test(line, climbed)

    if verbose:
        print(f"\nPhase 4-5 (JOKER + cross-accent):")
        print(f"  JOKER reward = {s_sound:.3f} × {s_sem:.3f} = {reward:.3f}")
        print(f"  EN ear hears : [{ca['en_reading']}] → match {ca['en_ear_score']:.3f}")
        print(f"  FR ear hears : [{ca['fr_reading']}] → match {ca['fr_ear_score']:.3f}")
        print(f"  cross-mean: {ca['cross_mean']:.3f}  cross-min: {ca['cross_min']:.3f}")
        print(f"  target IPA  : [{ca['target_ipa']}]")

    in_rooten = s_sound >= 0.55 and s_sem >= 0.45
    band = "✓ ROOTEN" if in_rooten else ("~ EDGE" if reward >= 0.20 else "  below")

    if verbose:
        print(f"\n{'='*60}")
        print(f"EN  : {line}")
        print(f"FR  : {climbed}")
        print(f"BAND: {band}  reward={reward:.3f}")

    return {"en":line,"fr":climbed,"reward":reward,"sound":s_sound,
            "semantic":s_sem,"cross_mean":ca["cross_mean"],"rooten":in_rooten,
            "en_ipa":ca["en_reading"],"fr_ipa":ca["fr_reading"],"target_ipa":ca["target_ipa"]}

# ═══════════════════════════════════════════════════════════════════
# SELF-IMPROVE — three iterations, never loosen judge
# ═══════════════════════════════════════════════════════════════════
def self_improve(line, db, iterations=3, verbose=True):
    """Run write(), then try to improve the weakest word. Repeat ×3."""
    if verbose: print(f"\n{'#'*60}\n# SELF-IMPROVE: '{line}'\n{'#'*60}")

    best_result = write(line, db, verbose=verbose)
    if not best_result: return None

    for it in range(1, iterations):
        if verbose: print(f"\n--- ITERATION {it+1}/{iterations} ---")
        # Find weakest word by per-word JOKER score
        words = best_result["fr"].split()
        ws_en = [w.lower().strip(".,;:!?'\"") for w in line.split() if w.strip(".,;:!?'\"")]
        if len(words) != len(ws_en):
            # Word count mismatch after babel merge — can't per-word judge
            break

        # Score each word pair
        word_scores = []
        for en_w, fr_w in zip(ws_en, words):
            r, s, m = joker_judge(en_w, fr_w)
            word_scores.append((r, s, m, en_w, fr_w))

        # Find the weakest (lowest reward)
        weakest = min(word_scores, key=lambda x: x[0])
        if verbose:
            for r,s,m,ew,fw in word_scores:
                mark = " ← WEAKEST" if (ew,fw)==(weakest[3],weakest[4]) else ""
                print(f"  {ew:15s} → {fw:15s}  r={r:.3f} s={s:.2f} m={m:.2f}{mark}")

        # Try alternatives for the weakest word
        alt_cands = ladder_chain_candidates(weakest[3], db, top=8)
        improved = False
        for _, s, m, alt_fr, ch in alt_cands[1:]:  # skip the current pick
            new_words = [fw if (ew,fw)!=(weakest[3],weakest[4]) else alt_fr
                        for ew,fw in zip(ws_en, words)]
            new_fr = " ".join(new_words)
            new_r, new_s, new_m = joker_judge(line, new_fr)
            if new_r > best_result["reward"]:
                best_result["fr"] = new_fr
                best_result["reward"] = new_r
                best_result["sound"] = new_s
                best_result["semantic"] = new_m
                ca = cross_accent_test(line, new_fr)
                best_result["cross_mean"] = ca["cross_mean"]
                best_result["en_ipa"] = ca["en_reading"]
                best_result["fr_ipa"] = ca["fr_reading"]
                best_result["target_ipa"] = ca["target_ipa"]
                improved = True
                if verbose:
                    print(f"  IMPROVE: {weakest[4]} → {alt_fr} [{ch}; r={new_r:.3f} (+{new_r-best_result['reward']:.3f})]")
                break

        if improved:
            # Re-climb after improvement
            climbed, _ = climb(line, best_result["fr"], db)
            new_r, new_s, new_m = joker_judge(line, climbed)
            if new_r > best_result["reward"]:
                best_result["fr"] = climbed
                best_result["reward"] = new_r
                best_result["sound"] = new_s
                best_result["semantic"] = new_m
                ca = cross_accent_test(line, climbed)
                best_result["cross_mean"] = ca["cross_mean"]
                best_result["en_ipa"] = ca["en_reading"]
                best_result["fr_ipa"] = ca["fr_reading"]
                if verbose:
                    print(f"  RECLIMB: → {climbed} [r={new_r:.3f}]")
        else:
            if verbose: print(f"  No improvement found for weakest word.")
            if it >= 2: break  # Stop if can't improve after 2 attempts

    # Final cross-accent
    ca = cross_accent_test(line, best_result["fr"])
    best_result["cross_mean"] = ca["cross_mean"]
    best_result["en_ipa"] = ca["en_reading"]
    best_result["fr_ipa"] = ca["fr_reading"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL (after {iterations} iterations):")
        print(f"  EN: {line}")
        print(f"  FR: {best_result['fr']}")
        print(f"  JOKER: {best_result['sound']:.3f} × {best_result['semantic']:.3f} = {best_result['reward']:.3f}")
        print(f"  EN-ear: {ca['en_ear_score']:.3f}  FR-ear: {ca['fr_ear_score']:.3f}  cross-mean: {ca['cross_mean']:.3f}")
        print(f"  EN-hears: [{ca['en_reading']}]")
        print(f"  FR-hears: [{ca['fr_reading']}]")
        print(f"  TARGET:   [{ca['target_ipa']}]")
        in_rooten = best_result["sound"]>=0.55 and best_result["semantic"]>=0.45
        print(f"  BAND: {'✓ ROOTEN' if in_rooten else '~ EDGE'}")

    return best_result

# ═══════════════════════════════════════════════════════════════════
# BENCHMARK
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
        if iters>1:
            r=self_improve(line,db,iters,verbose=True)
        else:
            r=write(line,db,verbose=True)
        if r: results.append(r)
    if results:
        rewards=[r["reward"] for r in results]
        sounds=[r["sound"] for r in results]
        sems=[r["semantic"] for r in results]
        crosses=[r.get("cross_mean",0) for r in results]
        rooten=sum(1 for r in results if r["rooten"])
        print(f"\n{'='*60}")
        print(f"BENCH ({len(results)} lines, {iters} iter):")
        print(f"  reward:  {np.mean(rewards):.3f}±{np.std(rewards):.3f}")
        print(f"  sound:   {np.mean(sounds):.3f}  semantic: {np.mean(sems):.3f}")
        print(f"  cross:   {np.mean(crosses):.3f}")
        print(f"  Rooten:  {rooten}/{len(results)} ({100*rooten/len(results):.0f}%)")
        for r in sorted(results,key=lambda x:-x["reward"])[:3]:
            print(f"  {r['reward']:.3f}  {r['en']} → {r['fr']}")

def main():
    ap=argparse.ArgumentParser(description="Homophone Writer v4 — ladder+chain, cross-accent, self-improve")
    ap.add_argument("text",nargs="*")
    ap.add_argument("--bench",type=int,default=0)
    ap.add_argument("--iter",type=int,default=1,help="Self-improve iterations (default: 1, max: 3)")
    ap.add_argument("--db-only",action="store_true")
    ap.add_argument("--base-dir",default=".")
    args=ap.parse_args()
    os.chdir(args.base_dir)
    db=load_db(args.base_dir)
    print(f"DB: {sum(len(v) for v in db['dual'].values())} dual, "
          f"{len(db['strict_gold'])} strict-gold, {len(db['v7_gold'])} v7-gold, "
          f"{len(db['ladder'])} ladder, {sum(len(v) for v in db['chain'].values())} chain, "
          f"{len(db['fr_vocab'])} vocab, {len(db['fr_idx'])} babel")
    print(f"     NO COGNATES USED\n")
    if args.db_only: return
    iters = min(args.iter, 3)
    if args.bench: bench(args.bench, iters, db); return
    if args.text:
        for t in args.text:
            if iters>1: self_improve(t, db, iters, verbose=True)
            else: write(t, db, verbose=True)
    else: bench(3, iters, db)

if __name__=="__main__": main()
