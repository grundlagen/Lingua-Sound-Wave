#!/usr/bin/env python3
"""
HOMOPHONE WRITER v3 — JOKER judging, gold-first routing, syllable-chain matching.

KEY CHANGES FROM v2:
  1. JOKER JUDGING: reward = prosodic_score × semantic_cosine (product, not blend)
     — from JOKER_METHODS.md: "combined phonetic-semantic embeddings"
  2. GOLD-FIRST: strict-gold + v7 dict consulted before any heuristic
     — 1,314 strict-gold + 2,070 v7 pairs = exact matches available
  3. SYLLABLE CHAIN: decompose IPA into syllables, match per-vowel-span,
     chain FR syllables together — syllable-to-syllable, not word-to-word
  4. CLEANER ROUTING: gold → cognate → dual → ladder → syllable-chain → window → glue
  5. PROPER SCORING: reward drives selection, not heuristic rank

Run: python homophone_writer_v3.py "the sea remembers every ship"
     python homophone_writer_v3.py --bench 12
"""

from __future__ import annotations

import argparse, os, re, subprocess, sys, unicodedata
from collections import defaultdict

import numpy as np
import panphon

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS (same as v2)
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
for group, cost in _EQUIV_RAW:
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            EQUIV[tuple(sorted((group[i], group[j])))] = min(cost, EQUIV.get(tuple(sorted((group[i], group[j]))), 1.0))

CHEAP_GAP = {"h":.08,"ə":.12,"ɚ":.12,"ʔ":.08,"ʲ":.08,"ʷ":.08,"j":.18,"w":.18}

# ═══════════════════════════════════════════════════════════════════
# G2P + SEGMENTS (cached)
# ═══════════════════════════════════════════════════════════════════
_ECACHE = {}
def _espeak(text, lang):
    k = (text, lang)
    if k in _ECACHE: return _ECACHE[k]
    try:
        r = subprocess.run(["espeak-ng","-q","--ipa","-v",
            {"en":"en-us","fr":"fr"}.get(lang,lang), text],
            capture_output=True, text=True, check=True)
        v = unicodedata.normalize("NFD", r.stdout.strip())
    except: v = ""
    _ECACHE[k] = v; return v

def g2p(text, lang):
    r = _espeak(text, lang)
    for c in "ˈˌ .‿|‖ˑː": r = r.replace(c,"")
    return r

def _segs(ipa):
    s, i, n = [], 0, len(ipa)
    while i < n:
        if i+1<n and ipa[i:i+2] in {"ɑ̃","ɛ̃","ɔ̃","œ̃","t͡ʃ","d͡ʒ","t͡s"}:
            s.append(ipa[i:i+2]); i+=2
        else: s.append(ipa[i]); i+=1
    return tuple(s)

_FCACHE = {}
def _feat(seg):
    if seg in _FCACHE: return _FCACHE[seg]
    try:
        v = FT.word_to_vector_list(seg, numeric=True)
        a = np.clip(np.array(v,dtype=np.float32),-1,1) if v else np.zeros(N_FEATURES,dtype=np.float32)
        if a.ndim==2: a=a[0]
        if a.shape[0]!=N_FEATURES:
            p=np.zeros(N_FEATURES,dtype=np.float32); p[:min(a.shape[0],N_FEATURES)]=a[:min(a.shape[0],N_FEATURES)]; a=p
    except: a=np.zeros(N_FEATURES,dtype=np.float32)
    _FCACHE[seg]=a; return a

def _ecost(a,b):
    if a==b: return 0.0
    f = EQUIV.get(tuple(sorted((a,b))), 0.90)
    d = min(1.0, float(np.tanh(np.sqrt(np.sum((_feat(a)-_feat(b))**2))/np.sqrt(N_FEATURES)/SHARPEN)))
    return min(f,d)

# ═══════════════════════════════════════════════════════════════════
# PHONETIC CHANNELS
# ═══════════════════════════════════════════════════════════════════
def ndice(ia, ib, n=2):
    def ng(s): return {s[i:i+n] for i in range(len(s)-n+1)} if len(s)>=n else {s}
    A,B=ng(ia),ng(ib); return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

def fnw(ia, ib):
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
    return max(0.0, 1.0-D[lx,ly]/max(lx,ly))

def combo(en,fr):
    qi,ci=g2p(en,"en"),g2p(fr,"fr")
    return 0.5*ndice(qi,ci)+0.5*fnw(qi,ci) if qi and ci else 0.0

# ═══════════════════════════════════════════════════════════════════
# JOKER JUDGING: reward = sound × meaning (product)
# ═══════════════════════════════════════════════════════════════════
def prosodic_score(en,fr):
    """Stress-weighted alignment + EN/FR diverged prosody."""
    def _stressed(text, lang):
        s_all,w_all=[],[]
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
            s_all+=segs; w_all+=sw
        return s_all,w_all

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
            _SEM = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except: _SEM=False
    if _SEM is False: return 0.5
    try:
        v=_SEM.encode([en,fr],normalize_embeddings=True); return float(v[0]@v[1])
    except: return 0.5

def joker_reward(en, fr):
    """JOKER-style: reward = sound × meaning (product, not blend)."""
    s = prosodic_score(en, fr.replace("'"," "))
    m = semcos(en, fr)
    return s * m, s, m

# ═══════════════════════════════════════════════════════════════════
# SYLLABLE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════
def syllables_from_ipa(ipa):
    """Split IPA string into syllable-like spans centered on vowels."""
    segs = _segs(ipa)
    vowel_positions = [i for i,s in enumerate(segs) if any(c in VOWELS for c in s)]
    if not vowel_positions:
        return [("".join(segs),)]
    syllables = []
    for vi, vpos in enumerate(vowel_positions):
        start = vowel_positions[vi-1]+1 if vi>0 else 0
        end = vowel_positions[vi+1] if vi+1<len(vowel_positions) else len(segs)
        syl = "".join(segs[start:end])
        if syl: syllables.append(syl)
    return tuple(syllables) if syllables else ("".join(segs),)

# ═══════════════════════════════════════════════════════════════════
# UNIFIED DATABASE
# ═══════════════════════════════════════════════════════════════════
_DB = None
def load_db(b="."):
    global _DB
    if _DB is not None: return _DB
    db = {
        "dual":defaultdict(list),"ladder":defaultdict(list),
        "glue":defaultdict(list),"fr_class":{},"en_class":{},
        "syn_en":defaultdict(set),"syn_fr":defaultdict(set),
        "trans":defaultdict(set),"bridge":defaultdict(list),
        "fr_vocab":set(),"cognates":{},
        "strict_gold":defaultdict(list),  # en → [(fr, sound, meaning), ...]
        "v7_gold":defaultdict(list),      # en → [(fr, sound, meaning, tier), ...]
        "fr_idx":{},"fr_units":[],"fr_bylen":None,"unit_bylen":None,
        "en_idx":{},                        # for FR→EN mirror
    }

    # ── strict-gold.tsv: judge-verified exact matches ──
    try:
        for i,line in enumerate(open(f"{b}/strict-gold.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=2:
                db["strict_gold"][p[0]].append((p[1],1.0,0.9))
    except FileNotFoundError: pass

    # ── dictionary-v7.tsv: curated homophone pairs (tier S/A/B) ──
    try:
        for i,line in enumerate(open(f"{b}/dictionary-v7.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=8 and p[3]=="1":   # gold column
                try: db["v7_gold"][p[6]].append((p[7],float(p[1]),1.0,p[0]))
                except: pass
    except FileNotFoundError: pass

    # ── dual-pairs ──
    try:
        for i,line in enumerate(open(f"{b}/dual-pairs.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=6: db["dual"][p[0]].append((float(p[2]),p[1]))
    except FileNotFoundError: pass

    # ── tier-ladder ──
    try:
        for i,line in enumerate(open(f"{b}/tier-ladder.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=12 and p[10]:
                try: db["ladder"][p[1]].append((float(p[10]),float(p[11]) if p[11] else 0.5,p[2]))
                except: continue
    except FileNotFoundError: pass

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

    # ── translations (MUSE) ──
    for mp in [f"{b}/muse-en-fr.txt","/tmp/muse-en-fr.txt"]:
        try:
            for line in open(mp,encoding="utf-8"):
                p=line.split();
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
    for fp in [f"{b}/en-word-ipa.tsv","en-word-ipa.tsv"]:
        try:
            for i,line in enumerate(open(fp,encoding="utf-8")):
                if i==0: continue
                p=line.rstrip("\n").split("\t")
                if len(p)>=2 and p[1]: db["en_idx"][p[0]]=p[1]
            break
        except: continue

    # ── French vocab ──
    for w in db["fr_idx"]:
        if "(en)" not in w: db["fr_vocab"].add(w)
    for cls in db["fr_class"].values():
        for w in cls:
            if w: db["fr_vocab"].add(w)

    # ── Cognates ──
    try:
        for i,line in enumerate(open(f"{b}/dictionary-v7.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=8 and p[5]=="1":
                en_w,fr_w=p[6],p[7]
                if not db["fr_vocab"] or fr_w in db["fr_vocab"]:
                    db["cognates"][en_w]=fr_w
    except: pass
    for en_w in list(db["dual"].keys()):
        for _s,fr_w in db["dual"][en_w]:
            if en_w.lower()==fr_w.lower():
                if not db["fr_vocab"] or fr_w.lower() in db["fr_vocab"]:
                    db["cognates"][en_w.lower()]=fr_w.lower()
                break
    COMMON = {"nation":"nation","nature":"nature","table":"table","art":"art",
              "centre":"centre","culture":"culture","moment":"moment","silence":"silence",
              "dance":"danse","chance":"chance","restaurant":"restaurant","important":"important"}
    for en,fr in COMMON.items():
        if not db["fr_vocab"] or fr in db["fr_vocab"]: db["cognates"][en]=fr

    # ── Pre-build length indices ──
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
# SYLLABLE CHAIN MATCHING
# ═══════════════════════════════════════════════════════════════════
def syllable_match(syllable_ipa, fr_bylen, top=5, tol=1):
    """Match one syllable IPA against FR words of similar length."""
    n = len(_segs(syllable_ipa))
    cands = []
    for L in range(max(1,n-tol), n+tol+1):
        cands.extend(fr_bylen.get(L,[]))
    if len(cands) > 1500: cands = cands[:1500]  # cap
    scored = []
    for w,p in cands:
        s = 0.5*ndice(syllable_ipa,p) + 0.5*fnw(syllable_ipa,p)
        if s >= 0.50: scored.append((s,w))
    scored.sort(reverse=True)
    return scored[:top]

def syllable_chain(en_word, db, top=3):
    """Decompose EN word IPA into syllables, match each, chain FR words."""
    ipa = g2p(en_word, "en")
    if not ipa: return []
    syls = syllables_from_ipa(ipa)
    if len(syls) == 1:
        # Single syllable → direct match
        return [(s, w) for s,w in syllable_match(syls[0], db["fr_bylen"], top=top)]

    # Multi-syllable: chain best matches
    chains = []
    for syl in syls:
        matches = syllable_match(syl, db["fr_bylen"], top=3)
        if matches:
            chains.append(matches)
        else:
            chains.append([(0.0, "?")])

    # Combine: pick best combination across all syllables
    # Simple approach: pick best from each syllable and join
    best = []
    for i, matches in enumerate(chains):
        best.append(matches[0])
    combined = " ".join(w for _,w in best)
    avg_s = sum(s for s,_ in best) / len(best)
    return [(avg_s, combined)]

# ═══════════════════════════════════════════════════════════════════
# GOLD-FIRST CANDIDATE SELECTION
# ═══════════════════════════════════════════════════════════════════
def gold_candidates(en_word, db, top=12):
    """JOKER-judged candidates: gold first, then heuristic channels."""
    results = []
    seen = set()

    def add(fr, channel, pre_score=None, pre_meaning=None):
        fr = fr.lower().strip(".,;:!?'\"«»")
        if fr in seen or not fr: return
        if " " not in fr and db["fr_vocab"] and fr not in db["fr_vocab"]: return
        seen.add(fr)
        if pre_score is not None and pre_meaning is not None:
            reward = pre_score * pre_meaning
            s, m = pre_score, pre_meaning
        else:
            reward, s, m = joker_reward(en_word, fr)
        results.append((reward, s, m, fr, channel))

    # ── 1. STRICT GOLD ──
    for fr, s, m in db["strict_gold"].get(en_word, [])[:5]:
        add(fr, "strict_gold", s, m)

    # ── 2. V7 GOLD ──
    for fr, s, m, tier in db["v7_gold"].get(en_word, [])[:5]:
        add(fr, f"v7_{tier}", s, m)

    # ── 3. COGNATE ──
    if en_word.lower() in db["cognates"]:
        add(db["cognates"][en_word.lower()], "cognate", 1.0, 1.0)

    # ── 4. DUAL ──
    for s, fr in db["dual"].get(en_word, [])[:6]:
        m = semcos(en_word, fr)
        add(fr, "dual", s, m)

    # ── 5. LADDER ──
    for s, m, fr in db["ladder"].get(en_word, [])[:6]:
        add(fr, "ladder", s, m)

    # ── 6. SYLLABLE CHAIN ──
    for s, fr_chain in syllable_chain(en_word, db, top=3):
        if "?" not in fr_chain:
            m = semcos(en_word, fr_chain)
            add(fr_chain, "syl_chain", s, m)

    # ── 7. GLUE ──
    ART = {"the":{"le","la","les","de","des","du"},"a":{"un","une","à","et"},"an":{"un","une"}}
    for s, fr in db["glue"].get(en_word, [])[:4]:
        if en_word in ART and fr not in ART[en_word]: continue
        m = 0.6
        add(fr, "glue", s, m)

    # ── 8. BRIDGE ──
    for s, m, fr in db["bridge"].get(en_word, [])[:4]:
        add(fr, "haiku", s, m)

    # ── 9. EN HOMOPHONE CLASS pivot ──
    for sib in db["en_class"].get(en_word, [])[:5]:
        if sib==en_word: continue
        for _s, fr in db["dual"].get(sib, [])[:2]:
            m = semcos(en_word, fr)
            add(fr, f"enclass:{sib}", _s, 0.6)
        for _s, _m, fr in db["ladder"].get(sib, [])[:2]:
            add(fr, f"enclass:{sib}", _s, 0.5*_m)

    # ── 10. FR HOMOPHONE CLASS meaning-max on best ──
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
# BABEL WINDOWS (same as v2, capped)
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
    if wl in _MANUAL: return _MANUAL[wl], w[len(_MANUAL[wl]):]
    for suf,rep in _SUFFIXES:
        if wl.endswith(suf) and len(wl)-len(suf)>=3:
            return wl[:len(wl)-len(suf)]+rep, suf
    return wl, ""

# ═══════════════════════════════════════════════════════════════════
# HILL-CLIMB
# ═══════════════════════════════════════════════════════════════════
def climb(en, fr, db, passes=4):
    r,_,_ = joker_reward(en, fr)
    best_r, best_m = r, semcos(en, fr)
    words = fr.split()
    for _ in range(passes):
        improved = False
        for i,w in enumerate(words):
            key = w.strip(",.;:!?'\"«»").lower()
            for alt in db["fr_class"].get(key,[])[:6]:
                if alt==key: continue
                cand = " ".join(words[:i]+[alt]+words[i+1:])
                r2,_,_ = joker_reward(en, cand)
                if r2<=best_r: continue
                m2 = semcos(en, cand)
                if m2>=best_m-0.05:
                    words[i]=alt; best_r,best_m=r2,m2; improved=True; break
            if improved: break
        if not improved: break
    return " ".join(words), best_r

# ═══════════════════════════════════════════════════════════════════
# AUTO-IMPROVE
# ═══════════════════════════════════════════════════════════════════
def improve(en, picks, db, threshold=0.30, max_tries=15):
    words = [fr for _,fr,_,_,_ in picks]
    best_r,_,_ = joker_reward(en, " ".join(words))
    best_picks = list(picks)
    tries = 0
    for pos in range(len(picks)):
        w = picks[pos][0]
        alts = gold_candidates(w, db, top=6)
        for _, s, m, fr, ch in alts[1:]:
            if tries>=max_tries: break
            nw = [p[1] for p in best_picks]; nw[pos]=fr
            r2,_,_ = joker_reward(en, " ".join(nw))
            if r2>best_r:
                best_r=r2; best_picks[pos]=(w,fr,s,m,ch); tries+=1
                if best_r>=threshold: break
        if best_r>=threshold: break
    return best_picks, best_r

# ═══════════════════════════════════════════════════════════════════
# WRITE — main pipeline
# ═══════════════════════════════════════════════════════════════════
def write(line, db=None, verbose=True, improve_threshold=0.30):
    if db is None: db = load_db()
    ws_raw = [w.lower().strip(".,;:!?'\"") for w in line.split() if w.strip(".,;:!?'\"")]
    if not ws_raw: return None

    # ── 0. DECOMPOSE ──
    decomposed = [(w,) + decompose(w) for w in ws_raw]

    # ── 1. GOLD-FIRST CANDIDATES ──
    picks = []
    for orig, stem, suffix in decomposed:
        cands = gold_candidates(stem, db, top=8)
        if not cands and stem != orig:
            cands = gold_candidates(orig, db, top=8)
        if cands:
            _, s, m, fr, ch = cands[0]
            picks.append((orig, fr, s, m, ch))
        else:
            picks.append((orig, orig, 0.0, 0.0, "miss"))
    raw = " ".join(fr for _,fr,_,_,_ in picks)

    if verbose:
        print(f"Phase 1 (gold-first, {len(picks)} words):")
        for w,fr,s,m,ch in picks:
            mark = " ★" if ch in ("strict_gold","v7_S","v7_A","cognate") else ""
            print(f"  {w:15s} → {fr:18s} [{ch:12s}; r={s*m:.3f} s={s:.2f} m={m:.2f}]{mark}")

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
                r,s,m=joker_reward(gram, fr_unit)
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
    reward, s_sound, s_sem = joker_reward(line, climbed)
    if verbose:
        print(f"\nPhase 4 (JOKER judge):")
        print(f"  sound={s_sound:.3f}  meaning={s_sem:.3f}  reward={reward:.3f}")

    # ── 5. AUTO-IMPROVE ──
    best_fr, best_r = climbed, reward
    if best_r < improve_threshold:
        imp_picks, imp_r = improve(line, picks, db, improve_threshold)
        if imp_r > best_r:
            best_fr = " ".join(fr for _,fr,_,_,_ in imp_picks)
            best_r = imp_r
            if verbose:
                print(f"\nPhase 5 (improve): reward {reward:.3f} → {best_r:.3f}")

    # ── Final ──
    in_rooten = s_sound >= 0.55 and s_sem >= 0.45
    band = "✓ ROOTEN" if in_rooten else ("~ EDGE" if best_r >= 0.20 else "  below")
    if verbose:
        print(f"\n{'='*60}")
        print(f"EN  : {line}")
        print(f"FR  : {best_fr}")
        print(f"BAND: {band}  reward={best_r:.3f}  (sound={s_sound:.3f} × meaning={s_sem:.3f})")

    return {"en":line,"fr":best_fr,"reward":best_r,"sound":s_sound,"semantic":s_sem,"rooten":in_rooten}

# ═══════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════
CORPUS = [
    "we see the moon at dawn","mary had a little lamb",
    "the sea remembers every ship","we call to the moon and she answers",
    "less debt less mess more soup","my sorrow sleeps in a deep well",
    "bless the dawn that made us free","the cat sat on the mat",
    "i love you more than words can say","she walks in beauty like the night",
    "a rose by any other name would smell as sweet","to be or not to be that is the question",
]

def bench(n=6):
    db=load_db()
    results=[]
    for line in CORPUS[:n]:
        print(f"\n{'─'*60}")
        r=write(line,db,verbose=True)
        if r: results.append(r)
    if results:
        rewards=[r["reward"] for r in results]
        sounds=[r["sound"] for r in results]
        sems=[r["semantic"] for r in results]
        rooten=sum(1 for r in results if r["rooten"])
        print(f"\n{'='*60}")
        print(f"BENCH ({len(results)} lines): reward={np.mean(rewards):.3f}±{np.std(rewards):.3f}")
        print(f"  sound={np.mean(sounds):.3f}  semantic={np.mean(sems):.3f}")
        print(f"  Rooten band: {rooten}/{len(results)} ({100*rooten/len(results):.0f}%)")
        for r in sorted(results,key=lambda x:-x["reward"])[:3]:
            print(f"  {r['reward']:.3f}  {r['en']} → {r['fr']}")

def main():
    ap=argparse.ArgumentParser(description="Homophone Writer v3 — JOKER judging")
    ap.add_argument("text",nargs="*")
    ap.add_argument("--bench",type=int,default=0)
    ap.add_argument("--db-only",action="store_true")
    ap.add_argument("--base-dir",default=".")
    args=ap.parse_args()
    os.chdir(args.base_dir)
    db=load_db(args.base_dir)
    print(f"DB: {sum(len(v) for v in db['dual'].values())} dual, "
          f"{len(db['strict_gold'])} strict-gold, {len(db['v7_gold'])} v7-gold, "
          f"{len(db['cognates'])} cognates, {len(db['fr_vocab'])} vocab, "
          f"{len(db['fr_idx'])} babel\n")
    if args.db_only: return
    if args.bench: bench(args.bench); return
    if args.text:
        for t in args.text: write(t,db,verbose=True)
    else: bench(4)

if __name__=="__main__": main()
