#!/usr/bin/env python3
"""
COMPOSITION WEB v6 — Non-stop set-theory algebraic mixing.
No nursery rhymes. Pure data.

ARCHITECTURE:
  Each English word → COMPOSITION WEB (graph):
    Layer 0: Direct matches (ladder, v7-gold, strict-gold, dual)
    Layer 1: Homophone class expansion (33k FR classes: vert→verre→vers→ver→vair)
    Layer 2: Synonym expansion (44k muse-pivot-syn edges)
    Layer 3: Chain-web hops (70k transitive edges through graph)
    Each node scored: cross_accent(EN_word, FR_node) × semantic_cosine × hop_trust

  At PHRASE level: set-cover over composition webs
    Content universe U = all EN content words in phrase
    For each EN word, select best FR node
    Greedy submodular: maximize meaning_coverage(U) subject to sound budget

  At CORPUS level (200+ words):
    Build webs for all words
    Measure: what % of words have cross≥0.40? what % have multiple paths?
    Composition stats: avg web size, homophone class fanout, synonym depth

Run: python composition_web.py --corpus 200
     python composition_web.py --web "beauty"
     python composition_web.py --phrase "the silent beauty of the endless sea"
"""

from __future__ import annotations

import argparse, json, math, os, re, subprocess, sys, unicodedata
from collections import defaultdict

import numpy as np
import panphon

# ═══════════════════════════════════════════════════════════════════
FT = panphon.FeatureTable()
N_FEATURES, SHARPEN, GAP = 24, 0.35, 0.42
VOWELS = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɑɒ"

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
            {"en":"en-us","fr":"fr"}.get(lang,lang), text],
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
# CROSS-ACCENT SCORING
# ═══════════════════════════════════════════════════════════════════
def cross_accent(en_text, fr_text):
    target = g2p(en_text, "en")
    if not target: return 0.0, {}
    en_read = g2p(fr_text, "en")
    fr_read = g2p(fr_text, "fr")
    en_m = 0.5*ndice(target, en_read) + 0.5*fnw(target, en_read) if en_read else 0.0
    fr_m = 0.5*ndice(target, fr_read) + 0.5*fnw(target, fr_read) if fr_read else 0.0
    return min(en_m, fr_m), {"target":target,"en":en_read,"fr":fr_read,"en_m":en_m,"fr_m":fr_m}

# ═══════════════════════════════════════════════════════════════════
# UNIFIED DATABASE (composition-web optimized)
# ═══════════════════════════════════════════════════════════════════
_DB = None
def load_db(b="."):
    global _DB
    if _DB is not None: return _DB
    db = {
        # Layer 0: direct matches
        "ladder": defaultdict(list),     # en → [(sound, meaning, fr), ...]
        "v7_gold": defaultdict(list),    # fr → [(en, sound, meaning, tier), ...]
        "strict_gold": defaultdict(list), # en → [(fr, sound, meaning), ...]
        "dual": defaultdict(list),       # en → [(sound, fr), ...]
        # Layer 1: homophone classes
        "fr_class": {},                   # fr_word → [homophone, ...] (33k)
        # Layer 2: synonyms
        "syn_fr": defaultdict(set),       # fr_word → {synonym, ...}
        # Layer 3: chain-web
        "chain": defaultdict(lambda: defaultdict(list)),  # en→fr→[(hops,quality)]
        # Meaning back-paths
        "fr_to_en": defaultdict(set),     # fr_word → {en_word, ...} (what can this FR word mean?)
        # Vocab
        "fr_vocab": set(),
    }

    # ── tier-ladder (primary data source) ──
    try:
        for i,line in enumerate(open(f"{b}/tier-ladder.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=12 and p[10]:
                try:
                    snd = float(p[10]); mng = float(p[11]) if p[11] else 0.5
                    en_w, fr_w = p[1], p[2]
                    if snd >= 0.55:  # quality threshold
                        db["ladder"][en_w].append((snd, mng, fr_w))
                        db["fr_to_en"][fr_w].add(en_w)
                except: continue
    except FileNotFoundError: pass

    # ── v7 gold ──
    try:
        for i,line in enumerate(open(f"{b}/dictionary-v7.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=9 and p[3]=="1":
                try:
                    db["v7_gold"][p[8]].append((p[7],float(p[1]),1.0,p[0]))
                    db["fr_to_en"][p[8]].add(p[7])
                except: pass
    except FileNotFoundError: pass

    # ── strict-gold ──
    try:
        for i,line in enumerate(open(f"{b}/strict-gold.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=2:
                db["strict_gold"][p[0]].append((p[1],1.0,0.9))
                db["fr_to_en"][p[1]].add(p[0])
    except FileNotFoundError: pass

    # ── dual-pairs (identity filtered) ──
    for fp in [f"{b}/dual-pairs.tsv","dual-pairs.tsv"]:
        try:
            for i,line in enumerate(open(fp,encoding="utf-8")):
                if i==0: continue
                p=line.rstrip("\n").split("\t")
                if len(p)>=6:
                    en_w, fr_w = p[0], p[1]
                    if en_w.lower() != fr_w.lower():  # skip identity pollution
                        db["dual"][en_w].append((float(p[2]),fr_w))
                        db["fr_to_en"][fr_w].add(en_w)
            break
        except: continue

    # ── FR homophone classes (the set-theoretic heart) ──
    for path in ["fr-homophone-classes-lexique.tsv","fr-homophone-classes.tsv"]:
        try:
            for i,line in enumerate(open(f"{b}/{path}",encoding="utf-8")):
                if i==0: continue
                ms = line.rstrip("\n").split("\t")[1].split()
                for m in ms: db["fr_class"][m] = ms
        except: pass

    # ── FR synonyms ──
    try:
        for line in open(f"{b}/muse-pivot-syn.tsv",encoding="utf-8"):
            a,b,_ = line.rstrip("\n").split("\t")
            if a.startswith("fr:") and b.startswith("fr:"):
                db["syn_fr"][a[3:]].add(b[3:])
                db["syn_fr"][b[3:]].add(a[3:])
    except: pass

    # ── chain-web ──
    try:
        for i,line in enumerate(open(f"{b}/chain-web-full-v7u.tsv",encoding="utf-8")):
            if i==0: continue
            p = line.rstrip("\n").split("\t")
            if len(p)>=5:
                a,b = p[0],p[1]
                if ":" in a and ":" in b:
                    sl,sw = a.split(":",1); tl,tw = b.split(":",1)
                    if sl=="en" and tl=="fr":
                        db["chain"][sw][tw].append((int(p[2]),float(p[3])))
                        db["fr_to_en"][tw].add(sw)
                    elif sl=="fr" and tl=="en":
                        db["chain"][tw][sw].append((int(p[2]),float(p[3])))
                        db["fr_to_en"][sw].add(tw)
    except: pass

    # ── FR vocab ──
    try:
        for i,line in enumerate(open(f"{b}/fr-word-ipa.tsv",encoding="utf-8")):
            if i==0: continue
            p = line.rstrip("\n").split("\t")
            if len(p)>=2 and p[1] and "(en)" not in p[0]:
                db["fr_vocab"].add(p[0])
    except: pass
    for cls in db["fr_class"].values():
        for w in cls:
            if w: db["fr_vocab"].add(w)

    # Sort
    for k in ("dual","ladder"):
        for v in db[k].values(): v.sort(reverse=True)

    _DB = db; return db

# ═══════════════════════════════════════════════════════════════════
# COMPOSITION WEB — expand one word through all layers
# ═══════════════════════════════════════════════════════════════════
def build_composition_web(en_word, db, max_nodes=80):
    """
    Build a COMPOSITION WEB for one English word.
    
    Returns: dict with layers, nodes, edges, and cross-accent scores.
    
    Layers:
      L0: direct matches (ladder, v7, strict-gold, dual)
      L1: homophone class expansion (for each L0 FR node, expand to siblings)
      L2: synonym expansion (for each FR node, expand to synonyms)
      L3: chain-web hops (transitive paths through the graph)
    
    Every node scored by: cross_accent(EN, FR) × semantic_trust × hop_decay
    """
    web = {"en_word": en_word, "layers": {}, "all_nodes": {}, "stats": {}}
    seen_fr = set()

    def add_node(fr, layer, source, trust=1.0, pre_cross=None, pre_meaning=None):
        fr = fr.lower().strip()
        if fr in seen_fr or not fr: return None
        if db["fr_vocab"] and fr not in db["fr_vocab"]: return None
        seen_fr.add(fr)

        if pre_cross is not None:
            cross = pre_cross
        else:
            cross, _ = cross_accent(en_word, fr)

        # Meaning: what EN words can this FR word mean?
        meaning_set = db["fr_to_en"].get(fr, set())
        meaning_size = len(meaning_set)

        # Composition score: cross_accent × trust
        score = cross * trust

        node = {
            "fr": fr, "layer": layer, "source": source,
            "cross": round(cross, 3), "score": round(score, 3),
            "trust": round(trust, 3),
            "meaning_set": meaning_set,
            "meaning_size": meaning_size,
            "homophone_class": db["fr_class"].get(fr, []),
            "synonyms": db["syn_fr"].get(fr, set()),
        }
        web["all_nodes"][fr] = node
        web["layers"].setdefault(layer, []).append(node)
        return node

    # ══ LAYER 0: Direct matches ══
    # strict-gold
    for fr, s, m in db["strict_gold"].get(en_word, [])[:8]:
        add_node(fr, 0, "S_gold", 1.0, s*m)
    # v7 gold
    for en_w, s, m, tier in db["v7_gold"].get(en_word, [])[:8]:
        add_node(en_w, 0, f"v7_{tier}", 1.0, s*m)
    # ladder (top by sound)
    for s, m, fr in db["ladder"].get(en_word, [])[:12]:
        add_node(fr, 0, "ladder", 1.0)
    # dual
    for s, fr in db["dual"].get(en_word, [])[:6]:
        add_node(fr, 0, "dual", 0.9)

    # ══ LAYER 1: Homophone class expansion ══
    l0_nodes = list(web["all_nodes"].values())
    for node in l0_nodes[:20]:  # expand top 20
        for sib in node["homophone_class"][:8]:
            if sib not in seen_fr:
                add_node(sib, 1, f"homophone({node['fr']})", 0.85, pre_cross=node["cross"])

    # ══ LAYER 2: Synonym expansion ══
    l0l1_nodes = list(web["all_nodes"].values())
    for node in l0l1_nodes[:30]:
        for syn in list(node["synonyms"])[:5]:
            if syn not in seen_fr:
                add_node(syn, 2, f"synonym({node['fr']})", 0.75)

    # ══ LAYER 3: Chain-web hops ══
    for fr, edges in db["chain"].get(en_word, {}).items():
        if edges and fr not in seen_fr:
            best = min(edges, key=lambda e: e[0])
            add_node(fr, 3, f"chain_h{best[0]}", best[1] * (1.0 - 0.15*(best[0]-1)))

    # ══ Stats ══
    web["stats"] = {
        "total_nodes": len(web["all_nodes"]),
        "L0_direct": len(web["layers"].get(0, [])),
        "L1_homophone": len(web["layers"].get(1, [])),
        "L2_synonym": len(web["layers"].get(2, [])),
        "L3_chain": len(web["layers"].get(3, [])),
        "cross_above_04": sum(1 for n in web["all_nodes"].values() if n["cross"] >= 0.40),
        "cross_above_05": sum(1 for n in web["all_nodes"].values() if n["cross"] >= 0.50),
        "top_node": max(web["all_nodes"].values(), key=lambda n: n["score"], default={"fr":"?","score":0}),
    }

    return web

# ═══════════════════════════════════════════════════════════════════
# SET-THEORETIC PHRASE COMPOSITION
# ═══════════════════════════════════════════════════════════════════
STOP = set("the a an of to in on at for and or is are was be it he she we you "
           "they my his her its our your this that not but so as by with do "
           "did had has have will shall".split())

def phrase_compose(en_phrase, db, verbose=True):
    """
    Set-theoretic composition of a phrase using composition webs.
    
    1. Extract content words (skip stop words)
    2. Build composition web for each content word
    3. Greedy submodular cover: pick FR words that maximize new meaning coverage
       while maintaining sound budget
    4. Report coverage statistics
    """
    ws = [w.lower().strip(".,;:!?'\"") for w in en_phrase.split() if w.strip(".,;:!?'\"")]
    content = [w for w in ws if w not in STOP and len(w) > 2]
    if not content: content = ws

    # Content universe: all EN content words (the set to cover via meaning)
    universe = set(content)

    if verbose:
        print(f"PHRASE: {en_phrase}")
        print(f"  content words: {content}")
        print(f"  meaning universe: {sorted(universe)}")

    # Build webs
    webs = {}
    for w in content:
        webs[w] = build_composition_web(w, db)

    if verbose:
        print(f"\n  COMPOSITION WEBS:")
        for w, web in webs.items():
            top = sorted(web["all_nodes"].values(), key=lambda n: -n["score"])[:3]
            top_str = "  ".join(f"{n['fr']}({n['cross']:.2f}/{n['score']:.2f})" for n in top)
            print(f"    {w:15s} → {web['stats']['total_nodes']:3d} nodes, "
                  f"L0={web['stats']['L0_direct']} L1={web['stats']['L1_homophone']} "
                  f"L2={web['stats']['L2_synonym']} L3={web['stats']['L3_chain']}")
            print(f"      top: {top_str}")

    # Greedy submodular set-cover
    # For each EN word, pick the FR node that maximizes new meaning coverage
    covered = set()
    picks = {}
    remaining = list(content)

    # Sort remaining by web quality (words with high-cross nodes first)
    remaining.sort(key=lambda w: max((n["cross"] for n in webs[w]["all_nodes"].values()), default=0), reverse=True)

    for w in remaining:
        best_node = None
        best_gain = -1
        web_nodes = sorted(webs[w]["all_nodes"].values(), key=lambda n: -n["score"])

        for node in web_nodes[:15]:
            # Gain = new meaning words this FR node brings
            gain = len(node["meaning_set"] - covered)
            # Weighted: gain × cross_accent (sound matters)
            weighted_gain = gain * node["cross"]

            if weighted_gain > best_gain:
                best_gain = weighted_gain
                best_node = node

        if best_node:
            picks[w] = best_node
            covered |= best_node["meaning_set"]

    # Coverage statistics
    uncovered = universe - covered
    coverage = len(covered & universe) / len(universe) if universe else 0

    if verbose:
        print(f"\n  SET-COVER RESULT:")
        print(f"    covered: {len(covered & universe)}/{len(universe)} ({coverage*100:.0f}%)")
        if uncovered:
            print(f"    uncovered: {sorted(uncovered)}")
        print(f"    picks:")
        for w, node in picks.items():
            print(f"      {w:15s} → {node['fr']:18s}  "
                  f"cross={node['cross']:.2f}  brings={len(node['meaning_set'] & universe)} meanings")

    # Final phrase
    final_fr = []
    for w in ws:
        if w in picks:
            final_fr.append(picks[w]["fr"])
        elif w in STOP:
            # Don't include stop words unless they have a known homophone
            pass
        else:
            final_fr.append(f"«{w}»")

    final = " ".join(final_fr)
    if verbose:
        print(f"\n  COMPOSED: {final}")

    return {
        "phrase": en_phrase, "composed": final,
        "coverage": coverage, "picks": picks,
        "webs": webs, "uncovered": uncovered,
    }

# ═══════════════════════════════════════════════════════════════════
# CORPUS-LEVEL COMPOSITION (200+ words)
# ═══════════════════════════════════════════════════════════════════
def corpus_compose(n_words=200, db=None, verbose=True):
    """
    Build composition webs for N content words from ladder data.
    Measure cross-accent coverage, web statistics, set-theoretic properties.
    """
    if db is None: db = load_db()

    # Extract N content words with high-quality matches
    candidates = []
    for en_w, entries in db["ladder"].items():
        if len(en_w) >= 3 and en_w not in STOP:
            best_s = max(s for s,_,_ in entries)
            if best_s >= 0.60:
                candidates.append((best_s, en_w, len(entries)))
    candidates.sort(reverse=True)

    corpus = [w for _,w,_ in candidates[:n_words]]
    if verbose:
        print(f"CORPUS: {len(corpus)} words selected from ladder (sound ≥ 0.60)")
        print(f"  sample: {', '.join(corpus[:30])}...\n")

    # Build webs for all words
    webs = {}
    if verbose: print(f"Building composition webs for {len(corpus)} words...")
    for i, w in enumerate(corpus):
        webs[w] = build_composition_web(w, db, max_nodes=50)
        if verbose and (i+1) % 50 == 0:
            print(f"  {i+1}/{len(corpus)} webs built...")

    # ── STATISTICS ──
    stats = defaultdict(list)
    for w, web in webs.items():
        nodes = web["all_nodes"]
        stats["total_nodes"].append(len(nodes))
        stats["L0_direct"].append(web["stats"]["L0_direct"])
        stats["L1_homophone"].append(web["stats"]["L1_homophone"])
        stats["L2_synonym"].append(web["stats"]["L2_synonym"])
        stats["L3_chain"].append(web["stats"]["L3_chain"])
        stats["cross_above_04"].append(web["stats"]["cross_above_04"])
        stats["cross_above_05"].append(web["stats"]["cross_above_05"])
        stats["best_cross"].append(max((n["cross"] for n in nodes.values()), default=0))

    n = len(webs)
    print(f"\n{'='*60}")
    print(f"COMPOSITION WEB STATISTICS ({n} words)")
    print(f"{'='*60}")
    print(f"  Nodes per web:       {np.mean(stats['total_nodes']):.1f} ± {np.std(stats['total_nodes']):.1f}")
    print(f"  L0 direct matches:   {np.mean(stats['L0_direct']):.1f}")
    print(f"  L1 homophone class:  {np.mean(stats['L1_homophone']):.1f}")
    print(f"  L2 synonym:          {np.mean(stats['L2_synonym']):.1f}")
    print(f"  L3 chain-web:        {np.mean(stats['L3_chain']):.1f}")
    print(f"  Cross ≥ 0.40:        {np.mean(stats['cross_above_04']):.1f} nodes/web")
    print(f"  Cross ≥ 0.50:        {np.mean(stats['cross_above_05']):.1f} nodes/web")
    print(f"  Best cross per word: {np.mean(stats['best_cross']):.3f}")

    # How many words have at least one node with cross ≥ 0.40?
    good = sum(1 for w,web in webs.items()
               if any(n["cross"] >= 0.40 for n in web["all_nodes"].values()))
    great = sum(1 for w,web in webs.items()
                if any(n["cross"] >= 0.50 for n in web["all_nodes"].values()))
    print(f"\n  Words with cross ≥ 0.40: {good}/{n} ({100*good/n:.0f}%)")
    print(f"  Words with cross ≥ 0.50: {great}/{n} ({100*great/n:.0f}%)")

    # Set-theoretic: meaning universe coverage
    all_meaning = set()
    for w, web in webs.items():
        for node in web["all_nodes"].values():
            all_meaning |= node["meaning_set"]
    print(f"\n  Total meaning universe: {len(all_meaning)} unique EN↔FR edges")
    print(f"  Words with meaning paths: {sum(1 for w,web in webs.items() if any(n['meaning_size']>0 for n in web['all_nodes'].values()))}/{n}")

    # Top 10 composition webs by quality
    by_quality = sorted(webs.items(), key=lambda x: max((n["cross"] for n in x[1]["all_nodes"].values()), default=0), reverse=True)
    print(f"\n  TOP COMPOSITION WEBS:")
    for w, web in by_quality[:10]:
        best = max(web["all_nodes"].values(), key=lambda n: n["cross"])
        nodes_str = "  ".join(
            f"{n['fr']}({n['cross']:.2f})" for n in
            sorted(web["all_nodes"].values(), key=lambda n: -n["cross"])[:5]
        )
        print(f"    {w:15s} → {nodes_str}")

    return webs, stats

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Composition Web v6 — set-theory algebraic mixing")
    ap.add_argument("--corpus", type=int, default=0, help="Build N-word composition corpus")
    ap.add_argument("--web", type=str, default="", help="Show composition web for one word")
    ap.add_argument("--phrase", type=str, default="", help="Set-cover compose a phrase")
    ap.add_argument("--db-only", action="store_true")
    ap.add_argument("--base-dir", default=".")
    args = ap.parse_args()
    os.chdir(args.base_dir)
    db = load_db(args.base_dir)

    stats_summary = (
        f"ladder: {sum(len(v) for v in db['ladder'].values())} edges, "
        f"v7: {len(db['v7_gold'])} entries, "
        f"strict: {len(db['strict_gold'])} entries, "
        f"dual: {sum(len(v) for v in db['dual'].values())} entries, "
        f"fr_class: {len(db['fr_class'])} homophone classes, "
        f"syn_fr: {sum(len(v) for v in db['syn_fr'].values())} synonym edges, "
        f"chain: {sum(len(v) for v in db['chain'].values())} chain edges, "
        f"fr_to_en: {len(db['fr_to_en'])} meaning paths"
    )
    print(f"COMPOSITION WEB DB: {stats_summary}\n")

    if args.db_only: return

    if args.corpus:
        corpus_compose(args.corpus, db)
    elif args.web:
        web = build_composition_web(args.web, db, max_nodes=100)
        print(f"COMPOSITION WEB: '{args.web}'")
        print(f"  {web['stats']['total_nodes']} nodes total")
        for layer_name, layer_nodes in sorted(web["layers"].items()):
            print(f"\n  LAYER {layer_name} ({len(layer_nodes)} nodes):")
            for n in sorted(layer_nodes, key=lambda n: -n["score"])[:10]:
                print(f"    {n['fr']:18s}  cross={n['cross']:.3f}  "
                      f"score={n['score']:.3f}  [{n['source']}]")
    elif args.phrase:
        phrase_compose(args.phrase, db)
    else:
        # Default: 200-word corpus
        corpus_compose(200, db)

if __name__ == "__main__":
    main()
