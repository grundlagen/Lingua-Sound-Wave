#!/usr/bin/env python3
"""
TOPOLOGICAL FLOW v2 — Algebraic closure, persistent homology, 200-word coverage.

THE MATHEMATICAL INSIGHT:

  The persistence diagram showed meaning coverage stalls at 74% because the
  raw meaning graph (fr_to_en) is sparse. We need ALGEBRAIC CLOSURE:

  For each FR word f, its CLOSED meaning set M*(f) includes:
    1. Direct meanings: {e | ∃ edge f→e in raw data}
    2. Homophone expansion: {e | ∃ f'≈f, edge f'→e} 
       (if "pourri" has no meaning but "pourrit" means "rotten", then
        "pourri" ≈ "pourrit" inherits that meaning)
    3. EN synonym expansion: {e' | e ∈ M*(f) AND e≡e'}
       (if f means "rotten" and "rotten"≡"decayed", then f also covers "decayed")
    4. Transitive chain-web paths: short-hop meaning chains

  With algebraic closure, the meaning graph becomes DENSE — every FR word
  potentially connects to MANY EN words through the equivalence classes.

  The FILTERED BIPARTITE GRAPH now has many more edges at each θ, and the
  persistence diagram should show higher meaning coverage.

ALGORITHM:
  Same persistent greedy cover, but using closed meaning sets M* instead of M.

Run: python topological_flow_v2.py --n 200 --persistence
"""

from __future__ import annotations

import argparse, json, os, sys
from collections import defaultdict
import numpy as np

# ═══════════════════════════════════════════════════════════════════
def build_closed_connection(b=".", min_sound=0.40):
    """Build sound matrix and ALGEBRAICALLY CLOSED meaning sets."""
    
    sound = defaultdict(list)
    raw_means = defaultdict(set)      # FR → {EN}
    homophone = defaultdict(set)       # FR ↔ FR
    synonym_en = defaultdict(set)      # EN ↔ EN
    chain_means = defaultdict(set)     # FR → {EN} via chain-web
    
    # ── tier-ladder ──
    print("  tier-ladder...")
    for i,line in enumerate(open(f"{b}/tier-ladder.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=12 and p[10]:
            try:
                s=float(p[10]); en=p[1]; fr=p[2]
                if s >= min_sound:
                    sound[en].append((fr,s))
                    raw_means[fr].add(en)
            except: continue
    
    # ── dual-pairs ──
    print("  dual-pairs...")
    for i,line in enumerate(open(f"{b}/dual-pairs.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=6 and p[0].lower()!=p[1].lower():
            s=float(p[2])
            if s >= min_sound:
                sound[p[0]].append((p[1],s))
                raw_means[p[1]].add(p[0])
    
    # ── strict-gold ──
    for i,line in enumerate(open(f"{b}/strict-gold.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=2:
            sound[p[0]].append((p[1],1.0))
            raw_means[p[1]].add(p[0])
    
    # ── v7 gold ──
    for i,line in enumerate(open(f"{b}/dictionary-v7.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=9 and p[3]=="1":
            sound[p[7]].append((p[8],float(p[1])))
            raw_means[p[8]].add(p[7])
    
    # ── FR homophone classes ──
    print("  homophone classes...")
    for path in [f"{b}/fr-homophone-classes-lexique.tsv",f"{b}/fr-homophone-classes.tsv"]:
        try:
            for i,line in enumerate(open(path,encoding="utf-8")):
                if i==0: continue
                ms=line.rstrip("\n").split("\t")[1].split()
                if len(ms)>=2:
                    for m in ms: homophone[m].update(ms)
        except: pass
    
    # ── EN synonyms ──
    print("  EN synonyms...")
    for line in open(f"{b}/muse-pivot-syn.tsv",encoding="utf-8"):
        a,b,_=line.rstrip("\n").split("\t")
        if a.startswith("en:") and b.startswith("en:"):
            synonym_en[a[3:]].add(b[3:])
            synonym_en[b[3:]].add(a[3:])
    
    # ── chain-web (transitive meaning paths) ──
    print("  chain-web...")
    try:
        for i,line in enumerate(open(f"{b}/chain-web-full-v7u.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=5:
                a,b=p[0],p[1]
                if ":" in a and ":" in b:
                    sl,sw=a.split(":",1); tl,tw=b.split(":",1)
                    if sl=="fr" and tl=="en":
                        chain_means[sw].add(tw)
                    elif sl=="en" and tl=="fr":
                        chain_means[tw].add(sw)
    except: pass

    # ── ALGEBRAIC CLOSURE: closed meaning sets ──
    print("  computing algebraic closure...")
    closed_means = {}
    for f, raw in raw_means.items():
        closed = set(raw)
        # Homophone expansion: inherit meanings from homophone siblings
        for f_hom in homophone.get(f, set()):
            if f_hom != f:
                closed |= raw_means.get(f_hom, set())
                # Chain-web expansion through homophones
                closed |= chain_means.get(f_hom, set())
        # Chain-web meanings
        closed |= chain_means.get(f, set())
        closed_means[f] = closed
    
    # EN synonym closure on meaning sets
    en_syn_closure = {}
    for e in synonym_en:
        closure = {e} | synonym_en[e]
        en_syn_closure[e] = closure

    # Sort
    for k in sound: sound[k].sort(key=lambda x:-x[1])

    stats = {
        "EN_sound": len(sound),
        "FR_meaning": len(closed_means),
        "homophone_classes": len(homophone),
        "synonym_en_edges": sum(len(v) for v in synonym_en.values()),
        "chain_edges": sum(len(v) for v in chain_means.values()),
        "total_closed_meanings": sum(len(v) for v in closed_means.values()),
    }

    return sound, closed_means, homophone, en_syn_closure, stats

# ═══════════════════════════════════════════════════════════════════
def persistent_cover_closed(en_words, sound, closed_means, homophone, en_syn,
                            min_sound=0.40):
    """
    Persistent greedy cover with ALGEBRAICALLY CLOSED meaning sets.
    
    For each triple (eᵢ, f, eⱼ), the meaning relation uses closed sets:
      eⱼ ∈ M*(f)  (the algebraic closure of f's meaning, including
                    homophone inheritance, chain-web, and synonym expansion)
    """
    E = set(en_words)
    n = len(E)

    # Pre-compute expanded meaning sets for all EN words
    # If eⱼ ∈ M*(f), then every synonym of eⱼ is also covered
    expanded_means = {}
    for f, closed in closed_means.items():
        expanded = set(closed)
        for e in list(closed):
            expanded |= en_syn.get(e, set())
        expanded_means[f] = expanded

    triples = []
    for e_i in E:
        for f, s_score in sound.get(e_i, [])[:30]:
            if s_score < min_sound: break
            meanings = expanded_means.get(f, set()) & E
            for e_j in meanings:
                triples.append((s_score, e_i, f, e_j))
            # Homophone relay: f ≈ f' where f' has meanings
            for f_hom in list(homophone.get(f, set()))[:8]:
                if f_hom == f: continue
                hom_meanings = expanded_means.get(f_hom, set()) & E
                for e_j in hom_meanings:
                    triples.append((s_score * 0.90, e_i, f"≈{f_hom}", e_j))

    triples.sort(reverse=True)

    assigned_sound = {}
    used_fr = set()
    covered_en = set()
    persistence = 1.0

    for score, e_i, f, e_j in triples:
        if e_i in assigned_sound: continue
        base_f = f.replace("≈","")
        if base_f in used_fr: continue
        assigned_sound[e_i] = (f, score)
        used_fr.add(base_f)
        covered_en.add(e_j)
        # Also cover all synonyms of e_j
        covered_en |= en_syn.get(e_j, set()) & E
        persistence = min(persistence, score)
        if len(assigned_sound) >= n and len(covered_en) >= n:
            break

    uncovered_sound = E - set(assigned_sound.keys())
    uncovered_meaning = E - covered_en

    # Try to fill uncovered meaning words through homophone relay
    for e_j in list(uncovered_meaning):
        for e_i, (f, score) in assigned_sound.items():
            base_f = f.replace("≈","")
            for f_hom in homophone.get(base_f, set()):
                if e_j in expanded_means.get(f_hom, set()):
                    covered_en.add(e_j)
                    uncovered_meaning.discard(e_j)
                    break
            if e_j not in uncovered_meaning:
                break

    return {
        "n": n, "triple_count": len(triples),
        "assigned_sound": len(assigned_sound),
        "covered_meaning": len(covered_en),
        "persistence": persistence,
        "uncovered_sound": uncovered_sound,
        "uncovered_meaning": uncovered_meaning,
    }

# ═══════════════════════════════════════════════════════════════════
def persistence_diagram_closed(en_words, sound, closed_means, homophone, en_syn):
    thresholds = np.arange(1.0, 0.0, -0.05)
    diagram = []
    for θ in thresholds:
        r = persistent_cover_closed(en_words, sound, closed_means, homophone, en_syn, θ)
        diagram.append({"theta": round(θ,2), "sound_pct": r["assigned_sound"]/r["n"],
                        "meaning_pct": r["covered_meaning"]/r["n"], "persistence": r["persistence"]})
    return diagram

# ═══════════════════════════════════════════════════════════════════
STOP = set("the a an of to in on at for and or is are was be it he she we you "
           "they my his her its our your this that not but so as by with do did".split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=200)
    ap.add_argument("--persistence",action="store_true")
    ap.add_argument("--min-sound",type=float,default=0.40)
    ap.add_argument("--base-dir",default=".")
    args=ap.parse_args()
    os.chdir(args.base_dir)

    print("Building closed connection matrix...")
    sound, closed_means, homophone, en_syn, stats = build_closed_connection(args.base_dir)
    print(f"  {stats['EN_sound']} EN words with sound paths")
    print(f"  {stats['FR_meaning']} FR words with meaning paths")
    print(f"  {stats['total_closed_meanings']} closed meaning edges "
          f"(×{stats['total_closed_meanings']/max(1,sum(len(v) for v in [closed_means])):.1f} expansion)")
    print(f"  {stats['homophone_classes']} homophone classes")
    print(f"  {stats['synonym_en_edges']} EN synonym edges")
    print(f"  {stats['chain_edges']} chain edges")

    # Load corpus
    corpus = []
    try:
        for i,line in enumerate(open("all_composition_words.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=2 and p[1] not in STOP and len(p[1])>=3:
                corpus.append(p[1])
    except: corpus = [w for w in sound if w not in STOP and len(w)>=3]
    corpus = corpus[:args.n]
    print(f"\nCorpus: {len(corpus)} words")

    if args.persistence:
        diagram = persistence_diagram_closed(corpus, sound, closed_means, homophone, en_syn)
        print(f"\nPERSISTENCE DIAGRAM (with algebraic closure):")
        print(f"  {'θ':>6s}  {'sound%':>7s}  {'meaning%':>9s}")
        for d in diagram:
            bar_s = "█"*int(d["sound_pct"]*20)
            bar_m = "█"*int(d["meaning_pct"]*20)
            print(f"  {d['theta']:6.2f}  {d['sound_pct']:6.1%} {bar_s:20s}  "
                  f"{d['meaning_pct']:7.1%} {bar_m:20s}")
        for d in reversed(diagram):
            if d["meaning_pct"]>=0.90 and d["sound_pct"]>=0.80:
                print(f"\n  CRITICAL θ={d['theta']:.2f}: sound={d['sound_pct']:.0%} "
                      f"meaning={d['meaning_pct']:.0%}")
                break
        return

    # Run cover
    r = persistent_cover_closed(corpus, sound, closed_means, homophone, en_syn, args.min_sound)
    print(f"\nCOVER (θ={args.min_sound:.2f}):")
    print(f"  sound: {r['assigned_sound']}/{r['n']}  meaning: {r['covered_meaning']}/{r['n']}")
    print(f"  persistence: {r['persistence']:.3f}")
    print(f"  uncovered meaning: {sorted(r['uncovered_meaning'])[:20]}...")

if __name__=="__main__": main()
