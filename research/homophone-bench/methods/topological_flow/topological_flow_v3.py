#!/usr/bin/env python3
"""
TOPOLOGICAL FLOW v3 — Polysemy, periphrastic, many-to-many algebraic flow.

THE DEEPER ALGEBRA:

  1. POLYSEMY: Words like "set" have 50+ senses. Each sense is a distinct
     target in the meaning universe. A French word covers a sense with
     semantic cosine ≥ threshold. Coverage is per-sense, not per-word.

  2. PERIPHRASTIC: x EN words can match to y FR words. Not 1:1.
     - 1 EN → many FR: "set off" → "dé" + "clencher" (carve decomposition)
     - Many EN → 1 FR: "see" and "sea" both → "si" (homophone sharing)
     - The relation is a BIPARTITE MANY-TO-MANY correspondence.

  3. MEANING AS WEIGHTED SETS: Each FR word f has a weighted meaning set
     M(f) = {(e, cos(f,e))} where cos is semantic cosine. Coverage is
     achieved when every sense in the universe has some f with cos ≥ θₘ.

  4. HOMOPHONE CLASS AS COVER AMPLIFIER: FR homophone [vɛʁ] = {vert, verre, 
     vers, ver, vair}. Each sibling brings its OWN meaning set. So one sound
     match unlocks 5+ meaning sets simultaneously. This is the ALGEBRAIC
     POWER of homophone classes — they multiply meaning coverage per sound.

  5. POETIC DRIFT: Metaphor/kenning edges with sound≥0.6, cos≥0.25 allow
     creative meaning coverage — "garden"≈"gardien" (metonym).

THE MANY-TO-MANY FLOW ALGORITHM:

  For each EN word e, generate multiple FR sound candidates.
  For each FR candidate f, compute its expanded meaning coverage M*(f).
  Build the hypergraph of possible (EN_set → FR_set) assignments.
  Greedy cover: pick the FR word that maximizes (new_senses_covered × sound_score).
  Allow EN words to share FR words (homophone reuse).
  Allow EN words to split into multiple FR words (periphrastic carve).

Run: python topological_flow_v3.py --n 200
"""

from __future__ import annotations

import argparse, os, sys, math
from collections import defaultdict, Counter
import numpy as np

# ═══════════════════════════════════════════════════════════════════
def build_polysemy_graph(b=".", min_sound=0.40):
    """Build the many-to-many algebraic composition graph."""
    
    sound = defaultdict(list)         # EN → [(FR, score), ...]
    raw_means = defaultdict(set)      # FR → {EN word, ...}
    homophone = defaultdict(set)       # FR ↔ FR
    synonym_en = defaultdict(set)      # EN ↔ EN
    synonym_fr = defaultdict(set)      # FR ↔ FR
    bridge = defaultdict(list)         # EN → [(FR, sound, meaning), ...] (Haiku cross-scope)
    carve = defaultdict(list)          # EN → [(multi_fr, score), ...] (periphrastic)
    
    # ── tier-ladder ──
    for i,line in enumerate(open(f"{b}/tier-ladder.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=12 and p[10]:
            try:
                s=float(p[10]); en=p[1]; fr=p[2]
                if s>=min_sound: sound[en].append((fr,s)); raw_means[fr].add(en)
            except: continue
    
    # ── dual-pairs ──
    for i,line in enumerate(open(f"{b}/dual-pairs.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=6 and p[0].lower()!=p[1].lower():
            s=float(p[2])
            if s>=min_sound: sound[p[0]].append((p[1],s)); raw_means[p[1]].add(p[0])
    
    # ── strict-gold + v7 ──
    for i,line in enumerate(open(f"{b}/strict-gold.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=2: sound[p[0]].append((p[1],1.0)); raw_means[p[1]].add(p[0])
    for i,line in enumerate(open(f"{b}/dictionary-v7.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=9 and p[3]=="1":
            sound[p[7]].append((p[8],float(p[1]))); raw_means[p[8]].add(p[7])
    
    # ── FR homophone classes ──
    for path in [f"{b}/fr-homophone-classes-lexique.tsv",f"{b}/fr-homophone-classes.tsv"]:
        try:
            for i,line in enumerate(open(path,encoding="utf-8")):
                if i==0: continue
                ms=line.rstrip("\n").split("\t")[1].split()
                if len(ms)>=2:
                    for m in ms: homophone[m].update(ms)
        except: pass
    
    # ── Synonyms (EN + FR) ──
    for line in open(f"{b}/muse-pivot-syn.tsv",encoding="utf-8"):
        a,b,_=line.rstrip("\n").split("\t")
        if a.startswith("en:") and b.startswith("en:"):
            synonym_en[a[3:]].add(b[3:]); synonym_en[b[3:]].add(a[3:])
        elif a.startswith("fr:") and b.startswith("fr:"):
            synonym_fr[a[3:]].add(b[3:]); synonym_fr[b[3:]].add(a[3:])
    
    # ── Haiku bridges (periphrastic/cross-scope) ──
    try:
        for i,line in enumerate(open(f"{b}/llm-bridge.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=4: bridge[p[0]].append((p[1],float(p[2]),float(p[3])))
    except: pass
    
    # Sort
    for k in sound: sound[k].sort(key=lambda x:-x[1])
    
    # ── ALGEBRAIC CLOSURE of meaning sets ──
    print("  computing polysemy closure...")
    closed_means = {}
    for f, raw in raw_means.items():
        closed = set(raw)
        # Homophone inheritance
        for f_hom in homophone.get(f, set()):
            closed |= raw_means.get(f_hom, set())
        # EN synonym expansion
        expanded = set(closed)
        for e in closed:
            expanded |= synonym_en.get(e, set())
        # FR synonym expansion on homophone siblings
        for f_hom in homophone.get(f, set()):
            for f_syn in synonym_fr.get(f_hom, set()):
                expanded |= raw_means.get(f_syn, set())
        closed_means[f] = expanded
    
    # ── Bridge expansion ──
    for en_w, entries in bridge.items():
        for fr, s, m in entries:
            if s >= min_sound:
                sound[en_w].append((fr, s))
                closed_means.setdefault(fr, set())
    
    # Sort sound again after adding bridges
    for k in sound: sound[k].sort(key=lambda x:-x[1])
    
    return sound, closed_means, homophone, synonym_en, synonym_fr, bridge

# ═══════════════════════════════════════════════════════════════════
# SENSE-LEVEL MEANING UNIVERSE
# ═══════════════════════════════════════════════════════════════════
def build_sense_universe(en_words, synonym_en):
    """
    Build a SENSE-LEVEL meaning universe.
    
    Each EN word contributes its SYNONYM CLUSTER as a "sense region."
    A French word that covers any synonym in the cluster covers the whole sense.
    This is the polysemy-aware formulation: covering "big" covers "large" too.
    """
    sense_universe = set()
    word_to_senses = {}
    
    for w in en_words:
        cluster = {w} | synonym_en.get(w, set())
        sense_id = frozenset(cluster)
        sense_universe.add(sense_id)
        word_to_senses[w] = sense_id
    
    return sense_universe, word_to_senses

# ═══════════════════════════════════════════════════════════════════
# MANY-TO-MANY PERSISTENT COVER
# ═══════════════════════════════════════════════════════════════════
def many_to_many_cover(en_words, sound, closed_means, homophone, synonym_en, synonym_fr,
                       bridge, min_sound=0.40, max_fr_per_en=3, allow_share=True):
    """
    Many-to-many persistent cover.
    
    Allows:
      - 1 EN → multiple FR words (periphrastic carve)
      - Multiple EN → same FR word (homophone sharing)
      - Meaning coverage via synonym clusters (polysemy-aware)
    
    Returns assignments + coverage statistics.
    """
    E = set(en_words)
    n = len(E)
    
    # Build sense universe
    senses = {}
    for w in E:
        senses[w] = {w} | synonym_en.get(w, set())
    
    # Generate ALL candidate assignments: (score, EN_set, FR_word, meaning_coverage)
    candidates = []
    
    for e_i in E:
        # Single-word matches
        for f, s_score in sound.get(e_i, [])[:15]:
            if s_score < min_sound: break
            m_set = closed_means.get(f, set())
            # How many EN words in our universe does this FR word cover?
            covered = m_set & E
            # Also cover synonyms of covered words
            expanded = set(covered)
            for cw in list(covered):
                expanded |= synonym_en.get(cw, set()) & E
            # Score: sound × coverage breadth
            score = s_score * (1.0 + 0.1 * len(expanded))
            candidates.append((score, frozenset([e_i]), f, expanded))
            
            # Homophone relay: f ≈ f_hom with different meanings
            for f_hom in list(homophone.get(f, set()))[:5]:
                if f_hom == f: continue
                hom_covered = closed_means.get(f_hom, set()) & E
                hom_expanded = set(hom_covered)
                for cw in list(hom_covered):
                    hom_expanded |= synonym_en.get(cw, set()) & E
                if hom_expanded - expanded:  # new coverage
                    score = s_score * 0.95 * (1.0 + 0.1 * len(hom_expanded))
                    candidates.append((score, frozenset([e_i]), f"~{f_hom}", hom_expanded))
        
        # Periphrastic: EN word → multiple FR words (carve)
        # For words with weak single-FR matches, split into multi-word French
        best_single = max((s for f,s in sound.get(e_i,[])[:3]), default=0)
        if best_single < 0.55 and len(e_i) >= 5:
            # Simple carve: split into 2-3 shorter FR syllables
            for f1, s1 in sound.get(e_i, [])[:5]:
                if s1 < 0.45: break
                for f2, s2 in sound.get(e_i, [])[:5]:
                    if s2 < 0.45: break
                    avg_s = (s1+s2)/2
                    m1 = closed_means.get(f1, set()) & E
                    m2 = closed_means.get(f2, set()) & E
                    combined_m = m1 | m2
                    expanded_m = set(combined_m)
                    for cw in list(combined_m):
                        expanded_m |= synonym_en.get(cw, set()) & E
                    score = avg_s * 0.85 * (1.0 + 0.1 * len(expanded_m))
                    candidates.append((score, frozenset([e_i]), f"{f1}+{f2}", expanded_m))
    
    candidates.sort(reverse=True)
    
    # Greedy assignment with many-to-many
    assigned_sound = defaultdict(list)   # EN → [(FR, score), ...]
    used_fr_sound = Counter()            # FR → usage count
    covered_en = set()                   # EN words whose meaning is covered
    persistence = 1.0
    
    for score, en_set, f, meaning_set in candidates:
        en_list = list(en_set)
        # Check: at least one EN word in the set still needs sound
        needs_sound = [e for e in en_list if len(assigned_sound[e]) < max_fr_per_en]
        if not needs_sound: continue
        
        # Check: French word not overused
        base_f = f.replace("~","").split("+")[0]
        if not allow_share and used_fr_sound[base_f] >= 1: continue
        if used_fr_sound[base_f] >= 3: continue  # max 3 EN words sharing one FR
        
        # Assign to the first EN word that needs it
        e = needs_sound[0]
        assigned_sound[e].append((f, score))
        used_fr_sound[base_f] += 1
        covered_en |= meaning_set
        persistence = min(persistence, score)
        
        # Check completion
        all_have_sound = all(len(assigned_sound[e]) >= 1 for e in E)
        all_covered = len(covered_en) >= n
        if all_have_sound and all_covered:
            break
    
    # Fill remaining gaps
    for e in E:
        if not assigned_sound[e]:
            for f, s in sound.get(e, [])[:10]:
                if s >= 0.30:
                    assigned_sound[e].append((f, s))
                    covered_en |= closed_means.get(f, set()) & E
                    persistence = min(persistence, s)
                    break
    
    # Homophone relay for uncovered meanings
    for e in list(E - covered_en):
        for e_i, fr_list in assigned_sound.items():
            for f, _ in fr_list:
                base_f = f.replace("~","").split("+")[0]
                for f_hom in homophone.get(base_f, set()):
                    if e in closed_means.get(f_hom, set()):
                        covered_en.add(e)
                        break
                if e in covered_en: break
            if e in covered_en: break
    
    uncovered_sound = {e for e in E if not assigned_sound[e]}
    uncovered_meaning = E - covered_en
    
    # Statistics
    total_fr = sum(len(v) for v in assigned_sound.values())
    shared_fr = sum(1 for f,c in used_fr_sound.items() if c > 1)
    periphrastic = sum(1 for v in assigned_sound.values() if len(v) > 1)
    
    return {
        "n": n, "candidates": len(candidates),
        "assigned_sound": n - len(uncovered_sound),
        "covered_meaning": n - len(uncovered_meaning),
        "persistence": persistence,
        "total_fr_words": total_fr,
        "shared_fr_words": shared_fr,
        "periphrastic_en": periphrastic,
        "uncovered_sound": uncovered_sound,
        "uncovered_meaning": uncovered_meaning,
    }

# ═══════════════════════════════════════════════════════════════════
def analyze_multiple_meanings(en_words, sound, closed_means, homophone):
    """Analyze polysemy: which FR words have the most diverse meaning coverage?"""
    E = set(en_words)
    fr_power = []
    
    for e in E:
        for f, s in sound.get(e, [])[:5]:
            if s < 0.50: break
            m_set = closed_means.get(f, set()) & E
            hom_set = set()
            for f_hom in homophone.get(f, set()):
                hom_set |= closed_means.get(f_hom, set()) & E
            total = m_set | hom_set
            fr_power.append((len(total), s, f, e, sorted(total)[:8]))
    
    fr_power.sort(reverse=True)
    return fr_power

# ═══════════════════════════════════════════════════════════════════
STOP = set("the a an of to in on at for and or is are was be it he she we you "
           "they my his her its our your this that not but so as by with do did".split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=200)
    ap.add_argument("--analyze",action="store_true")
    ap.add_argument("--base-dir",default=".")
    args=ap.parse_args()
    os.chdir(args.base_dir)
    
    print("Building polysemy graph...")
    sound, closed_means, homophone, syn_en, syn_fr, bridge = build_polysemy_graph(args.base_dir)
    print(f"  {len(sound)} EN sound nodes, {len(closed_means)} FR meaning nodes")
    print(f"  {len(homophone)} homophone classes, {sum(len(v) for v in syn_en.values())} EN syn edges")
    print(f"  {sum(len(v) for v in bridge.values())} bridge edges")
    
    # Load corpus
    corpus = []
    try:
        for i,line in enumerate(open("all_composition_words.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=2 and p[1] not in STOP and len(p[1])>=2:
                corpus.append(p[1])
    except: corpus = [w for w in sound if w not in STOP and len(w)>=2]
    corpus = corpus[:args.n]
    print(f"\nCorpus: {len(corpus)} words")
    
    if args.analyze:
        # Polysemy analysis: which FR words cover the most meanings?
        powers = analyze_multiple_meanings(corpus, sound, closed_means, homophone)
        print(f"\nPOLYSEMY POWER ANALYSIS — top FR words by meaning coverage:")
        print(f"  {'FR_word':20s} {'EN_source':15s} {'sound':>6s} {'#meanings':>10s} {'sample_meanings'}")
        for n_mean, s, f, e, sample in powers[:30]:
            print(f"  {f:20s} {e:15s} {s:6.3f} {n_mean:10d} {sample}")
        return
    
    # Run many-to-many cover
    r = many_to_many_cover(corpus, sound, closed_means, homophone, syn_en, syn_fr, bridge,
                           min_sound=0.40, max_fr_per_en=2, allow_share=True)
    print(f"\nMANY-TO-MANY COVER:")
    print(f"  sound covered:  {r['assigned_sound']}/{r['n']}")
    print(f"  meaning covered: {r['covered_meaning']}/{r['n']}")
    print(f"  persistence:     {r['persistence']:.3f}")
    print(f"  total FR words:  {r['total_fr_words']} (avg {r['total_fr_words']/max(1,r['assigned_sound']):.1f}/EN)")
    print(f"  shared FR:       {r['shared_fr_words']} words used by multiple EN")
    print(f"  periphrastic:    {r['periphrastic_en']} EN words using multiple FR")
    print(f"  candidates:      {r['candidates']} triples")
    if r['uncovered_sound']:
        print(f"  no sound:        {sorted(r['uncovered_sound'])[:15]}")
    if r['uncovered_meaning']:
        print(f"  no meaning:      {sorted(r['uncovered_meaning'])[:15]}")

if __name__=="__main__": main()
