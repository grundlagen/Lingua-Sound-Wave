#!/usr/bin/env python3
"""
TOPOLOGICAL SEQUENCE — Constrained path through the tensor product graph.

THE MATHEMATICAL STRUCTURE:

  Let G_EN = (V_EN, E_EN) be the English word adjacency graph.
    Edge (e₁, e₂) exists if: synonym(e₁,e₂) OR e₁+e₂ is a known bigram.
  Let G_FR = (V_FR, E_FR) be the French word adjacency graph.
  Let R_S = {(e,f) | S(e,f) ≥ θ} be the sound relation.
  
  The TENSOR PRODUCT GRAPH T = G_EN ⊗ G_FR restricted by R_S:
    Nodes: {(e,f) | S(e,f) ≥ θ, e ∈ V_EN, f ∈ V_FR}
    Edges: (e₁,f₁) → (e₂,f₂) iff (e₁,e₂) ∈ E_EN AND (f₁,f₂) ∈ E_FR

  A PATH through T is a bicameral sequence:
    (e₁,f₁) → (e₂,f₂) → ... → (eₙ,fₙ)
  where:
    - The English projection e₁...eₙ is a coherent English text
    - The French projection f₁...fₙ is a coherent French text  
    - Each pair satisfies S(e_i, f_i) ≥ θ

  We find the LONGEST PATH through T under sound quality constraints.
  This is the maximal bicameral paragraph.

ALGORITHM: Greedy path extension with semantic coherence.
  1. Start from the highest-scoring sound pair (best S(e,f))
  2. Extend: find next node (e',f') where both languages transition naturally
  3. Prefer transitions that advance the meaning (new semantic ground)
  4. Stop when no more coherent extensions exist
  5. Connect path segments into stanzas

Run: python topological_sequence.py --n 100
"""

from __future__ import annotations

import argparse, os, random
from collections import defaultdict, Counter
import numpy as np

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")
from topological_flow_v3 import build_polysemy_graph, STOP

FUNC_GLUE = {
    "the":"le","a":"un","an":"un","of":"de","to":"à","in":"en","on":"sur",
    "at":"à","for":"pour","and":"et","or":"ou","is":"est","are":"sont",
    "was":"était","be":"être","it":"il","he":"il","she":"elle","we":"nous",
    "they":"ils","my":"mon","his":"son","her":"sa","our":"notre",
    "this":"ce","not":"pas","but":"mais","so":"donc",
    "by":"par","with":"avec","from":"de","no":"non","all":"tout",
    "more":"plus","very":"très",
}

# ── BUILD ADJACENCY GRAPHS ──
def build_adjacency(sound, syn_en, syn_fr, homophone, E, min_sound=0.55):
    """
    Build G_EN and G_FR adjacency matrices for the 200-word universe.
    Adjacency: words that can naturally follow each other.
    """
    en_adj = defaultdict(set)   # en_word → {en_words that can follow}
    fr_adj = defaultdict(set)   # fr_word → {fr_words that can follow}
    # Sound pairs: all valid (e,f) with S≥θ
    sound_pairs = {}             # (en, fr) → score
    fr_of_en = defaultdict(list) # en → [(fr, score), ...]
    en_of_fr = defaultdict(list) # fr → [(en, score), ...]
    
    # Build sound pairs for words in E
    for en_w in E:
        for f, s in sound.get(en_w, [])[:8]:
            if s >= min_sound:
                sound_pairs[(en_w, f)] = s
                fr_of_en[en_w].append((f, s))
                en_of_fr[f].append((en_w, s))
    
    # EN adjacency: synonyms + shared FR sound
    for w1 in E:
        # Synonyms
        for w2 in syn_en.get(w1, set()) & E:
            en_adj[w1].add(w2)
        # Words sharing French sounds (co-sounding)
        for f, _ in fr_of_en.get(w1, []):
            for w2, _ in en_of_fr.get(f, []):
                if w2 != w1 and w2 in E:
                    en_adj[w1].add(w2)
    
    # FR adjacency: synonyms + shared EN meaning + homophone class
    all_fr_words = set()
    for pairs in fr_of_en.values():
        for f, _ in pairs:
            all_fr_words.add(f)
    
    for f1 in all_fr_words:
        # Synonyms
        for f2 in syn_fr.get(f1, set()) & all_fr_words:
            fr_adj[f1].add(f2)
        # Homophone class siblings
        for f2 in homophone.get(f1, set()) & all_fr_words:
            if f2 != f1:
                fr_adj[f1].add(f2)
        # Shared EN source
        for en_w, _ in en_of_fr.get(f1, []):
            for f2, _ in fr_of_en.get(en_w, []):
                if f2 != f1:
                    fr_adj[f1].add(f2)
    
    return en_adj, fr_adj, sound_pairs, fr_of_en, en_of_fr

# ── FIND PATHS THROUGH THE TENSOR PRODUCT ──
def find_bicameral_paths(E, en_adj, fr_adj, sound_pairs, fr_of_en, max_paths=20):
    """
    Find paths through G_EN ⊗ G_FR where each node satisfies R_S.
    Returns list of paths: [(e₁,f₁,s₁), (e₂,f₂,s₂), ...]
    """
    used_en = set()
    used_fr = set()
    paths = []
    
    # Sort potential starting nodes by sound quality
    starts = sorted(sound_pairs.items(), key=lambda x: -x[1])
    
    for (en_w, fr_w), score in starts:
        if en_w in used_en or fr_w in used_fr:
            continue
        
        # Start a new path
        path = [(en_w, fr_w, score)]
        used_en.add(en_w)
        used_fr.add(fr_w)
        
        # Extend greedily
        current_en = en_w
        current_fr = fr_w
        
        for _ in range(10):  # max path length
            # Find valid next nodes: (e',f') where e'∈en_adj(current_en), f'∈fr_adj(current_fr)
            next_candidates = []
            for next_en in en_adj.get(current_en, set()):
                if next_en in used_en: continue
                for next_fr in fr_adj.get(current_fr, set()):
                    if next_fr in used_fr: continue
                    key = (next_en, next_fr)
                    if key in sound_pairs:
                        next_candidates.append((sound_pairs[key], next_en, next_fr))
            
            if not next_candidates:
                # Try looser: just sound match, any FR word
                for next_en in en_adj.get(current_en, set()):
                    if next_en in used_en: continue
                    for next_fr, s in fr_of_en.get(next_en, []):
                        if next_fr not in used_fr:
                            next_candidates.append((s, next_en, next_fr))
                            break
            
            if not next_candidates:
                break
            
            # Pick best
            next_candidates.sort(reverse=True)
            best_s, best_en, best_fr = next_candidates[0]
            
            path.append((best_en, best_fr, best_s))
            used_en.add(best_en)
            used_fr.add(best_fr)
            current_en = best_en
            current_fr = best_fr
        
        if len(path) >= 2:
            paths.append(path)
    
    return paths

# ── RENDER BICAMERAL TEXT ──
def render_paths(paths, E, sound_pairs, en_adj):
    """Render paths as readable English and French text."""
    
    # Build stanzas from paths
    stanzas_en = []
    stanzas_fr = []
    
    for path in paths:
        en_words = [e for e,f,s in path]
        fr_words = [f for e,f,s in path]
        scores = [s for e,f,s in path]
        
        # Add function words for flow
        en_text = []
        fr_text = []
        for i, (en_w, fr_w, s) in enumerate(zip(en_words, fr_words, scores)):
            if i > 0 and i % 3 == 0:
                # Add a natural connector
                conn = "and" if i % 6 == 0 else "of"
                en_text.append(conn)
                fr_text.append(FUNC_GLUE.get(conn, conn))
            en_text.append(en_w)
            fr_text.append(fr_w)
        
        stanzas_en.append(" ".join(en_text))
        stanzas_fr.append(" ".join(fr_text))
    
    total_en = sum(len(p) for p in paths)
    total_fr = sum(len(p) for p in paths)
    all_scores = [s for p in paths for _,_,s in p]
    
    print(f"\n{'='*70}")
    print(f"TOPOLOGICAL SEQUENCE — {len(paths)} paths, {total_en} words")
    print(f"{'='*70}")
    
    print(f"\n  Sound: μ={np.mean(all_scores):.3f} σ={np.std(all_scores):.3f}")
    print(f"  ≥0.70: {sum(1 for s in all_scores if s>=0.70)}/{len(all_scores)}")
    
    print(f"\n{'─'*70}")
    print(f"BICAMERAL SEQUENCE (English → French):")
    print(f"{'─'*70}")
    
    for i, path in enumerate(paths):
        print(f"\n  PATH {i+1}:")
        en_line = []
        fr_line = []
        for e, f, s in path:
            en_line.append(f"{e}[{s:.2f}]")
            fr_line.append(f)
        print(f"    EN: {' '.join(en_line)}")
        print(f"    FR: {' '.join(fr_line)}")
    
    # Combined paragraph
    print(f"\n{'─'*70}")
    print(f"ENGLISH PARAGRAPH:")
    print(f"{'─'*70}")
    for stanza in stanzas_en:
        print(f"  {stanza}")
    
    print(f"\n{'─'*70}")
    print(f"FRENCH PARAGRAPH:")
    print(f"{'─'*70}")
    for stanza in stanzas_fr:
        print(f"  {stanza}")
    
    # Save
    with open("topological_sequence.txt","w") as f:
        f.write(f"TOPOLOGICAL SEQUENCE — {total_en} words, {len(paths)} paths\n\n")
        f.write("ENGLISH:\n")
        for stanza in stanzas_en:
            f.write(f"  {stanza}\n")
        f.write("\nFRENCH:\n")
        for stanza in stanzas_fr:
            f.write(f"  {stanza}\n")
        f.write("\nDETAIL:\n")
        for i, path in enumerate(paths):
            f.write(f"  PATH {i+1}: {' → '.join(f'{e}[{s:.2f}]→{f}' for e,f,s in path)}\n")
    
    print(f"\n  Saved to topological_sequence.txt")
    return stanzas_en, stanzas_fr, paths

# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=100)
    ap.add_argument("--min-sound",type=float,default=0.55)
    args=ap.parse_args()
    
    print("Building algebraic graph...")
    sound, closed_means, homophone, syn_en, syn_fr, bridge = build_polysemy_graph(".")
    
    # Load corpus
    corpus = []
    for i,line in enumerate(open("all_composition_words.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=2 and p[1] not in STOP and 3<=len(p[1])<=8:
            corpus.append(p[1])
    corpus = corpus[:args.n]
    E = set(corpus)
    
    print(f"Corpus: {len(E)} words")
    print(f"Building adjacency graphs...")
    en_adj, fr_adj, sound_pairs, fr_of_en, en_of_fr = build_adjacency(
        sound, syn_en, syn_fr, homophone, E, args.min_sound
    )
    
    total_en_edges = sum(len(v) for v in en_adj.values())
    total_fr_edges = sum(len(v) for v in fr_adj.values())
    print(f"  EN adjacency: {total_en_edges} edges")
    print(f"  FR adjacency: {total_fr_edges} edges")
    print(f"  Sound pairs:  {len(sound_pairs)} valid (e,f)")
    
    # Find paths
    print(f"Finding bicameral paths through tensor product...")
    paths = find_bicameral_paths(E, en_adj, fr_adj, sound_pairs, fr_of_en)
    
    # Sort paths by total score
    paths.sort(key=lambda p: -sum(s for _,_,s in p))
    
    if paths:
        render_paths(paths, E, sound_pairs, en_adj)
    else:
        print("No paths found. Trying with relaxed constraints...")
        # Relax: allow any FR match, not just adjacency
        # Just pair best sound matches
        used_en = set()
        used_fr = set()
        pairs = []
        for en_w in sorted(E, key=lambda w: -max((s for f,s in sound.get(w,[(0,0)])[:5]),default=0)):
            if en_w in used_en: continue
            for f, s in sound.get(en_w, [])[:5]:
                if s >= args.min_sound and f not in used_fr:
                    pairs.append((en_w, f, s))
                    used_en.add(en_w)
                    used_fr.add(f)
                    break
        
        if pairs:
            # Build adjacency from these pairs - any sequential order
            paths = []
            for i in range(0, len(pairs), 6):
                chunk = pairs[i:i+6]
                if len(chunk) >= 2:
                    paths.append(chunk)
            
            if paths:
                render_paths(paths, E, sound_pairs, en_adj)

if __name__=="__main__": main()
