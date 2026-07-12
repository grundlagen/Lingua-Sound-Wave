#!/usr/bin/env python3
"""
ROUND-RABBIT BICAMERAL — Full algebraic closure with diverse semantic traversal.

USES:
  - round_rabbit.py: semantic/sound lattice, UnionFind components
  - chain-web-full-v7u.tsv: 70k transitive hops (EN→FR→EN→FR→...)
  - loop-certified-pairs-v7u.tsv: 814 cycles verifying closure
  - tier-ladder.tsv: 118k ranked homophone pairs
  - fr-homophone-classes: 33k FR homophone classes
  - muse-pivot-syn: 44k EN synonyms, 51k FR synonyms

KEY INSIGHT:
  Words should not just cluster with their closest synonyms.
  Through the chain-web, "steel" can reach "metal" → "métal" → "fer" → "iron"
  through a series of meaning-preserving hops. Each hop opens new French
  sound matches. The paragraph should USE these transitive paths to create
  DIVERSE, meaningful text where the semantics travel through the graph.

  The bicameral paragraph structure:
    Each position i: English word e_i → French word f_i via loop-certified path
    The meaning of f_i covers e_j for some j (possibly different position)
    The full set {e_i} is covered by the union of meanings of {f_i}
    Every hop in the chain-web is considered for meaning connectivity
    Loop-certified pairs have higher trust (verified algebraic closure)

Run: python round_rabbit_bicameral.py --n 200
"""

from __future__ import annotations

import argparse, os, random
from collections import defaultdict, Counter
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from topological_flow_v3 import build_polysemy_graph, STOP

FUNC_GLUE = {
    "the":"le","a":"un","an":"un","of":"de","to":"à","in":"en","on":"sur",
    "at":"à","for":"pour","and":"et","or":"ou","is":"est","are":"sont",
    "was":"était","be":"être","it":"il","he":"il","she":"elle","we":"nous",
    "they":"ils","my":"mon","his":"son","her":"sa","our":"notre",
    "this":"ce","not":"pas","but":"mais","so":"donc",
    "by":"par","with":"avec","from":"de","no":"non","all":"tout",
    "their":"leur","your":"votre","its":"son",
    "more":"plus","very":"très","some":"quelques",
}

def load_chain_web(b="."):
    """Load chain-web as: en_word → [(fr_word, hops, quality, subchain_path), ...]"""
    chain = defaultdict(list)
    for i,line in enumerate(open(f"{b}/chain-web-full-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=5:
            a,b,hops,q,sub = p[0],p[1],int(p[2]),float(p[3]),p[4]
            if ":" in a and ":" in b:
                sl,sw = a.split(":",1); tl,tw = b.split(":",1)
                if sl=="en" and tl=="fr":
                    chain[sw].append((tw, hops, q, sub))
                elif sl=="fr" and tl=="en":
                    chain[tw].append((sw, hops, q, sub))
    return chain

def load_loops(b="."):
    """Load loop-certified pairs: en_word → {fr_word with certification_count}"""
    loops = defaultdict(dict)
    for i,line in enumerate(open(f"{b}/loop-certified-pairs-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=4:
            en_w, fr_w, certs = p[0], p[1], int(p[2])
            loops[en_w][fr_w] = certs
    return loops

def build_diverse_corpus(n=200, word_len=(3,10)):
    """Build diverse corpus: skip consecutive alphabetically-similar words."""
    words_all = []
    for i,line in enumerate(open("all_composition_words.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=2 and p[1] not in STOP:
            wl = len(p[1])
            if word_len[0] <= wl <= word_len[1]:
                words_all.append(p[1])
    
    # Select diverse words: pick every Nth word to spread across alphabet
    diverse = []
    step = max(1, len(words_all) // n)
    for i in range(0, len(words_all), step):
        diverse.append(words_all[i])
        if len(diverse) >= n:
            break
    return diverse

def build_round_rabbit_paragraph(n_words=200, min_sound=0.55, min_loop_certs=0):
    """Build bicameral paragraph using round-rabbit + chain-web + loops."""
    
    print("Building algebraic graph...")
    sound, closed_means, homophone, syn_en, syn_fr, bridge = build_polysemy_graph(".")
    print("Loading chain-web and loops...")
    chain = load_chain_web(".")
    loops = load_loops(".")
    print(f"  chain-web: {sum(len(v) for v in chain.values())} edges")
    print(f"  loops: {len(loops)} EN words with certified loop pairs")
    
    # Build diverse corpus
    corpus = build_diverse_corpus(n_words)
    E = set(corpus)
    print(f"\nCorpus: {len(corpus)} diverse words")
    
    # ── STAGE 1: For each EN word, find best FR candidate ──
    # Priority: loop-certified > chain-web > ladder > dual
    
    assignments = []  # (en_word, fr_word, score, channel, meaning_set)
    used_fr = set()
    
    for en_w in corpus:
        best_fr = None
        best_score = 0
        best_channel = ""
        best_meaning = set()
        
        # 1. Loop-certified pairs (highest trust — algebraic closure verified)
        for fr_w, certs in loops.get(en_w, {}).items():
            if certs >= min_loop_certs and fr_w not in used_fr:
                s = max((sc for f,sc in sound.get(en_w,[(0,0)])[:10] if f==fr_w), default=0)
                if s > best_score:
                    best_score = s
                    best_fr = fr_w
                    best_channel = f"loop({certs})"
                    best_meaning = closed_means.get(fr_w, set()) & E
        
        # 2. Chain-web paths (transitive meaning — can reach different semantics)
        if not best_fr or best_score < 0.70:
            for fr_w, hops, q, subpath in chain.get(en_w, []):
                if fr_w not in used_fr and q > best_score:
                    best_score = q
                    best_fr = fr_w
                    best_channel = f"chain_h{hops}"
                    best_meaning = closed_means.get(fr_w, set()) & E
        
        # 3. Direct sound matches
        if not best_fr or best_score < min_sound:
            for f, s in sound.get(en_w, [])[:8]:
                if s >= min_sound and f not in used_fr:
                    best_score = s
                    best_fr = f
                    best_channel = "direct"
                    best_meaning = closed_means.get(f, set()) & E
                    break
        
        if best_fr and best_score >= min_sound:
            used_fr.add(best_fr)
            assignments.append((en_w, best_fr, best_score, best_channel, best_meaning))
    
    used_en = {a[0] for a in assignments}
    uncovered_en = E - used_en
    uncovered_meaning = E - set().union(*(a[4] for a in assignments))
    
    # ── STAGE 2: Fill meaning gaps via homophone relay ──
    for e_j in list(uncovered_meaning):
        for en_w, fr_w, score, ch, ms in assignments:
            for f_hom in homophone.get(fr_w, set()):
                if e_j in closed_means.get(f_hom, set()):
                    uncovered_meaning.discard(e_j)
                    break
            if e_j not in uncovered_meaning:
                break
    
    # ── STAGE 3: Group into semantic stanzas ──
    # Use chain-web to find semantic neighborhoods: words connected by ≤3 hops
    semantic_groups = []
    assigned_to_group = set()
    
    for en_w in corpus:
        if en_w in assigned_to_group: continue
        if en_w not in used_en: continue
        
        # Find all words reachable via chain-web from this word
        group = {en_w}
        for fr_w, hops, q, sub in chain.get(en_w, []):
            if hops <= 3:
                for target_en in closed_means.get(fr_w, set()) & used_en:
                    if target_en not in assigned_to_group:
                        group.add(target_en)
        
        if len(group) >= 2:
            semantic_groups.append(list(group))
            assigned_to_group.update(group)
    
    # Remaining solo words get their own groups
    for en_w in corpus:
        if en_w in used_en and en_w not in assigned_to_group:
            semantic_groups.append([en_w])
            assigned_to_group.add(en_w)
    
    # ── STAGE 4: Build the paragraph ──
    en_text_lines = []
    fr_text_lines = []
    detail_lines = []
    
    total_assigned = 0
    
    for group in semantic_groups:
        en_line = []
        fr_line = []
        detail = []
        
        # Find assignments for these words
        group_pairs = []
        for en_w in group:
            for a in assignments:
                if a[0] == en_w:
                    group_pairs.append(a)
                    break
        
        # Sort by semantic diversity: prefer words that bring NEW meaning coverage
        covered_so_far = set()
        group_pairs.sort(key=lambda a: -len(a[4] - covered_so_far))
        
        for i, (en_w, fr_w, score, ch, ms) in enumerate(group_pairs):
            # Add connector every 3-4 words
            if i > 0 and i % 3 == 0:
                conn = random.choice(["and","of","in","with","for","to","the","a"])
                en_line.append(conn)
                fr_line.append(FUNC_GLUE.get(conn, conn))
            
            en_line.append(en_w)
            fr_line.append(fr_w)
            covered_so_far |= ms
            detail.append(f"{en_w}[{score:.2f}|{ch}]→{fr_w}")
            total_assigned += 1
        
        if len(en_line) >= 2:
            en_text_lines.append(" ".join(en_line))
            fr_text_lines.append(" ".join(fr_line))
            detail_lines.append("  " + " ".join(detail))
    
    # ── OUTPUT ──
    print(f"\n{'='*70}")
    print(f"ROUND-RABBIT BICAMERAL — {total_assigned} words, {len(semantic_groups)} groups")
    print(f"{'='*70}")
    
    # Compute stats
    scores = [a[2] for a in assignments]
    channels = Counter(a[3] for a in assignments)
    loop_count = sum(1 for a in assignments if "loop" in a[3])
    chain_count = sum(1 for a in assignments if "chain" in a[3])
    
    print(f"\n  Words:         {len(assignments)} assigned, {len(uncovered_en)} uncovered")
    print(f"  Meaning:       {len(E)-len(uncovered_meaning)}/{len(E)} covered")
    print(f"  Sound:         μ={np.mean(scores):.3f} σ={np.std(scores):.3f}")
    print(f"  ≥0.70:         {sum(1 for s in scores if s>=0.70)}/{len(scores)}")
    print(f"  Loop-certified:{loop_count}  Chain-web: {chain_count}")
    print(f"  Channels:      {dict(channels.most_common(5))}")
    
    print(f"\n{'─'*70}")
    print(f"ENGLISH:")
    print(f"{'─'*70}")
    for i, line in enumerate(en_text_lines):
        print(f"  [{i+1}] {line}")
    
    print(f"\n{'─'*70}")
    print(f"FRENCH:")
    print(f"{'─'*70}")
    for i, line in enumerate(fr_text_lines):
        print(f"  [{i+1}] {line}")
    
    print(f"\n{'─'*70}")
    print(f"DETAIL (hops visible):")
    print(f"{'─'*70}")
    for i, detail in enumerate(detail_lines[:15]):
        print(f"  [{i+1}] {detail}")
    
    # Save
    with open("round_rabbit_paragraph.txt","w") as f:
        f.write(f"ROUND-RABBIT BICAMERAL — {total_assigned} words\n\n")
        f.write("ENGLISH:\n")
        for line in en_text_lines:
            f.write(f"  {line}\n")
        f.write("\nFRENCH:\n")
        for line in fr_text_lines:
            f.write(f"  {line}\n")
        f.write("\nDETAIL:\n")
        for line in detail_lines:
            f.write(f"{line}\n")
        f.write(f"\nSTATS: μ={np.mean(scores):.3f}, "
                f"loop={loop_count}, chain={chain_count}\n")
    
    print(f"\n  Saved to round_rabbit_paragraph.txt")
    return en_text_lines, fr_text_lines, assignments

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=200)
    ap.add_argument("--min-sound",type=float,default=0.55)
    ap.add_argument("--min-loop",type=int,default=0)
    args=ap.parse_args()
    build_round_rabbit_paragraph(args.n, args.min_sound, args.min_loop)

if __name__=="__main__": main()
