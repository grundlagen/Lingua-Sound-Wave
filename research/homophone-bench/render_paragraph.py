#!/usr/bin/env python3
"""Render the 200-word dual composition as readable text output."""
import os, sys
from collections import defaultdict, Counter

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")
sys.path.insert(0, ".")

# Reuse the graph builder from topological_flow_v3
from topological_flow_v3 import (
    build_polysemy_graph, many_to_many_cover, STOP
)

def render_paragraph(n=200, min_sound=0.40):
    """Build and render a 200-word dual composition paragraph."""
    
    print("Building graph...")
    sound, closed_means, homophone, syn_en, syn_fr, bridge = build_polysemy_graph(".")
    
    # Load corpus
    corpus = []
    for i,line in enumerate(open("all_composition_words.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=2 and p[1] not in STOP and len(p[1])>=2:
            corpus.append(p[1])
    corpus = corpus[:n]
    E = set(corpus)
    
    # Run many-to-many cover
    result = many_to_many_cover(corpus, sound, closed_means, homophone, syn_en, syn_fr, bridge,
                                min_sound=min_sound, max_fr_per_en=3, allow_share=True)
    
    # Collect assignments
    # We need to re-run with tracking
    # Actually, the cover function doesn't return assignments, let me build them here
    
    # Re-implement tracking version
    assigned_sound = defaultdict(list)
    used_fr_sound = Counter()
    covered_en = set()
    assignments = []  # (en_word, fr_word, score, meanings_covered)
    
    n = len(corpus)
    candidates = []
    for e_i in E:
        for f, s_score in sound.get(e_i, [])[:15]:
            if s_score < min_sound: break
            m_set = closed_means.get(f, set()) & E
            expanded = set(m_set)
            for cw in list(m_set):
                expanded |= syn_en.get(cw, set()) & E
            score = s_score * (1.0 + 0.1 * len(expanded))
            candidates.append((score, e_i, f, expanded))
            
            for f_hom in list(homophone.get(f, set()))[:5]:
                if f_hom == f: continue
                hom_covered = closed_means.get(f_hom, set()) & E
                hom_expanded = set(hom_covered)
                for cw in list(hom_covered):
                    hom_expanded |= syn_en.get(cw, set()) & E
                if hom_expanded - expanded:
                    score = s_score * 0.95 * (1.0 + 0.1 * len(hom_expanded))
                    candidates.append((score, e_i, f"≈{f_hom}", hom_expanded))
    
    candidates.sort(reverse=True)
    
    for score, e_i, f, meaning_set in candidates:
        if len(assigned_sound[e_i]) >= 3: continue
        base_f = f.replace("≈","").split("+")[0]
        if used_fr_sound[base_f] >= 3: continue
        
        assigned_sound[e_i].append((f, score))
        used_fr_sound[base_f] += 1
        covered_en |= meaning_set
        assignments.append((e_i, f, score, sorted(meaning_set & E)))
        
        all_have = all(len(assigned_sound[e]) >= 1 for e in E)
        all_covered = len(covered_en) >= n
        if all_have and all_covered:
            break
    
    # Fill gaps
    for e in E:
        if not assigned_sound[e]:
            for f, s in sound.get(e, [])[:10]:
                if s >= 0.30:
                    assigned_sound[e].append((f, s))
                    covered_en |= closed_means.get(f, set()) & E
                    assignments.append((e, f, s, sorted(closed_means.get(f, set()) & E)))
                    break
    
    # ── BUILD THE PARAGRAPH ──
    # For each English word, use its primary French assignment
    primary = {}
    for e_i, f, score, ms in assignments:
        if e_i not in primary:
            primary[e_i] = (f, score, ms)
    
    # Build the French text in order
    fr_words = []
    en_to_fr = {}
    for w in corpus:
        if w in primary:
            fr = primary[w][0].replace("≈","")
            fr_words.append(fr)
            en_to_fr[w] = fr
        else:
            fr_words.append(f"«{w}»")
            en_to_fr[w] = f"«{w}»"
    
    composed = " ".join(fr_words)
    
    # ── OUTPUT ──
    print(f"\n{'='*70}")
    print(f"200-WORD DUAL COMPOSITION")
    print(f"{'='*70}")
    print(f"\n  Persistence: {min(s for _,s,_ in primary.values()):.3f}")
    print(f"  Words: {len(corpus)} EN → {len(fr_words)} FR")
    print(f"  Sound covered: {sum(1 for w in corpus if w in primary)}/{len(corpus)}")
    print(f"  Meaning covered: {len(covered_en & E)}/{len(E)}")
    
    print(f"\n{'─'*70}")
    print(f"COMPOSED FRENCH TEXT ({len(fr_words)} words):")
    print(f"{'─'*70}")
    # Print in chunks of ~10 words per line
    for i in range(0, len(fr_words), 10):
        chunk = fr_words[i:i+10]
        print(f"  {' '.join(chunk)}")
    
    print(f"\n{'─'*70}")
    print(f"WORD-BY-WORD MAPPING:")
    print(f"  {'EN':18s} {'FR':22s} {'snd':>5s} {'meanings covered'}")
    print(f"  {'─'*18} {'─'*22} {'─'*5} {'─'*40}")
    
    for w in corpus:
        if w in primary:
            f, s, ms = primary[w]
            f_clean = f.replace("≈","")
            ms_str = ",".join(ms[:6])
            self_mark = " ↺" if w in ms else ""
            print(f"  {w:18s} {f_clean:22s} {s:5.3f} [{ms_str}]{self_mark}")
        else:
            print(f"  {w:18s} {'«no match»':22s} 0.000 [—]")
    
    # ── COVERAGE STATS ──
    # How many words are self-covered (their FR word means them)?
    self_covered = sum(1 for w in corpus if w in primary and w in primary[w][2])
    # How many French words are reused?
    fr_counts = Counter(f.replace("≈","") for f,_,_ in primary.values())
    reused = sum(1 for f,c in fr_counts.items() if c > 1)
    
    print(f"\n{'─'*70}")
    print(f"COVERAGE STATISTICS:")
    print(f"  Self-covered:     {self_covered}/{len(corpus)} ({100*self_covered/len(corpus):.0f}%)")
    print(f"  FR words reused:  {reused}")
    print(f"  Unique FR words:  {len(fr_counts)}")
    print(f"  Total meaning edges covered: {len(covered_en & E)}")
    
    # Save to file
    with open("composed_200_words.txt","w") as f:
        f.write(f"200-WORD DUAL COMPOSITION\n")
        f.write(f"{'='*70}\n\n")
        f.write("COMPOSED FRENCH TEXT:\n")
        f.write(composed + "\n\n")
        f.write("WORD-BY-WORD:\n")
        for w in corpus:
            if w in primary:
                fr, s, ms = primary[w]
                f.write(f"  {w} → {fr.replace('≈','')}  s={s:.3f}  means: {','.join(ms[:8])}\n")
            else:
                f.write(f"  {w} → NO MATCH\n")
    
    print(f"\n  Saved to composed_200_words.txt")
    return composed, primary, corpus

if __name__ == "__main__":
    composed, primary, corpus = render_paragraph(200, min_sound=0.40)
