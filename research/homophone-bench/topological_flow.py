#!/usr/bin/env python3
"""
TOPOLOGICAL FLOW — Persistent homology dual composition on 200-word sets.

MATHEMATICAL FORMULATION:

  Let E = {e₁,...,eₙ} be N English content words.
  Let F be the set of all French candidate words.

  Sound relation:  S(e, f) ∈ [0,1] — how well f sounds like e
  Meaning relation: M(f, e) ∈ [0,1] — how well f means e

  A DUAL COMPOSITION at threshold θ is a pair of matchings (μ, π):
    μ: E → F   (sound — each e matched to one f, S(e,μ(e)) ≥ θ)
    π: F → E   (meaning — each f's meaning covers some e, M(f,π(f)) ≥ θ)
    such that π∘μ is a surjection on E (every e is covered by some meaning)

  EQUIVALENTLY: find a set of French words F' ⊂ F such that:
    1. |F'| = n (one per English word)
    2. ∃ bijection b: E → F' with S(e, b(e)) ≥ θ
    3. ⋃_{f∈F'} M(f) ⊇ E (meaning coverage)

TOPOLOGICAL PERSPECTIVE:

  At threshold θ, build the bipartite connection graph G(θ):
    Edge (eᵢ, eⱼ) exists iff ∃ f ∈ F with S(eᵢ, f) ≥ θ AND M(f, eⱼ) ≥ θ.
    (eᵢ sounds like f, and f means eⱼ)

  As θ decreases from 1.0 → 0, G(θ) gains edges monotonically.
  This is a FILTERED GRAPH — the central object of persistent homology.

  The PERSISTENCE of a connection = the highest θ at which it exists.
  A persistent path cover at θ = a spanning subgraph where every node 
  has indegree ≥ 1 AND outdegree ≤ 1 (sound constraint) AND total 
  indegree ≥ n (meaning coverage).

  We find the MAXIMUM θ for which such a cover exists, then construct 
  the cover at that persistence level.

ALGORITHM (persistent greedy cover):

  1. Build the full connection matrix C[eᵢ][eⱼ] = max_f S(eᵢ,f) × M(f,eⱼ)
  2. Sort all triples (eᵢ, f, eⱼ) by score descending (persistence order)
  3. Greedily add edges (eᵢ, f, eⱼ) to the cover, maintaining:
     - Each eᵢ gets at most 1 sound edge (outdegree ≤ 1)
     - Each f gets used at most once per sound AND once per meaning
     - Track meaning coverage: which eⱼ are covered
  4. Stop when all eⱼ are covered
  5. Report the minimum score in the cover (= persistence level)

Run: python topological_flow.py --n 200
     python topological_flow.py --persistence
"""

from __future__ import annotations

import argparse, json, os, sys
from collections import defaultdict
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# BUILD THE FULL CONNECTION GRAPH
# ═══════════════════════════════════════════════════════════════════
def build_connection_matrix(b=".", min_sound=0.40, min_meaning_paths=1):
    """
    Build:
      sound_matrix:  EN word → [(FR word, score), ...]   (sorted by score)
      means_matrix:  FR word → [(EN word, score), ...]   (sorted)
      fr_to_en_set:  FR word → {EN word, ...}            (meaning set)
    
    From ladder + v7 + strict + dual sources.
    """
    sound = defaultdict(list)
    means = defaultdict(list)
    fr_to_en = defaultdict(set)

    # ── tier-ladder ──
    for i,line in enumerate(open(f"{b}/tier-ladder.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=12 and p[10]:
            try:
                s=float(p[10]); en=p[1]; fr=p[2]
                if s >= min_sound:
                    sound[en].append((fr,s))
                    means[fr].append((en,s))
                    fr_to_en[fr].add(en)
            except: continue

    # ── dual-pairs ──
    for i,line in enumerate(open(f"{b}/dual-pairs.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=6 and p[0].lower()!=p[1].lower():
            s=float(p[2])
            if s >= min_sound:
                sound[p[0]].append((p[1],s))
                means[p[1]].append((p[0],s))
                fr_to_en[p[1]].add(p[0])

    # ── strict-gold ──
    for i,line in enumerate(open(f"{b}/strict-gold.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=2:
            sound[p[0]].append((p[1],1.0))
            means[p[1]].append((p[0],1.0))
            fr_to_en[p[1]].add(p[0])

    # ── v7 gold ──
    for i,line in enumerate(open(f"{b}/dictionary-v7.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=9 and p[3]=="1":
            sound[p[7]].append((p[8],float(p[1])))
            means[p[8]].append((p[7],float(p[1])))
            fr_to_en[p[8]].add(p[7])

    # ── FR homophone classes ──
    homophone = defaultdict(set)
    for path in [f"{b}/fr-homophone-classes-lexique.tsv",f"{b}/fr-homophone-classes.tsv"]:
        try:
            for i,line in enumerate(open(path,encoding="utf-8")):
                if i==0: continue
                ms=line.rstrip("\n").split("\t")[1].split()
                if len(ms)>=2:
                    for m in ms: homophone[m].update(ms)
        except: pass

    # Sort
    for k in sound: sound[k].sort(key=lambda x:-x[1])
    for k in means: means[k].sort(key=lambda x:-x[1])

    return sound, means, fr_to_en, homophone

# ═══════════════════════════════════════════════════════════════════
# PERSISTENT GREEDY COVER — the core algorithm
# ═══════════════════════════════════════════════════════════════════
def persistent_cover(english_words, sound, means, fr_to_en, homophone,
                     min_sound=0.40, verbose=True):
    """
    Persistent greedy cover for a set of English words.
    
    1. For each eᵢ, generate all candidate triples (eᵢ, f, eⱼ) where:
       S(eᵢ, f) ≥ min_sound AND eⱼ ∈ M(f) (f can mean eⱼ)
    2. Score triple = S(eᵢ, f) × (1 if eⱼ∈E else 0.5)
    3. Sort all triples by score descending
    4. Greedily assign, maintaining constraints
    5. Track persistence = minimum score among assigned triples
    
    Returns: assignment, coverage stats, persistence level.
    """
    E = set(english_words)
    n = len(E)

    # Generate all candidate triples
    triples = []
    for e_i in E:
        for f, s_score in sound.get(e_i, [])[:20]:  # top 20 FR candidates
            if s_score < min_sound: break
            meanings = fr_to_en.get(f, set()) & E  # only count meanings in our universe
            for e_j in meanings:
                # Score: sound × meaning_presence
                score = s_score * 1.0  # full score for in-universe meanings
                triples.append((score, e_i, f, e_j))
            # Also check homophone class: f ≈ f' where f' has meanings in E
            for f_hom in list(homophone.get(f, set()))[:10]:
                if f_hom == f: continue
                hom_meanings = fr_to_en.get(f_hom, set()) & E
                for e_j in hom_meanings:
                    score = s_score * 0.85  # slight decay for homophone hop
                    triples.append((score, e_i, f"{f}≈{f_hom}", e_j))

    # Sort by score descending (persistence order)
    triples.sort(reverse=True)

    # Greedy assignment
    assigned_sound = {}   # e_i → (f, score)
    used_fr_sound = set()  # French words used for sound
    used_fr_meaning = defaultdict(set)  # f → {e_j} (what e_j does f cover?)
    covered_en = set()    # e_j covered by some meaning
    persistence = 1.0

    for score, e_i, f, e_j in triples:
        # Check constraints
        if e_i in assigned_sound: continue  # already has sound match
        if f in used_fr_sound and f not in [v[0] for v in assigned_sound.values()]:
            # French word can only be used for sound once
            # But if it's already claimed, skip
            if f in used_fr_sound: continue

        # Assign
        assigned_sound[e_i] = (f, score)
        used_fr_sound.add(f)
        used_fr_meaning[f].add(e_j)
        covered_en.add(e_j)
        persistence = min(persistence, score)

        # Stop condition: all EN words assigned AND all meanings covered
        if len(assigned_sound) >= n and len(covered_en) >= n:
            break

    # If not all covered, try to fill gaps
    uncovered_sound = E - set(assigned_sound.keys())
    uncovered_meaning = E - covered_en

    if verbose:
        print(f"\n  PERSISTENT COVER RESULTS:")
        print(f"    universe: {n} words")
        print(f"    candidates: {len(triples)} triples")
        print(f"    assigned sound: {len(assigned_sound)}/{n}")
        print(f"    covered meaning: {len(covered_en)}/{n}")
        print(f"    persistence: {persistence:.3f}")
        if uncovered_sound:
            print(f"    no sound match: {sorted(uncovered_sound)[:20]}...")
        if uncovered_meaning:
            print(f"    no meaning cover: {sorted(uncovered_meaning)[:20]}...")

    # Build the assignment display
    assignments = []
    for e_i in sorted(E):
        if e_i in assigned_sound:
            f, score = assigned_sound[e_i]
            m_set = used_fr_meaning.get(f, set())
            assignments.append((e_i, f, score, sorted(m_set)))
        else:
            assignments.append((e_i, "?", 0.0, []))

    return {
        "n": n,
        "triple_count": len(triples),
        "assigned_sound": len(assigned_sound),
        "covered_meaning": len(covered_en),
        "persistence": persistence,
        "assignments": assignments,
        "uncovered_sound": uncovered_sound,
        "uncovered_meaning": uncovered_meaning,
    }

# ═══════════════════════════════════════════════════════════════════
# PERSISTENCE DIAGRAM — sweep thresholds
# ═══════════════════════════════════════════════════════════════════
def persistence_diagram(english_words, sound, means, fr_to_en, homophone):
    """
    Sweep through thresholds and measure coverage vs persistence.
    This is the PERSISTENCE DIAGRAM of the filtered bipartite graph.
    """
    thresholds = np.arange(1.0, 0.0, -0.05)
    diagram = []

    for θ in thresholds:
        result = persistent_cover(english_words, sound, means, fr_to_en, homophone,
                                  min_sound=θ, verbose=False)
        diagram.append({
            "theta": round(θ, 2),
            "sound_pct": result["assigned_sound"] / result["n"],
            "meaning_pct": result["covered_meaning"] / result["n"],
            "persistence": result["persistence"],
        })

    return diagram

# ═══════════════════════════════════════════════════════════════════
# TOPOLOGICAL FLOW VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
def topological_summary(en_words, result, verbose=True):
    """
    Summarize the topological structure of the composition.
    """
    assignments = result["assignments"]

    # How many words have their OWN meaning covered by their OWN French word?
    self_covered = sum(1 for e, f, s, ms in assignments if e in ms)

    # How many French words cover meanings different from their source?
    cross_covered = sum(1 for e, f, s, ms in assignments
                        if ms and any(m != e for m in ms))

    # Distribution of meaning set sizes
    meaning_sizes = [len(ms) for _,_,_,ms in assignments if ms]

    if verbose:
        print(f"\n{'='*60}")
        print(f"TOPOLOGICAL FLOW — {len(en_words)}-word composition")
        print(f"{'='*60}")
        print(f"  Persistence level: {result['persistence']:.3f}")
        print(f"  Sound coverage:    {result['assigned_sound']}/{result['n']}")
        print(f"  Meaning coverage:  {result['covered_meaning']}/{result['n']}")
        print(f"  Self-covered:      {self_covered}/{result['n']} "
              f"({100*self_covered/result['n']:.0f}%)")
        print(f"  Cross-covered:     {cross_covered}/{result['n']} "
              f"({100*cross_covered/result['n']:.0f}%)")
        if meaning_sizes:
            print(f"  Mean meaning set:  {np.mean(meaning_sizes):.1f} ± "
                  f"{np.std(meaning_sizes):.1f} per FR word")

        print(f"\n  TOP 20 ASSIGNMENTS (by persistence):")
        sorted_a = sorted([a for a in assignments if a[2]>0], key=lambda x:-x[2])
        for e, f, score, ms in sorted_a[:20]:
            ms_str = ",".join(ms[:5])
            self_mark = " ↺" if e in ms else ""
            print(f"    {e:18s} ~ {f:22s} [{score:.2f}]{self_mark}  → means: [{ms_str}]")

        # Uncovered
        if result["uncovered_sound"]:
            print(f"\n  NO SOUND MATCH: {sorted(result['uncovered_sound'])}")
        if result["uncovered_meaning"]:
            print(f"  NO MEANING COVER: {sorted(result['uncovered_meaning'])}")

    return {
        "self_covered": self_covered,
        "cross_covered": cross_covered,
        "mean_meaning_size": np.mean(meaning_sizes) if meaning_sizes else 0,
    }

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
STOP = set("the a an of to in on at for and or is are was be it he she we you "
           "they my his her its our your this that not but so as by with do did".split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="Number of words")
    ap.add_argument("--persistence", action="store_true", help="Show persistence diagram")
    ap.add_argument("--min-sound", type=float, default=0.40)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--words-file", default="all_composition_words.tsv")
    args = ap.parse_args()
    os.chdir(args.base_dir)

    print("Building connection matrix...")
    sound, means, fr_to_en, homophone = build_connection_matrix(args.base_dir)
    print(f"  {len(sound)} EN words with sound paths")
    print(f"  {len(means)} FR words with meaning paths")
    print(f"  {len(homophone)} FR homophone classes")

    # Load corpus words
    corpus_words = []
    try:
        for i,line in enumerate(open(args.words_file,encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=2 and p[1] not in STOP and len(p[1])>=3:
                corpus_words.append(p[1])
    except FileNotFoundError:
        # Fallback: take words from sound matrix directly
        for w in sound:
            if w not in STOP and len(w)>=3:
                corpus_words.append(w)

    # Take top N
    corpus_words = corpus_words[:args.n]
    print(f"\nCorpus: {len(corpus_words)} words")

    if args.persistence:
        # Build persistence diagram
        print("\nBuilding persistence diagram...")
        diagram = persistence_diagram(corpus_words, sound, means, fr_to_en, homophone)
        print(f"\nPERSISTENCE DIAGRAM (filtered bipartite graph):")
        print(f"  {'θ':>6s}  {'sound%':>7s}  {'meaning%':>9s}  {'persist':>8s}")
        print(f"  {'─'*6}  {'─'*7}  {'─'*9}  {'─'*8}")
        for d in diagram:
            bar_s = "█" * int(d["sound_pct"]*20)
            bar_m = "█" * int(d["meaning_pct"]*20)
            print(f"  {d['theta']:6.2f}  {d['sound_pct']:6.1%} {bar_s:20s}  "
                  f"{d['meaning_pct']:7.1%} {bar_m:20s}  {d['persistence']:7.3f}")

        # Find the critical threshold: highest θ where meaning coverage ≥ 90%
        critical = None
        for d in reversed(diagram):
            if d["meaning_pct"] >= 0.90 and d["sound_pct"] >= 0.80:
                critical = d
                break
        if critical:
            print(f"\n  CRITICAL THRESHOLD: θ={critical['theta']:.2f} "
                  f"(sound={critical['sound_pct']:.0%}, meaning={critical['meaning_pct']:.0%})")
        else:
            print(f"\n  No critical threshold found for 90%+90% coverage.")
        return

    # Run persistent cover
    result = persistent_cover(corpus_words, sound, means, fr_to_en, homophone,
                              min_sound=args.min_sound, verbose=False)
    topological_summary(corpus_words, result, verbose=True)

if __name__ == "__main__":
    main()
