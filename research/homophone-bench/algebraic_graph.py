#!/usr/bin/env python3
"""
ALGEBRAIC COMPOSITION GRAPH — Recursive, bidirectional, set-theoretic.

EN can change FR. FR can change EN. Walk the graph recursively.

GRAPH STRUCTURE (bipartite, bidirectional):
  Nodes: EN words, FR words
  Edge types:
    ~  (sound):  EN → FR  (sounds like)     [from ladder, v7, strict, dual]
    =  (means):  FR → EN  (translates to)    [reverse of above]
    ≈  (homophone): FR ↔ FR  (same sound)    [33k classes]
    ≡  (synonym):   FR ↔ FR  (same meaning)  [51k edges]
    →  (chain-hop): EN → FR  (transitive)    [70k edges]
    ⋈  (EN-synonym): EN ↔ EN                  [44k edges]

RECURSIVE WALKS:
  Start at any EN word, walk depth-first through the graph.
  Each walk produces a set of reachable FR words.
  The composition of a phrase = intersection of reachable sets,
  then greedy set-cover over the meaning universe.

ALGEBRAIC PROPERTIES:
  - Closure: for any FR word f, homophone(f) = all FR words with same sound
  - For any EN word e, sound(e) = all FR words that sound like e
  - For any FR word f, meaning(f) = all EN words f can mean
  - Composition: sound(en) ∩ meaning^-1(target_meaning) gives dual candidates
  - The reachable closure: for any pair (en, fr), we can walk en → fr → en' → fr' → ...

Run: python algebraic_graph.py --build    (build full graph JSON)
     python algebraic_graph.py --stats    (graph statistics)
     python algebraic_graph.py --walk 3 beauty (3-step walks from 'beauty')
     python algebraic_graph.py --compose "the silent beauty of the endless sea"
"""

from __future__ import annotations

import argparse, json, os, sys
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# BUILD THE FULL BIDIRECTIONAL GRAPH
# ═══════════════════════════════════════════════════════════════════
def build_graph(b="."):
    """Build the complete algebraic composition graph from all data sources."""
    G = {
        "EN_nodes": set(),
        "FR_nodes": set(),
        "sound": defaultdict(list),        # EN → [(FR, score), ...]
        "means": defaultdict(list),         # FR → [(EN, score), ...]
        "homophone": defaultdict(set),      # FR ↔ FR
        "synonym_fr": defaultdict(set),     # FR ↔ FR
        "synonym_en": defaultdict(set),     # EN ↔ EN
        "chain": defaultdict(list),         # EN → [(FR, hops, quality), ...]
        "v7_gold": defaultdict(list),       # EN → [(FR, tier), ...]
        "strict_gold": defaultdict(set),    # EN → {FR, ...}
    }

    # ── tier-ladder (EN → FR sound edges) ──
    print("  tier-ladder...")
    for i,line in enumerate(open(f"{b}/tier-ladder.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=12 and p[10]:
            try:
                snd=float(p[10]); en_w=p[1]; fr_w=p[2]
                G["EN_nodes"].add(en_w); G["FR_nodes"].add(fr_w)
                G["sound"][en_w].append((fr_w,snd))
                G["means"][fr_w].append((en_w,snd))
            except: continue
    print(f"    {sum(len(v) for v in G['sound'].values())} sound edges")

    # ── v7 gold ──
    print("  v7 gold...")
    for i,line in enumerate(open(f"{b}/dictionary-v7.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=9 and p[3]=="1":
            G["v7_gold"][p[7]].append((p[8],p[0]))  # EN→FR with tier
            G["EN_nodes"].add(p[7]); G["FR_nodes"].add(p[8])

    # ── strict-gold ──
    print("  strict-gold...")
    for i,line in enumerate(open(f"{b}/strict-gold.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=2:
            G["strict_gold"][p[0]].add(p[1])
            G["EN_nodes"].add(p[0]); G["FR_nodes"].add(p[1])

    # ── dual-pairs ──
    print("  dual-pairs...")
    for i,line in enumerate(open(f"{b}/dual-pairs.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=6 and p[0].lower()!=p[1].lower():
            G["EN_nodes"].add(p[0]); G["FR_nodes"].add(p[1])
            G["sound"][p[0]].append((p[1],float(p[2])))
            G["means"][p[1]].append((p[0],float(p[2])))

    # ── FR homophone classes (the algebraic closure) ──
    print("  homophone classes...")
    for path in [f"{b}/fr-homophone-classes-lexique.tsv",f"{b}/fr-homophone-classes.tsv"]:
        try:
            for i,line in enumerate(open(path,encoding="utf-8")):
                if i==0: continue
                ms = line.rstrip("\n").split("\t")[1].split()
                if len(ms)>=2:
                    # Build complete graph within each class
                    for m1 in ms:
                        for m2 in ms:
                            if m1!=m2:
                                G["homophone"][m1].add(m2)
                        G["FR_nodes"].add(m1)
        except FileNotFoundError: pass
    print(f"    {sum(len(v) for v in G['homophone'].values())} homophone edges")

    # ── FR synonyms ──
    print("  FR synonyms...")
    for line in open(f"{b}/muse-pivot-syn.tsv",encoding="utf-8"):
        a,b,_=line.rstrip("\n").split("\t")
        if a.startswith("fr:") and b.startswith("fr:"):
            G["synonym_fr"][a[3:]].add(b[3:])
            G["synonym_fr"][b[3:]].add(a[3:])
        elif a.startswith("en:") and b.startswith("en:"):
            G["synonym_en"][a[3:]].add(b[3:])
            G["synonym_en"][b[3:]].add(a[3:])
    print(f"    {sum(len(v) for v in G['synonym_fr'].values())} FR synonym edges")
    print(f"    {sum(len(v) for v in G['synonym_en'].values())} EN synonym edges")

    # ── chain-web ──
    print("  chain-web...")
    try:
        for i,line in enumerate(open(f"{b}/chain-web-full-v7u.tsv",encoding="utf-8")):
            if i==0: continue
            p=line.rstrip("\n").split("\t")
            if len(p)>=5:
                a,b=p[0],p[1]
                if ":" in a and ":" in b:
                    sl,sw=a.split(":",1); tl,tw=b.split(":",1)
                    if sl=="en" and tl=="fr":
                        G["chain"][sw].append((tw,int(p[2]),float(p[3])))
                    elif sl=="fr" and tl=="en":
                        G["chain"][tw].append((sw,int(p[2]),float(p[3])))
    except FileNotFoundError: pass
    print(f"    {sum(len(v) for v in G['chain'].values())} chain edges")

    # ── Sort sound and means edges ──
    for k in G["sound"]: G["sound"][k].sort(key=lambda x:-x[1])
    for k in G["means"]: G["means"][k].sort(key=lambda x:-x[1])

    G["stats"] = {
        "EN_nodes": len(G["EN_nodes"]),
        "FR_nodes": len(G["FR_nodes"]),
        "sound_edges": sum(len(v) for v in G["sound"].values()),
        "means_edges": sum(len(v) for v in G["means"].values()),
        "homophone_edges": sum(len(v) for v in G["homophone"].values()),
        "synonym_fr_edges": sum(len(v) for v in G["synonym_fr"].values()),
        "synonym_en_edges": sum(len(v) for v in G["synonym_en"].values()),
        "chain_edges": sum(len(v) for v in G["chain"].values()),
        "v7_gold_entries": sum(len(v) for v in G["v7_gold"].values()),
        "strict_gold_entries": sum(len(v) for v in G["strict_gold"].values()),
    }

    return G

# ═══════════════════════════════════════════════════════════════════
# RECURSIVE ALGEBRAIC WALKS
# ═══════════════════════════════════════════════════════════════════
def walk(G, en_word, max_steps=3, max_nodes=500):
    """
    Recursive bidirectional walk through the algebraic graph.
    
    From EN_word, walk:
      Step 1: EN → FR (sound edges)
      Step 2: FR → FR' (homophone class — same sound, different meaning)
      Step 3: FR' → EN' (means edges — what does it mean?)
      Step 4: EN' → EN'' (synonym — alternative English wording)
      Step 5: EN'' → FR'' (sound again with different meaning)
      ... and so on, recursively.
    
    Returns: set of reachable (language, word) pairs with paths.
    """
    visited = set()
    reachable = []  # [(step, lang, word, path_description), ...]
    
    def dfs(word, lang, step, path):
        if step > max_steps: return
        if len(reachable) >= max_nodes: return
        key = (lang, word)
        if key in visited: return
        visited.add(key)
        reachable.append((step, lang, word, path))

        if lang == "EN":
            # EN → FR via sound
            for fr, score in G["sound"].get(word, [])[:5]:
                dfs(fr, "FR", step+1, f"{path} ~({score:.2f})→ {fr}")
            # EN → EN via synonym
            for syn in list(G["synonym_en"].get(word, set()))[:3]:
                dfs(syn, "EN", step+1, f"{path} ≡→ {syn}")

        elif lang == "FR":
            # FR → FR via homophone
            for hom in list(G["homophone"].get(word, set()))[:5]:
                dfs(hom, "FR", step+1, f"{path} ≈→ {hom}")
            # FR → FR via synonym
            for syn in list(G["synonym_fr"].get(word, set()))[:3]:
                dfs(syn, "FR", step+1, f"{path} ≡→ {syn}")
            # FR → EN via meaning
            for en, score in G["means"].get(word, [])[:5]:
                dfs(en, "EN", step+1, f"{path} =({score:.2f})→ {en}")

    dfs(en_word, "EN", 0, en_word)
    return reachable

# ═══════════════════════════════════════════════════════════════════
# EXTRACT ALL WORDS — the full corpus
# ═══════════════════════════════════════════════════════════════════
def extract_all_words(G, min_sound=0.55, min_en_len=3):
    """
    Extract ALL words from the graph that have at least one good sound match.
    Returns sorted list of (en_word, best_fr, best_score, n_candidates).
    """
    words = []
    for en_w in G["EN_nodes"]:
        if len(en_w) < min_en_len: continue
        candidates = G["sound"].get(en_w, [])
        good = [(fr,s) for fr,s in candidates if s >= min_sound]
        if good:
            best_fr, best_s = good[0]
            words.append((best_s, en_w, best_fr, len(good), len(candidates)))
    words.sort(reverse=True)
    return words

# ═══════════════════════════════════════════════════════════════════
# COMPOSE — greedy set-cover over the graph
# ═══════════════════════════════════════════════════════════════════
STOP = set("the a an of to in on at for and or is are was be it he she we you "
           "they my his her its our your this that not but so as by with do did".split())

def compose_phrase(G, en_phrase, max_candidates=5):
    """Greedy set-cover composition over the algebraic graph."""
    ws = [w.lower().strip(".,;:!?'\"") for w in en_phrase.split() if w.strip(".,;:!?'\"")]
    content = [w for w in ws if w not in STOP and len(w)>=3]
    if not content: content = ws

    # For each content word, get reachable FR candidates via walk
    webs = {}
    for w in content:
        r = walk(G, w, max_steps=3, max_nodes=200)
        fr_nodes = {}
        for step, lang, word, path in r:
            if lang == "FR":
                # Score: sound quality from the first edge
                sound_edges = G["sound"].get(w, [])
                best_sound = max((s for fr,s in sound_edges if fr==word), default=0.5)
                fr_nodes[word] = {"fr": word, "sound": best_sound, "path": path, "step": step}
        webs[w] = fr_nodes

    # Greedy set-cover
    covered = set()
    picks = {}

    for w in content:
        best_fr = None
        best_gain = -1
        for fr, node in sorted(webs[w].items(), key=lambda x: -x[1]["sound"]):
            # What EN words can this FR word mean?
            meaning = {en for en,_ in G["means"].get(fr, [])} if fr in G["means"] else set()
            gain = len(meaning - covered)
            weighted = gain * node["sound"]
            if weighted > best_gain:
                best_gain = weighted
                best_fr = (fr, node, meaning)

        if best_fr:
            fr, node, meaning = best_fr
            picks[w] = {"fr": fr, "sound": node["sound"], "meaning": meaning, "path": node["path"]}
            covered |= meaning

    # Coverage
    universe = set(content)
    coverage = len(covered & universe) / len(universe) if universe else 0

    # Final composed text
    final = []
    for w in ws:
        if w in picks:
            final.append(picks[w]["fr"])
        elif w not in STOP:
            final.append(f"«{w}»")

    return {
        "phrase": en_phrase,
        "composed": " ".join(final),
        "coverage": coverage,
        "picks": picks,
        "webs": webs,
    }

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="Build full graph JSON")
    ap.add_argument("--stats", action="store_true", help="Graph statistics")
    ap.add_argument("--words", type=int, default=0, help="Extract N best words")
    ap.add_argument("--walk", type=int, default=0, help="Walk N steps from word")
    ap.add_argument("--compose", type=str, default="", help="Compose a phrase")
    ap.add_argument("word", nargs="?", default="")
    ap.add_argument("--base-dir", default=".")
    args = ap.parse_args()
    os.chdir(args.base_dir)

    if args.build:
        G = build_graph(args.base_dir)
        # Save stats
        print(f"\nGRAPH STATS:")
        for k,v in G["stats"].items():
            print(f"  {k}: {v}")
        # Save full graph as JSON
        out = {
            "stats": G["stats"],
            "EN_nodes": sorted(G["EN_nodes"])[:5000],  # sample for size
            "sample_sound": {k: v[:3] for k,v in list(G["sound"].items())[:100]},
        }
        with open("algebraic_graph.json","w") as f:
            json.dump(out, f, ensure_ascii=False)
        print(f"\n  Saved algebraic_graph.json")
        return

    # Quick load from existing
    G = build_graph(args.base_dir)
    print(f"Graph: {G['stats']['EN_nodes']} EN, {G['stats']['FR_nodes']} FR, "
          f"{G['stats']['sound_edges']} sound edges")

    if args.stats:
        for k,v in G["stats"].items():
            print(f"  {k}: {v}")
        # Distribution of sound quality
        sounds = []
        for en,edges in G["sound"].items():
            for fr,s in edges: sounds.append(s)
        import numpy as np
        print(f"\n  Sound score distribution:")
        print(f"    mean: {np.mean(sounds):.3f}  median: {np.median(sounds):.3f}")
        print(f"    ≥0.70: {sum(1 for s in sounds if s>=0.70)} ({100*sum(1 for s in sounds if s>=0.70)/len(sounds):.0f}%)")
        print(f"    ≥0.55: {sum(1 for s in sounds if s>=0.55)} ({100*sum(1 for s in sounds if s>=0.55)/len(sounds):.0f}%)")
        return

    if args.words:
        words = extract_all_words(G, min_sound=0.55, min_en_len=3)
        print(f"WORDS with sound ≥ 0.55: {len(words)}")
        print(f"  Sample:")
        for s,en,fr,n_good,n_total in words[:args.words]:
            print(f"  {en:18s} → {fr:18s}  s={s:.3f}  ({n_good}/{n_total} good)")
        # Save all words to file
        with open("all_composition_words.tsv","w") as f:
            f.write("sound\ten\tbest_fr\tgood_candidates\ttotal_candidates\n")
            for s,en,fr,n_good,n_total in words:
                f.write(f"{s:.3f}\t{en}\t{fr}\t{n_good}\t{n_total}\n")
        print(f"\n  Saved all_composition_words.tsv ({len(words)} words)")
        return

    if args.walk and args.word:
        r = walk(G, args.word, max_steps=args.walk, max_nodes=200)
        print(f"WALK from '{args.word}' ({args.walk} steps):")
        FR_reached = set()
        for step, lang, word, path in r:
            if lang == "FR":
                FR_reached.add(word)
                prefix = "  " * step
                print(f"{prefix}[{step}] {lang}: {word:20s}  {path}")
        print(f"\n  Reached {len(FR_reached)} unique FR words: {sorted(FR_reached)[:20]}...")
        return

    if args.compose:
        result = compose_phrase(G, args.compose)
        print(f"COMPOSE: {args.compose}")
        print(f"  → {result['composed']}")
        print(f"  coverage: {result['coverage']*100:.0f}%")
        for w, pick in result["picks"].items():
            print(f"  {w:15s} → {pick['fr']:18s}  s={pick['sound']:.2f}  "
                  f"means: {sorted(pick['meaning'])[:5]}")
        return

    # Default: extract all words
    if not args.walk and not args.compose:
        words = extract_all_words(G, min_sound=0.55, min_en_len=3)
        print(f"EXTRACTED: {len(words)} words with sound ≥ 0.55")
        print(f"  Top 30:")
        for s,en,fr,n_good,n_total in words[:30]:
            print(f"  {en:18s} → {fr:18s}  s={s:.3f}  ({n_good}/{n_total})")

if __name__ == "__main__":
    main()
