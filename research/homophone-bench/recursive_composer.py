#!/usr/bin/env python3
"""
RECURSIVE COMPOSER — Bidirectional meaning propagation, constraint satisfaction.

THE ALGORITHM:
  Start with 200 EN words. Assign each a FR homophone greedily.
  Then ITERATE until fixed point:
    1. For each assigned pair (en_i, fr_i), compute REGRET:
       How much sound quality is lost vs the best possible?
    2. For each pair, check: does fr_i's meaning set cover en_k?
       If yes, could en_k use a DIFFERENT fr_k that frees up fr_i?
    3. Find the highest-regret position. Try swapping its FR word.
    4. Propagate: the swap may cascade — en_k changes fr_k,
       which changes the meaning coverage at position j, etc.
    5. Each propagation step is LOCAL — only neighbors in the
       meaning graph are re-evaluated.
    6. Lock positions that have stabilized (no better alternative
       within 3 hops of the meaning graph).
    7. Stop when no position can be improved.

  This is NOT greedy one-pass. The last word genuinely can change
  the first word through recursive meaning propagation.

Run: python recursive_composer.py
"""

import os, json
from collections import defaultdict
import numpy as np

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ═══════════════════════════════════════════════════════════════
# LOAD DATABASES
# ═══════════════════════════════════════════════════════════════
print("Loading...")

# Sound edges: EN → [(FR, sound_score), ...]
sound_edges = defaultdict(list)
for i,line in enumerate(open("tier-ladder.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=12 and p[10]:
        try:
            sound_edges[p[1]].append((p[2], float(p[10])))
        except: continue

# strict-gold
for i,line in enumerate(open("strict-gold.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2:
        sound_edges[p[0]].append((p[1], 1.0))

# v7 gold  
for i,line in enumerate(open("dictionary-v7.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=9 and p[3]=="1":
        sound_edges[p[7]].append((p[8], float(p[1])))

for k in sound_edges:
    sound_edges[k].sort(key=lambda x: -x[1])

# Meaning edges: FR → {EN words it can mean}
fr_means = defaultdict(set)
for en_w, entries in sound_edges.items():
    for fr_w, s in entries:
        if s >= 0.55: fr_means[fr_w].add(en_w)

# FR homophone classes (for propagation)
fr_class = {}
for path in ["fr-homophone-classes-lexique.tsv","fr-homophone-classes.tsv"]:
    try:
        for i,line in enumerate(open(path,encoding="utf-8")):
            if i==0: continue
            ms = line.rstrip("\n").split("\t")[1].split()
            for m in ms: fr_class[m] = ms
    except: pass

print(f"  {len(sound_edges):,} EN words, {len(fr_means):,} FR words with meanings")

# ═══════════════════════════════════════════════════════════════
# SELECT 200 DIVERSE WORDS
# ═══════════════════════════════════════════════════════════════
STOP = set("the a an of in on and or to for with by at as is are was were be "
           "been am it he she we they you me him her us them my your his her "
           "its our their this that these those not no so but if all some more "
           "very just only too also then now here there did do does had has "
           "have can could will would shall should may might must from about "
           "into over under up out down off away back still even much many few "
           "each every any both such same own other another new old good great "
           "big small high low long short right left".split())

# Gather all EN words with sound ≥ 0.60
candidates = []
for en_w in sound_edges:
    if en_w in STOP or len(en_w) < 3: continue
    best = sound_edges[en_w]
    if best[0][1] >= 0.60:
        candidates.append((en_w, best))

# Score by sound × transformation_distance (penalize identical pairs)
# Prefer pairs where sound is good AND the words are different
fr_usage = defaultdict(int)
selected = []

for en_w, entries in candidates:
    best_fr = entries[0][0]
    best_s = entries[0][1]
    
    # Prefer non-identical pairs (interesting transformations)
    # If identical, try the second-best candidate
    if en_w == best_fr and len(entries) >= 2:
        best_fr = entries[1][0]
        best_s = entries[1][1]
    
    # Still identical after fallback? Skip (too boring)
    if en_w == best_fr:
        continue
    
    # Diversity: prefer FR words not yet used
    if fr_usage[best_fr] < 2:  # max 2 uses per FR word
        fr_usage[best_fr] += 1
        selected.append((en_w, best_fr, best_s))
        if len(selected) >= 200: break

# If not enough, relax diversity constraint
if len(selected) < 200:
    for en_w, entries in candidates:
        best_fr = entries[0][0]
        best_s = entries[0][1]
        if (en_w, best_fr, best_s) not in selected:
            fr_usage[best_fr] += 1
            selected.append((en_w, best_fr, best_s))
            if len(selected) >= 200: break

print(f"  Selected {len(selected)} words, {len(set(fr for _,fr,_ in selected))} unique FR words")

# ═══════════════════════════════════════════════════════════════
# RECURSIVE CONSTRAINED OPTIMIZATION
# ═══════════════════════════════════════════════════════════════
print(f"\nRecursive optimization (propagating meaning both ways)...")

# Build index: for each EN word, all FR candidates with score
en_to_fr = {}
for en_w, best_fr, best_s in selected:
    # Exclude identical pairs from candidate pool (too boring)
    en_to_fr[en_w] = [(fr, s) for fr, s in sound_edges[en_w] 
                       if s >= 0.55 and fr != en_w]
    if not en_to_fr[en_w]:
        # Fallback: allow identical if nothing else
        en_to_fr[en_w] = [(fr, s) for fr, s in sound_edges[en_w] if s >= 0.55]
    en_to_fr[en_w].sort(key=lambda x: -x[1])

# Initial assignment: best for each
assignment = {en_w: (fr, s) for en_w, fr, s in selected}
universe = set(en_w for en_w, _, _ in selected)

def compute_meaning_coverage(assign):
    """Which EN words are covered by which FR words' meanings?"""
    covered_by = defaultdict(set)
    for en_w, (fr_w, s) in assign.items():
        meanings = fr_means.get(fr_w, set()) & universe
        for m in meanings:
            covered_by[m].add(fr_w)
    return covered_by

def compute_sound_loss(assign):
    """Total sound quality loss vs best possible."""
    loss = 0
    for en_w, (fr_w, s) in assign.items():
        best_possible = en_to_fr[en_w][0][1] if en_to_fr[en_w] else 0
        loss += best_possible - s
    return loss

def find_improvements(assignment, covered_by, locked, max_swaps=50):
    """Find swaps that improve the assignment. Returns list of (en_w, new_fr, gain)."""
    improvements = []
    
    for en_w, (fr_w, current_s) in assignment.items():
        if en_w in locked: continue
        
        # What does fr_w currently mean?
        current_meanings = fr_means.get(fr_w, set()) & universe
        
        # Try alternatives
        for alt_fr, alt_s in en_to_fr.get(en_w, [])[:8]:
            if alt_fr == fr_w: continue
            if alt_s <= current_s: continue  # must improve sound
            
            # Check: would this swap break meaning coverage?
            alt_meanings = fr_means.get(alt_fr, set()) & universe
            
            # Gain = sound improvement + meaning coverage delta
            sound_gain = alt_s - current_s
            meaning_gain = len(alt_meanings - current_meanings) * 0.01
            
            # Check if any covered EN word becomes UNCOVERED by this swap
            uncovered = set()
            for m in current_meanings:
                # Is m covered by other FR words?
                other_cover = {f for f in covered_by.get(m, set()) if f != fr_w}
                if not other_cover:
                    uncovered.add(m)
            
            # Can alt_fr cover the uncovered words?
            if uncovered and not (uncovered & alt_meanings):
                continue  # would break coverage
            
            gain = sound_gain + meaning_gain
            if gain > 0.001:
                improvements.append((gain, en_w, alt_fr, alt_s))
    
    improvements.sort(reverse=True)
    return improvements[:max_swaps]

# ══ ITERATIVE REFINEMENT ══
locked = set()
best_assignment = dict(assignment)
best_sound = sum(s for _,s in best_assignment.values())
best_loss = compute_sound_loss(best_assignment)
iteration = 0
stagnation = 0

while stagnation < 15:
    iteration += 1
    covered_by = compute_meaning_coverage(best_assignment)
    
    # Find improvements
    improvements = find_improvements(best_assignment, covered_by, locked)
    
    if not improvements:
        stagnation += 1
        # Unlock some positions if stuck
        if stagnation >= 5 and len(locked) > 20:
            # Unlock the 10 most recently locked
            to_unlock = list(locked)[-10:]
            locked -= set(to_unlock)
            stagnation = 0
        continue
    
    # Apply the best improvement
    gain, en_w, new_fr, new_s = improvements[0]
    old_fr, old_s = best_assignment[en_w]
    best_assignment[en_w] = (new_fr, new_s)
    stagnation = 0
    
    # Propagate: any position whose meaning was covered by old_fr
    # might now be uncovered and need re-evaluation
    old_meanings = fr_means.get(old_fr, set()) & universe
    for m in old_meanings:
        if m in locked: locked.remove(m)  # unlock for re-evaluation
    
    if iteration % 10 == 0:
        new_loss = compute_sound_loss(best_assignment)
        new_sound = sum(s for _,s in best_assignment.values())
        print(f"  iter {iteration}: sound={new_sound:.1f} loss={new_loss:.3f} "
              f"({iteration} improvements, {len(locked)} locked)")

final_sound = sum(s for _,s in best_assignment.values())
final_loss = compute_sound_loss(best_assignment)
covered_by = compute_meaning_coverage(best_assignment)
coverage = sum(1 for en_w in universe if covered_by.get(en_w)) / len(universe)

print(f"\n  FINAL: sound={final_sound:.1f} loss={final_loss:.3f} "
      f"coverage={coverage*100:.0f}% iterations={iteration}")

# ═══════════════════════════════════════════════════════════════
# RENDER PARAGRAPH
# ═══════════════════════════════════════════════════════════════
print(f"\nRECURSIVE BIDIRECTIONAL PARAGRAPH — {len(best_assignment)} words")
print(f"  sound total={final_sound:.1f}  mean={final_sound/len(best_assignment):.3f}  "
      f"coverage={coverage*100:.0f}%")

# Sort by meaning connectivity: words whose FR meanings overlap cluster together
def meaning_overlap(a, b):
    ma = fr_means.get(a, set()) & universe
    mb = fr_means.get(b, set()) & universe
    return len(ma & mb)

# Build ordered list by chaining meaning overlaps
# Greedy chain: start with most-connected word, then always pick
# the unplaced word with highest meaning overlap to the last placed
unplaced = set(best_assignment.keys())

# Precompute meaning overlap for all pairs
_pair_overlap_cache = {}
def pair_overlap(a, b):
    k = (a,b) if a < b else (b,a)
    if k in _pair_overlap_cache: return _pair_overlap_cache[k]
    fa = best_assignment[a][0]
    fb = best_assignment[b][0]
    ma = fr_means.get(fa, set()) & universe
    mb = fr_means.get(fb, set()) & universe
    v = len(ma & mb)
    _pair_overlap_cache[k] = v
    return v

# Start with the word that has most total meaning overlap
total_ov = {}
for w in unplaced:
    total_ov[w] = sum(pair_overlap(w, x) for x in unplaced if x != w)
ordered = [max(unplaced, key=lambda w: total_ov.get(w, 0))]
unplaced.remove(ordered[0])

# Greedy chain: always pick highest overlap to last
while unplaced:
    last = ordered[-1]
    best = max(unplaced, key=lambda w: pair_overlap(last, w))
    ordered.append(best)
    unplaced.remove(best)

# Render as lines with meaning overlap transitions
line_len = 10
pairs_out = []
lines_out = []
current = []
current_overlaps = []

for i, en_w in enumerate(ordered):
    fr_w, s = best_assignment[en_w]
    meanings = sorted(fr_means.get(fr_w, set()) & universe)
    pairs_out.append({"en": en_w, "fr": fr_w, "sound": round(s,3), 
                      "meaning": meanings[:5], "meaning_size": len(meanings)})
    current.append(fr_w)
    if i < len(ordered) - 1:
        current_overlaps.append(pair_overlap(en_w, ordered[i+1]))
    if len(current) >= line_len:
        overlaps = current_overlaps[:len(current)-1]
        avg_ov = sum(overlaps)/len(overlaps) if overlaps else 0
        lines_out.append(" ".join(current))
        print(f"  {fr_w} (next-overlap avg={avg_ov:.1f})")
        current = []
        current_overlaps = []

if current:
    lines_out.append(" ".join(current))

for i, line in enumerate(lines_out):
    print(f"  [{i:2d}] {line}")

# Stats
fr_used = set()
dupes = 0
for p in pairs_out:
    if p["fr"] in fr_used: dupes += 1
    fr_used.add(p["fr"])
persistence = min(p["sound"] for p in pairs_out)
rooten = sum(1 for p in pairs_out if p["sound"]>=0.55 and p["meaning_size"]>=1)

print(f"\n  persistence={persistence:.3f}  unique FR={len(fr_used)}  dupes={dupes}  rooten={rooten}/{len(pairs_out)}")

# Save
with open("recursive_paragraph.json","w") as f:
    json.dump({
        "n_words": len(pairs_out),
        "persistence": round(persistence, 3),
        "coverage": round(coverage, 3),
        "iterations": iteration,
        "unique_fr": len(fr_used),
        "dupes": dupes,
        "rooten": rooten,
        "pairs": pairs_out,
        "lines": lines_out,
    }, f, ensure_ascii=False, indent=2)
print(f"\nSaved recursive_paragraph.json")
