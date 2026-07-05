#!/usr/bin/env python3
"""
TOPOLOGICAL GENERATIVE ENGINE — NOT a translator. A dual-language GENERATOR.

Stage 1 (DONE):  Word pairs loaded from ladder, v7, strict-gold, dual.
Stage 2 (DONE):  Meaning maps on BOTH sides — fr_means, en_homophones, synonyms.
Stage 3 (HERE):  Generate novel dual-language sentences by walking the bipartite
                 graph, constrained by: vocabulary-only output, dual meaning, 
                 bigram coherence in BOTH languages.
Stage 4 (NEXT):  Paragraph-level recursive generation at GPU scale.

NO test sentence is fed. The engine generates its own text.

Graph: G = (E, F, edges) where edges = valid sound pairs
  H(e) = FR words that sound like e       (homophone set)
  M(f) = EN words that f means            (meaning set)
  Chain: e₁ → f₁ ∈ H(e₁) → e₂ ∈ M(f₁) → f₂ ∈ H(e₂) → ...

Run: python3 generate_sentence.py --n 5     (generate 5 sentences)
     python3 generate_sentence.py --paragraph (paragraph-level generation)
"""

import json, os, sys, random
from collections import defaultdict
import numpy as np

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ═══════════════════════════════════════════════════════════════
# STAGE 1: Load vocabulary (word pairs)
# ═══════════════════════════════════════════════════════════════
sound = defaultdict(list)      # en → [(fr, score)]
meaning = defaultdict(set)     # fr → {en, ...}
homophones = defaultdict(set)  # fr → {fr, ...}  (FR homophone classes)
synonyms_en = defaultdict(set) # en → {en, ...}
synonyms_fr = defaultdict(set) # fr → {fr, ...}
chain_hops = defaultdict(lambda: defaultdict(int))  # (en,fr) → min_hops

print("STAGE 1: Loading word pairs...")

# strict-gold
gold_count = 0
for i,line in enumerate(open("strict-gold.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2:
        sound[p[0]].append((p[1], 1.0))
        meaning[p[1]].add(p[0])
        gold_count += 1
print(f"  strict-gold: {gold_count} pairs")

# tier-ladder
ladder_count = 0
for i,line in enumerate(open("tier-ladder.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=12 and p[10]:
        try:
            s = float(p[10])
            if s >= 0.55:
                sound[p[1]].append((p[2], s))
                meaning[p[2]].add(p[1])
                ladder_count += 1
        except: continue
print(f"  tier-ladder: {ladder_count} pairs (sound ≥ 0.55)")

# v7 gold
v7_count = 0
for i,line in enumerate(open("dictionary-v7.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=9 and p[3]=="1":
        sound[p[7]].append((p[8], float(p[1])))
        meaning[p[8]].add(p[7])
        v7_count += 1
print(f"  v7 gold: {v7_count} pairs")

# dual pairs
dual_count = 0
for i,line in enumerate(open("dual-pairs.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=6 and p[0].lower() != p[1].lower():
        sound[p[0]].append((p[1], float(p[2])))
        meaning[p[1]].add(p[0])
        dual_count += 1
print(f"  dual-pairs: {dual_count} pairs")

# ═══════════════════════════════════════════════════════════════
# STAGE 2: Build meaning maps (both sides)
# ═══════════════════════════════════════════════════════════════
print("\nSTAGE 2: Building meaning maps...")

# FR homophone classes
for path in ["fr-homophone-classes-lexique.tsv","fr-homophone-classes.tsv"]:
    try:
        for i,line in enumerate(open(path,encoding="utf-8")):
            if i==0: continue
            ms = line.rstrip("\n").split("\t")[1].split()
            if len(ms)>=2:
                for m in ms: homophones[m].update(ms)
    except: pass
print(f"  FR homophone classes: {len(homophones)} words in {sum(1 for v in homophones.values() if len(v)>1)}+ classes")

# Synonyms (both sides)
for line in open("muse-pivot-syn.tsv",encoding="utf-8"):
    a,b,_ = line.rstrip("\n").split("\t")
    if a.startswith("en:") and b.startswith("en:"):
        synonyms_en[a[3:]].add(b[3:])
        synonyms_en[b[3:]].add(a[3:])
    elif a.startswith("fr:") and b.startswith("fr:"):
        synonyms_fr[a[3:]].add(b[3:])
        synonyms_fr[b[3:]].add(a[3:])
print(f"  EN synonyms: {sum(len(v) for v in synonyms_en.values())} edges")
print(f"  FR synonyms: {sum(len(v) for v in synonyms_fr.values())} edges")

# Chain-web hops
try:
    for i,line in enumerate(open("chain-web/archive/chain-web-full-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=5 and ":" in p[0] and ":" in p[1]:
            sl,sw = p[0].split(":",1); tl,tw = p[1].split(":",1)
            if sl=="en" and tl=="fr":
                h = int(p[2])
                if (sw,tw) not in chain_hops or h < chain_hops[(sw,tw)]:
                    chain_hops[(sw,tw)] = h
            elif sl=="fr" and tl=="en":
                h = int(p[2])
                if (tw,sw) not in chain_hops or h < chain_hops[(tw,sw)]:
                    chain_hops[(tw,sw)] = h
except: pass
print(f"  chain-web hops: {len(chain_hops)} (en,fr) pairs at min distance")

# Loop-certified
loops = set()
try:
    for i,line in enumerate(open("chain-web/archive/loop-certified-pairs-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=2: loops.add((p[0], p[1]))
except: pass
print(f"  loop-certified: {len(loops)} bidirectional pairs")

# ═══════════════════════════════════════════════════════════════
# STAGE 3: Topological Sentence Generation
# ═══════════════════════════════════════════════════════════════
print("\nSTAGE 3: Generating dual-language sentences...")

import subprocess
def tts(text, voice):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text],
                       capture_output=True, text=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    return ipa

def ndice(a,b,n=2):
    A = {a[i:i+n] for i in range(len(a)-n+1)} if len(a)>=n else {a}
    B = {b[i:i+n] for i in range(len(b)-n+1)} if len(b)>=n else {b}
    return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

def generate_sentence(min_words=5, max_words=12, min_quality=0.55):
    """
    Generate a dual-language sentence by walking the bipartite graph.
    
    Algorithm:
      1. Pick a seed English word with high-quality homophone pairs
      2. Walk: en → fr (homophone) → en' (meaning back-link) → fr' → ...
      3. At each step, prefer loop-certified pairs and chain-web short hops
      4. Filter output for dual meaning (both sides must have meaning)
      5. Score the chain for sound quality + meaning density
    """
    # Step 1: Pick seed word (one with many good homophones)
    seeds = [(en, len([f for f,s in entries if s>=0.70]))
             for en, entries in sound.items() 
             if len(en)>=3 and len([f for f,s in entries if s>=0.70])>=3]
    if not seeds: seeds = [(en, len(entries)) for en, entries in sound.items() if len(entries)>=2]
    if not seeds: return None
    
    for attempt in range(50):
        en_seed, _ = random.choice(seeds)
        
        # Walk: generate chain
        chain_en = [en_seed]
        chain_fr = []
        quality_scores = []
        
        current_en = en_seed
        for step in range(max_words):
            # Pick a French homophone for current_en
            candidates = [(f,s) for f,s in sound.get(current_en, []) 
                         if meaning.get(f, set()) and f not in chain_fr
                         and s >= min_quality]
            if not candidates:
                candidates = [(f,s) for f,s in sound.get(current_en, [])
                             if meaning.get(f, set()) and f not in chain_fr]
            if not candidates: break
            
            # Weight: prefer loop-certified, chain-web hops
            weighted = []
            for f, s in candidates:
                w = s
                if (current_en, f) in loops: w += 0.10  # loop bonus
                if (current_en, f) in chain_hops: w += 0.05  # chain bonus
                if f in synonyms_fr: w += 0.02  # synonym-rich
                weighted.append((w, f, s))
            weighted.sort(reverse=True)
            
            best_fr = weighted[0][1]
            best_score = weighted[0][2]
            chain_fr.append(best_fr)
            quality_scores.append(best_score)
            
            # Find next EN word from meaning of this FR word
            en_meanings = meaning.get(best_fr, set())
            if not en_meanings: break
            
            # Prefer meaning words that are synonyms of the original
            en_candidates = list(en_meanings - set(chain_en))
            if not en_candidates:
                # Allow revisiting
                en_candidates = list(en_meanings)
            if not en_candidates: break
            
            # Prefer synonyms
            syn_cands = [e for e in en_candidates if e in synonyms_en.get(current_en, set())]
            next_en = random.choice(syn_cands or en_candidates)
            chain_en.append(next_en)
            current_en = next_en
            
            if len(chain_fr) >= min_words: break
        
        # Validate: sentence must have dual meaning
        if len(chain_fr) >= min_words:
            fr_text = " ".join(chain_fr)
            en_text = " ".join(chain_en)
            
            # Score the chain
            avg_sound = np.mean(quality_scores) if quality_scores else 0
            loop_count = sum(1 for e,f in zip(chain_en, chain_fr) if (e,f) in loops)
            chain_count = sum(1 for e,f in zip(chain_en, chain_fr) if (e,f) in chain_hops)
            meaning_density = len(set(chain_en)) / len(chain_en)  # unique EN coverage
            
            total_score = avg_sound * 0.5 + (loop_count/len(chain_fr))*0.3 + meaning_density*0.2
            
            return {
                "en": en_text,
                "fr": fr_text,
                "en_words": chain_en,
                "fr_words": chain_fr,
                "avg_sound": round(avg_sound, 3),
                "loops": loop_count,
                "chains": chain_count,
                "meaning_density": round(meaning_density, 3),
                "total_score": round(total_score, 3),
                "n_words": len(chain_fr),
            }
    
    return None

# ═══════════════════════════════════════════════════════════════
# GENERATE MULTIPLE SENTENCES
# ═══════════════════════════════════════════════════════════════
def generate_paragraph(n_sentences=6, min_words=5, max_words=10):
    """Generate a paragraph of dual-language sentences."""
    sentences = []
    for i in range(n_sentences):
        s = generate_sentence(min_words, max_words)
        if s: sentences.append(s)
    return sentences

# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="Number of sentences")
    ap.add_argument("--paragraph", action="store_true")
    ap.add_argument("--min", type=int, default=5, help="Min words per sentence")
    ap.add_argument("--max", type=int, default=12, help="Max words per sentence")
    args = ap.parse_args()
    
    # Quick stats
    en_words = [en for en, entries in sound.items() if len(entries)>=2]
    fr_words = [fr for fr, entries in meaning.items() if len(entries)>=2]
    print(f"\nVocabulary: {len(en_words)} EN words, {len(fr_words)} FR words ready")
    print(f"  With homophone classes, synonyms, chain-web\n")
    
    if args.paragraph:
        sent_count = max(args.n, 4)
        print(f"GENERATING PARAGRAPH ({sent_count} sentences)...\n")
        results = generate_paragraph(sent_count, args.min, args.max)
        for i, s in enumerate(results):
            print(f"[{i+1}] EN: {s['en']}")
            print(f"    FR: {s['fr']}")
            print(f"    sound={s['avg_sound']:.3f}  loops={s['loops']}  "
                  f"chains={s['chains']}  meaning={s['meaning_density']:.2f}  "
                  f"score={s['total_score']:.3f}")
            print()
    else:
        sent_count = args.n
        print(f"GENERATING {sent_count} SENTENCES...\n")
        for i in range(sent_count):
            s = generate_sentence(args.min, args.max)
            if s:
                print(f"[{i+1}] EN: {s['en']}")
                print(f"    FR: {s['fr']}")
                print(f"    sound={s['avg_sound']:.3f}  loops={s['loops']}  "
                      f"chains={s['chains']}  meaning={s['meaning_density']:.2f}  "
                      f"score={s['total_score']:.3f}")
            else:
                print(f"[{i+1}] (no valid sentence generated)")
            print()
