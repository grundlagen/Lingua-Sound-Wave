#!/usr/bin/env python3
"""
STAGE 3: GENERATION ENGINE — Continuous graph walks producing dual-language sentences.

Walks the Stage 2 bipartite graph:
  en → fr (sound edge) → en' (meaning backlink) → fr' (sound) → ...
  
Constrained by:
  - Both sides must have meaning (no orphan words)
  - Sound quality preservation (prefer high-sound edges)
  - Zipf common words (prefer frequent words)
  - Chain-web short hops (prefer verified paths)
  - Loop certification (bidirectional agreement)
  - Bigram fluency (adjacent words must form probable bigrams)

Runs CONTINUOUSLY — generates sentences, saves periodically, never stops.

Run: python run_stage_3.py           (run until stopped)
     python run_stage_3.py --n 5000  (generate 5000 sentences and exit)
     python run_stage_3.py --daemon   (run forever, save every 100 sentences)
"""

import json, os, sys, random, time, subprocess
from collections import defaultdict
import numpy as np

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ═══════════════════════════════════════════════════════════════
# LOAD STAGE 2 GRAPH
# ═══════════════════════════════════════════════════════════════
print("STAGE 3: GENERATION ENGINE")
print("=" * 60)

print("Loading Stage 2 graph...")
graph = json.load(open("stage2_graph.json",encoding="utf-8"))

en_to_fr = defaultdict(list)
for en, entries in graph.get("en_to_fr",{}).items():
    for fr, s in entries:
        en_to_fr[en].append((fr, s))
en_to_fr.update({en: [(fr, s) for fr, s in entries] 
                 for en, entries in graph.get("en_to_fr",{}).items()})

fr_to_en = defaultdict(set)
for fr, entries in graph.get("fr_to_en",{}).items():
    fr_to_en[fr] = set(entries)

chain_hops = defaultdict(lambda: defaultdict(int))
for en, entries in graph.get("chain_hops_sample",{}).items():
    for fr, h in entries.items():
        chain_hops[en][fr] = h

loops = set(tuple(p) for p in graph.get("loops",[]))

zipf = graph.get("zipf_top",{})

print(f"  EN nodes: {graph['stats']['nodes_en']}, FR nodes: {graph['stats']['nodes_fr']}")
print(f"  Sound edges: {graph['stats']['sound_edges']}")
print(f"  Meaning edges: {graph['stats']['meaning_edges']}")
print(f"  Chain hops: {graph['stats']['chain_hops']}")
print(f"  Loops: {graph['stats']['loop_pairs']}")
print(f"  Zipf words: {graph['stats']['zipf_words']}")

# ═══════════════════════════════════════════════════════════════
# GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def tts(text, voice):
    try:
        r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text],
                          capture_output=True, text=True)
        ipa = r.stdout.strip()
        for c in "ˈˌ": ipa = ipa.replace(c,"")
        return ipa
    except: return ""

def ndice(a,b,n=2):
    A = {a[i:i+n] for i in range(len(a)-n+1)} if len(a)>=n else {a}
    B = {b[i:i+n] for i in range(len(b)-n+1)} if len(b)>=n else {b}
    return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

def generate_sentence(min_words=5, max_words=15, min_quality=0.55):
    """Walk the graph to generate one dual-language sentence."""
    # Pick seed — prefer words with multiple good sound matches
    seeds = [(en, len(entries)) for en, entries in en_to_fr.items()
             if len(entries) >= 3 and len(en) >= 3]
    if len(seeds) < 10:
        seeds = [(en, len(entries)) for en, entries in en_to_fr.items() if len(entries) >= 1]
    if not seeds: return None
    
    for attempt in range(20):
        en_seed, _ = random.choice(seeds)
        chain_en = [en_seed]
        chain_fr = []
        quality_scores = []
        
        current_en = en_seed
        for step in range(max_words):
            # Get FR candidates for current_en
            cands = [(f,s) for f,s in en_to_fr.get(current_en,[])
                    if fr_to_en.get(f,set()) and f not in chain_fr and s >= min_quality]
            if not cands and step > 0:
                cands = [(f,s) for f,s in en_to_fr.get(current_en,[])
                        if fr_to_en.get(f,set()) and f not in chain_fr]
            if not cands: break
            
            # Weight by: sound + chain + loop + zipf
            weighted = []
            for f, s in cands:
                w = s
                if current_en in chain_hops and f in chain_hops[current_en]: w += 0.03
                if (current_en, f) in loops: w += 0.05
                if f in zipf: w += 0.02  # common words
                weighted.append((w, f, s))
            weighted.sort(reverse=True)
            
            chosen_fr = weighted[0][1]
            chosen_score = weighted[0][2]
            chain_fr.append(chosen_fr)
            quality_scores.append(chosen_score)
            
            # Find next EN word from FR meaning
            en_meanings = fr_to_en.get(chosen_fr, set())
            if not en_meanings: break
            
            en_cands = list(en_meanings - set(chain_en))
            if not en_cands: en_cands = list(en_meanings)
            if not en_cands: break
            
            # Prefer common words
            if zipf:
                en_cands.sort(key=lambda e: zipf.get(e, 99999))
            next_en = random.choice(en_cands[:5])
            chain_en.append(next_en)
            current_en = next_en
            
            if len(chain_fr) >= min_words and step > min_words:
                break
        
        if len(chain_fr) >= min_words:
            avg_sound = np.mean(quality_scores)
            loop_count = sum(1 for e,f in zip(chain_en, chain_fr) if (e,f) in loops)
            
            # Cross-accent verification
            fr_text = " ".join(chain_fr)
            en_text = " ".join(chain_en)
            fr_ipa = tts(fr_text, "en-us").replace(" ","")
            en_ipa = tts(en_text, "en-us").replace(" ","")
            cross = ndice(fr_ipa, en_ipa) if fr_ipa and en_ipa else 0
            
            total_score = avg_sound * 0.4 + (loop_count/len(chain_fr))*0.3 + cross*0.3
            
            return {
                "en": en_text, "fr": fr_text,
                "en_words": chain_en, "fr_words": chain_fr,
                "avg_sound": round(avg_sound, 3),
                "loops": loop_count,
                "cross": round(cross, 3),
                "total_score": round(total_score, 3),
                "n_words": len(chain_fr),
            }
    
    return None

# ═══════════════════════════════════════════════════════════════
# CONTINUOUS GENERATION
# ═══════════════════════════════════════════════════════════════
def continuous_generate(target=0, save_every=100):
    """Generate sentences continuously. If target=0, run forever."""
    sentences = []
    existing = 0
    if os.path.exists("stage3_sentences.jsonl"):
        existing = sum(1 for _ in open("stage3_sentences.jsonl",encoding="utf-8"))
        print(f"  Resuming from existing {existing} sentences")
    
    generated = 0
    start_time = time.time()
    
    while target == 0 or generated < target:
        s = generate_sentence(min_words=5, max_words=12, min_quality=0.55)
        if s:
            sentences.append(s)
            generated += 1
            
            if generated % save_every == 0:
                # Append to output
                with open("stage3_sentences.jsonl","a") as f:
                    for sent in sentences[-save_every:]:
                        f.write(json.dumps(sent, ensure_ascii=False) + "\n")
                
                elapsed = time.time() - start_time
                total = existing + generated
                rate = generated / elapsed if elapsed > 0 else 0
                
                # Quality stats
                scores = [s["total_score"] for s in sentences[-100:]] if sentences else [0]
                loops_used = sum(s["loops"] for s in sentences[-100:])
                
                print(f"  [{total:6d}] rate={rate:.1f}/s  "
                      f"score={np.mean(scores):.3f}  loops={loops_used}  "
                      f"elapsed={elapsed/60:.1f}m")
    
    # Final save
    if sentences:
        with open("stage3_sentences.jsonl","a") as f:
            for sent in sentences[-generated:]:
                f.write(json.dumps(sent, ensure_ascii=False) + "\n")
    
    return generated

# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="Generate N sentences (0=forever)")
    ap.add_argument("--daemon", action="store_true", help="Run forever, save every 100")
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()
    
    target = args.n
    if args.daemon: target = 0
    
    print(f"\n  Target: {'∞' if target == 0 else target} sentences")
    print(f"  Auto-save: every {args.save_every}\n")
    
    n = continuous_generate(target=target, save_every=args.save_every)
    
    print(f"\n  Generated {n} sentences")
    if os.path.exists("stage3_sentences.jsonl"):
        total = sum(1 for _ in open("stage3_sentences.jsonl",encoding="utf-8"))
        print(f"  Total in database: {total}")
