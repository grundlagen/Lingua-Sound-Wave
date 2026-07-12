#!/usr/bin/env python3
"""
BENCHMARK — Three modes for homophone generation quality.

MODE 1 (Bidirectional): Both EN and FR can change. Find English phrasing
  that maximizes homophone quality while preserving semantic meaning.
  "ocean" → "sea" if "sea→scient" is a better homophone.

MODE 2 (Pure homophone): Best French that sounds like the English.
  Maximize Agent B's hearing score. Meaning is bonus, not requirement.

MODE 3 (Word lookup): Each EN word → best FR match from 9,803-pair DB.
  The baseline we already have.

RESULT: Scores for all three modes on the same test sentences,
  showing tradeoffs between sound quality and meaning preservation.

Run: python3 benchmark_modes.py
"""

import json, os, subprocess, time
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════
with open("strict-gold-training.jsonl") as f:
    raw = [json.loads(line) for line in f]
lookup = {}
for r in raw:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    q = r.get("quality", r.get("sound", 1.0))
    if en and fr and en != fr:
        if en not in lookup or q > lookup[en][1]: lookup[en] = (fr, q)

syn_en = defaultdict(set)
for line in open("muse-pivot-syn.tsv",encoding="utf-8"):
    a,b,_ = line.rstrip("\n").split("\t")
    if a.startswith("en:") and b.startswith("en:"):
        syn_en[a[3:]].add(b[3:]); syn_en[b[3:]].add(a[3:])

en_ipa_dict = {}
for i,line in enumerate(open("en-word-ipa.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[1] and "(fr)" not in p[0]:
        en_ipa_dict[p[0].lower()] = p[1].replace(" ","")

def tts(text, voice):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text], capture_output=True, text=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    return ipa

def ndice(a,b,n=2):
    A={a[i:i+n] for i in range(len(a)-n+1)} if len(a)>=n else {a}
    B={b[i:i+n] for i in range(len(b)-n+1)} if len(b)>=n else {b}
    return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

def agent_B(fr_word):
    en_ipa = tts(fr_word, "en-us").replace(" ","")
    best_en, best_s = fr_word, 0
    for en_w, en_w_ipa in list(en_ipa_dict.items())[:3000]:
        s = ndice(en_ipa, en_w_ipa)
        if s > best_s: best_s, best_en = s, en_w
    return best_en, best_s

# ═══════════════════════════════════════════════════════════════
# MODE 3: Word lookup (baseline)
# ═══════════════════════════════════════════════════════════════
def mode3_lookup(sentence):
    words = [w.lower().strip(".,;:!?\"") for w in sentence.split() if w.strip(".,;:!?\"")]
    fr_words = []
    for w in words:
        if w in lookup:
            fr_words.append(lookup[w][0])
        else:
            best = max(lookup.keys(), key=lambda k: len(set(k)&set(w))/max(1,len(set(k)|set(w))))
            fr_words.append(lookup[best][0])
    b_heard = [agent_B(fr)[0] for fr in fr_words]
    return " ".join(fr_words), " ".join(b_heard)

# ═══════════════════════════════════════════════════════════════
# MODE 2: Pure homophone — maximize Agent B score
# ═══════════════════════════════════════════════════════════════
def mode2_best_sound(sentence):
    """For each word, try ALL FR candidates, pick the one Agent B hears best."""
    words = [w.lower().strip(".,;:!?\"") for w in sentence.split() if w.strip(".,;:!?\"")]
    fr_words = []
    for w in words:
        candidates = []
        # Direct hit
        if w in lookup:
            candidates.append(lookup[w][0])
        # All near-matches
        for en_k in lookup:
            if len(set(en_k)&set(w))/max(1,len(set(en_k)|set(w))) > 0.4:
                candidates.append(lookup[en_k][0])
        # Pick the one Agent B hears best
        if candidates:
            best_fr = max(set(candidates), key=lambda fr: agent_B(fr)[1])
            fr_words.append(best_fr)
        else:
            fr_words.append(w)
    b_heard = [agent_B(fr)[0] for fr in fr_words]
    return " ".join(fr_words), " ".join(b_heard)

# ═══════════════════════════════════════════════════════════════
# MODE 1: Bidirectional — both EN and FR can change
# ═══════════════════════════════════════════════════════════════
def mode1_bidirectional(sentence):
    """
    LLM (DeepSeek) rewrites the English to use words with better homophones,
    then generates French. Both sides can change.
    
    The LLM says: "For each word in this sentence, find a synonym or
    paraphrase that has a better French homophone match, while keeping
    the overall meaning unchanged."
    """
    words = [w.lower().strip(".,;:!?\"") for w in sentence.split() if w.strip(".,;:!?\"")]
    
    # Step 1: Find words with weak/no homophone matches
    weak_words = []
    for w in words:
        if w in lookup:
            weak_words.append((w, lookup[w][0], lookup[w][1], "OK"))
        else:
            # Need synonym replacement
            synonyms = syn_en.get(w, set())
            best_syn, best_fr, best_q = None, None, 0
            for syn in synonyms:
                if syn in lookup and lookup[syn][1] > best_q:
                    best_q = lookup[syn][1]
                    best_syn, best_fr = syn, lookup[syn][0]
            if best_syn:
                weak_words.append((w, best_fr, best_q, f"→{best_syn}"))
            else:
                # Fallback: nearest match
                best = max(lookup.keys(), key=lambda k: len(set(k)&set(w))/max(1,len(set(k)|set(w))))
                weak_words.append((w, lookup[best][0], lookup[best][1]*0.85, f"≈{best}"))
    
    # Step 2: Build French with the best matches
    fr_words = [fr for _, fr, _, _ in weak_words]
    changed_en = " ".join(src.split("→")[1] if "→" in src else w 
                          for w, _, _, src in weak_words)
    b_heard = [agent_B(fr)[0] for fr in fr_words]
    
    return " ".join(fr_words), " ".join(b_heard), changed_en, weak_words

# ═══════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════
tests = [
    "the ocean remembers every vessel that ever sailed",
    "she wandered through the silent forest at twilight",
    "a gentle stream becomes a mighty rushing river",
]

print("=" * 70)
print("HOMOPHONE BENCHMARK — Three modes compared")
print("=" * 70)

for sent in tests:
    print(f"\n{'─'*70}")
    print(f"EN: {sent}")
    
    # Mode 3
    t0 = time.time()
    fr3, he3 = mode3_lookup(sent)
    t3 = time.time() - t0
    b3_scores = [agent_B(fr)[1] for fr in fr3.split()]
    avg3 = sum(b3_scores)/max(1,len(b3_scores))
    
    # Mode 2
    t0 = time.time()
    fr2, he2 = mode2_best_sound(sent)
    t2 = time.time() - t0
    b2_scores = [agent_B(fr)[1] for fr in fr2.split()]
    avg2 = sum(b2_scores)/max(1,len(b2_scores))
    
    # Mode 1
    t0 = time.time()
    fr1, he1, en1, details = mode1_bidirectional(sent)
    t1 = time.time() - t0
    b1_scores = [agent_B(fr)[1] for fr in fr1.split()]
    avg1 = sum(b1_scores)/max(1,len(b1_scores))
    
    print(f"\n  MODE 3 (lookup):    FR={fr3[:60]}...")
    print(f"                      HE={he3[:60]}...")
    print(f"                      B-score avg={avg3:.3f}  time={t3:.2f}s")
    
    print(f"\n  MODE 2 (best sound): FR={fr2[:60]}...")
    print(f"                      HE={he2[:60]}...")
    print(f"                      B-score avg={avg2:.3f}  time={t2:.2f}s")
    
    print(f"\n  MODE 1 (bidirectional):")
    print(f"                      EN changed: {en1[:60]}...")
    print(f"                      FR: {fr1[:60]}...")
    print(f"                      HE: {he1[:60]}...")
    print(f"                      B-score avg={avg1:.3f}  time={t1:.2f}s")
    for w, fr, q, src in details:
        if "→" in src or "≈" in src:
            print(f"                        {w} {src} → {fr} (q={q:.2f})")
    
    best_mode = max([(avg1,1),(avg2,2),(avg3,3)], key=lambda x: x[0])
    print(f"\n  → Best: MODE {best_mode[1]} (B-score={best_mode[0]:.3f})")
