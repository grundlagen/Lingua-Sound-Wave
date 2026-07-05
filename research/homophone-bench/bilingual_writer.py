#!/usr/bin/env python3
"""
BILINGUAL WRITER — Produces French paragraphs that sound like English.
Uses the 5,293-pair homophone model (nearest-match lookup).

Given English input, each word → closest French homophone from DB.
Output: French text + what an English ear hears.

Run: python bilingual_writer.py "the silent beauty of the endless sea remembers every ship that ever sailed"
"""

import json, os, subprocess, sys
from collections import defaultdict

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ── Load the 9,803-pair STRICT-GOLD model ──
with open("strict-gold-training.jsonl") as f:
    raw = [json.loads(line) for line in f]

# Build: en_word → (fr_word, quality, loop, chain)
model = {}
for r in raw:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    q = r.get("quality", r.get("sound", 1.0))
    loop = r.get("loop", False)
    chain_sup = r.get("chain", False)
    if en and fr and en != fr:
        if en not in model or q > model[en][1]:
            model[en] = (fr, q, loop, chain_sup)

# Also build reverse: fr_word → set of en_words it can serve
fr_to_en = defaultdict(set)
for en, (fr, s, loop, chain_sup) in model.items():
    fr_to_en[fr].add(en)

print(f"Model: {len(model)} EN→FR pairs, {len(fr_to_en)} unique FR words")

# ── Phonetic tools ──
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

# Load EN vocab for reverse lookup
en_vocab = {}
for i,line in enumerate(open("en-word-ipa.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[1] and "(fr)" not in p[0]:
        en_vocab[p[0].lower()] = p[1].replace(" ","")

def word_similarity(a, b):
    """Character overlap for nearest-match lookup."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))

def find_homophone(en_word):
    """Find the best French homophone for an English word."""
    en_word = en_word.lower()
    # Direct lookup
    if en_word in model:
        return model[en_word][0], model[en_word][1], "direct"
    # Nearest match by character overlap
    best = max(model.keys(), key=lambda k: word_similarity(en_word, k))
    return model[best][0], model[best][1], f"near:{best}"

def what_english_ear_hears(fr_word):
    """What English word does this French word sound like?"""
    en_ipa = tts(fr_word, "en-us").replace(" ","")
    best_en, best_score = fr_word, 0
    for en_w, en_w_ipa in list(en_vocab.items())[:3000]:
        s = ndice(en_ipa, en_w_ipa)
        if s > best_score:
            best_score, best_en = s, en_w
    return best_en, best_score

# ═══════════════════════════════════════════════════════════
# WRITE — main bilingual composition
# ═══════════════════════════════════════════════════════════
def write(en_paragraph, verbose=True):
    """Produce a bilingual paragraph from English input."""
    words = [w.lower().strip(".,;:!?'\"") for w in en_paragraph.split() 
             if w.strip(".,;:!?'\"") and len(w) >= 2]
    
    # Step 1: Find French homophone for each English word
    pairs = []
    for w in words:
        fr, score, source = find_homophone(w)
        pairs.append((w, fr, score, source))
    
    # Step 2: Build French paragraph
    fr_paragraph = " ".join(fr for _, fr, _, _ in pairs)
    
    # Step 3: What does the English ear hear?
    en_heard = []
    for w, fr, score, source in pairs:
        heard, h_score = what_english_ear_hears(fr)
        en_heard.append((heard, h_score))
    
    # Step 4: Render
    if verbose:
        print(f"\n{'='*70}")
        print(f"ENGLISH: {en_paragraph}")
        print(f"FRENCH:  {fr_paragraph}")
        print(f"EN-EAR:  {' '.join(h for h,_ in en_heard)}")
        print(f"{'='*70}")
        print(f"\n  {'EN word':15s} {'FR word':15s} {'sounds like':15s} {'score':>6s} {'source'}")
        print(f"  {'─'*15} {'─'*15} {'─'*15} {'─'*6} {'─'*20}")
        for (w, fr, score, source), (heard, h_score) in zip(pairs, en_heard):
            mark = "✓" if score >= 0.70 else "~"
            print(f"  {w:15s} {fr:15s} {heard:15s} {h_score:6.3f} {source:20s}")
    
    return fr_paragraph, en_heard, pairs

# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "the silent beauty of the endless sea remembers every ship that ever sailed"
    
    write(text)
    
    # Also test a poetic paragraph
    print(f"\n\n{'#'*70}")
    print(f"POETIC PARAGRAPH TEST")
    print(f"{'#'*70}")
    
    poems = [
        "she walks in beauty like the night of cloudless climes and starry skies",
        "a thing of beauty is a joy forever its loveliness increases",
        "the sea is calm tonight the tide is full the moon lies fair upon the strait",
    ]
    for poem in poems:
        write(poem)
