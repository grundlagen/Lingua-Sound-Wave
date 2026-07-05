#!/usr/bin/env python3
"""
DIRECT-LOOKUP VAN ROOTEN — Word-level homophone from gold DB + LM fluency.
No carving. No beam search. Just: each EN word → best FR homophone from DB.
Then bigram-LM score the result. Strictly judge.

This answers: "Can we make Van Rooten paragraphs from our database alone?"
"""

import subprocess, os, sys, json
from collections import defaultdict

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ── Load gold database (word-level) ──
gold = defaultdict(list)  # en_word → [(fr_word, sound_score), ...]

# Load strict-gold
for i,line in enumerate(open("strict-gold.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2:
        gold[p[0]].append((p[1], 1.0))

# Load tier-ladder
for i,line in enumerate(open("tier-ladder.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=12 and p[10]:
        try:
            gold[p[1]].append((p[2], float(p[10])))
        except: continue

# Load v7 gold
for i,line in enumerate(open("dictionary-v7.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=9 and p[3]=="1":
        gold[p[7]].append((p[8], float(p[1])))

# Sort by sound score
for k in gold: gold[k].sort(key=lambda x: -x[1])

# ── French vocabulary (for fluency gate) ──
fr_vocab = set()
for w in gold: fr_vocab.add(w)
for v in gold.values():
    for fr_w,_ in v: fr_vocab.add(fr_w)

# ── Bigram LM ──
try:
    import bigram_lm as blm
    LM = blm.load("fr")
    print(f"LM: {LM.N:,} tokens")
except: LM = None

# ── Strict judge (same as strict_judge.py) ──
def tts(text, voice):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text],
                       capture_output=True, text=True, check=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    return ipa

def ndice(a,b,n=2):
    A = {a[i:i+n] for i in range(len(a)-n+1)} if len(a)>=n else {a}
    B = {b[i:i+n] for i in range(len(b)-n+1)} if len(b)>=n else {b}
    return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

def strict_judge(en, fr_text):
    fr_ipa = tts(en, "en-us").replace(" ","")  # EN voice reads FR source
    en_ipa = tts(fr_text, "en-us").replace(" ","")  # EN voice reads EN output
    ph = ndice(fr_ipa, en_ipa)
    return ph

# ═══════════════════════════════════════════════════════════
MOTHER_GOOSE = [
    "Humpty Dumpty sat on a wall",
    "Humpty Dumpty had a great fall", 
    "All the king's horses and all the king's men",
    "Couldn't put Humpty together again",
]

STOP = {"the","a","an","of","in","on","and","or","to","for","with","by","at","as","is","are","was","were","be","been","am","it","he","she","we","they","you","me","him","her","us","them","my","your","his","her","its","our","their","this","that","these","those","not","no","so","but","if","all","some","more","very","just","only","too"}

print("DIRECT-LOOKUP VAN ROOTEN — Word-level homophone from gold DB")
print("=" * 60)

for line in MOTHER_GOOSE:
    words = [w.lower().strip(".,;:!?'\"") for w in line.split() if w.strip(".,;:!?'\"")]
    
    # For each word, get best FR homophone
    fr_words = []
    for w in words:
        if w in STOP:
            # Stop words: use function-word glue (zipf-glue.tsv)
            fr_words.append(None)
            continue
        if w in gold and gold[w]:
            fr_words.append(gold[w][0][0])  # best FR match
        else:
            fr_words.append(f"«{w}»")
    
    # Filter out None (stop words)
    fr_filtered = [w for w in fr_words if w is not None]
    
    # Build French line
    fr_line = " ".join(fr_filtered)
    
    # Fluency
    flu = LM.fluency(fr_filtered) if LM and len(fr_filtered)>=2 else 0.5
    
    # Judge
    ph = strict_judge(line, fr_line)
    
    print(f"\nEN: {line}")
    print(f"FR: {fr_line}")
    print(f"   ph={ph:.3f}  flu={flu:.3f}  words={len(fr_filtered)}")
    
    # Show word-level mapping
    for en_w, fr_w in zip(words, fr_words):
        if fr_w and fr_w in gold:
            score = gold[en_w][0][1] if en_w in gold else 0
            print(f"   {en_w:15s} → {fr_w:15s} (db={score:.2f})")
