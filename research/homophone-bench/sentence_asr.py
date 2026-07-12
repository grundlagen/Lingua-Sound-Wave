#!/usr/bin/env python3
"""
FULL-SENTENCE WRONG-LANGUAGE ASR — FR sentences → EN ear → English words.

THE INSIGHT:
  "dans un" → EN ear [dænzʌn] → sounds like "dancing"
  At sentence scale, French produces English-sounding sequences naturally.
  This IS the discovery mechanism for homophonic translations.

  Run full French sentences through EN TTS, split into English-sounding
  phonetic words, and back-map each to the closest English word.
  The result is an English sentence that "sounds like" the French original.

  This is what a badly-configured Whisper would output if fed French audio.

Run: python sentence_asr.py
"""

import subprocess, os, re, json
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

# ── Load English dictionary (word → IPA) for reverse lookup ──
en_dict = {}
for i,line in enumerate(open("en-word-ipa.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[1] and "(fr)" not in p[0]:
        en_dict[p[0].lower()] = p[1].replace(" ","")

# ── Load French sentences from corpus ──
sentences = []
for fp in ["/tmp/fr-candide.txt","/tmp/fr-monte-cristo.txt",
           "/tmp/fr-les-mis.txt","/tmp/fr-vingt-mille.txt"]:
    try:
        txt = open(fp,encoding="utf-8",errors="ignore").read()
        txt = re.sub(r'\*\*\*.*?\*\*\*', '', txt, flags=re.S)
        for s in re.split(r'[.!?]+', txt):
            s = s.strip()
            if 40 <= len(s) <= 150:
                sentences.append(s)
                if len(sentences) >= 30: break
        if len(sentences) >= 30: break
    except: continue

# ── Build English IPA lookup structures ──
en_ipa_list = [(w, ipa) for w, ipa in en_dict.items() if len(ipa) >= 3]
# Pre-compute word lengths for faster lookup
en_by_len = {}
for w, ipa in en_ipa_list:
    length = len(ipa)
    if length not in en_by_len: en_by_len[length] = []
    en_by_len[length].append((w, ipa))

# ── Process sentences ──
print("FULL-SENTENCE WRONG-LANGUAGE ASR")
print("="*65)

for si, sent in enumerate(sentences[:15]):
    # French native
    fr_ipa = tts(sent, "fr").replace(" ","")
    # English ear (simulated Whisper output)
    en_ipa = tts(sent, "en-us")
    
    # Split EN-ear IPA into word-like chunks (by spaces that EN voice inserts)
    en_words_ipa = [w.replace(" ","") for w in en_ipa.split() if len(w.replace(" ","")) >= 2]
    
    # Back-map each chunk to closest English word
    back_mapped = []
    for chunk_ipa in en_words_ipa:
        best_word, best_score = "?", 0.0
        # Search within ±2 length range
        cl = len(chunk_ipa)
        for length in range(max(2, cl-2), min(20, cl+3)):
            for en_w, en_w_ipa in en_by_len.get(length, []):
                s = ndice(chunk_ipa, en_w_ipa)
                if s > best_score:
                    best_score = s
                    best_word = en_w
        if best_score >= 0.40:
            back_mapped.append((best_word, best_score, chunk_ipa))
        else:
            back_mapped.append((f"[{chunk_ipa}]", 0.0, chunk_ipa))
    
    en_sentence = " ".join(w for w,_,_ in back_mapped)
    
    print(f"\n{'─'*65}")
    print(f"FR: {sent[:100]}...")
    print(f"FR IPA: [{fr_ipa[:80]}]")
    print(f"EN ear: {en_ipa[:100]}")
    print(f"EN words: {' '.join(en_words_ipa[:12])}")
    print(f"→ English: {en_sentence}")
    print(f"  detail: {'  '.join(f'{w}({s:.2f})' for w,s,_ in back_mapped[:8])}")

# ── Specific demo: "dans un" → "dancing" ──
print(f"\n{'='*65}")
print(f"DISCOVERY: French bigrams → English words")
print(f"{'='*65}")

discoveries = [
    ("dans un", "dancing"),
    ("vous êtes", "voo zair"),
    ("tout le monde", "tool mond"),
    ("je ne sais pas", "juh nuh say pah"),
    ("c'est la vie", "say la vee"),
    ("petit déjeuner", "puh tee day zhuh nay"),
]

for fr_phrase, en_approx in discoveries:
    en_ipa = tts(fr_phrase, "en-us").replace(" ","")
    fr_ipa = tts(fr_phrase, "fr").replace(" ","")
    
    # Find closest English word to the EN-ear IPA
    best_word, best_score = "", 0
    for en_w, en_w_ipa in en_ipa_list:
        s = ndice(en_ipa, en_w_ipa)
        if s > best_score:
            best_score = s; best_word = en_w
    
    print(f"  \"{fr_phrase:25s}\" → EN ear [{en_ipa:18s}] → EN word \"{best_word}\" ({best_score:.3f})")
    print(f"    FR native: [{fr_ipa}]")

# ── Full sentence that sounds like English ──
print(f"\n{'='*65}")
print(f"BEST MATCH: French sentences that sound MOST like English")
print(f"{'='*65}")

# Score each sentence by how much of it maps to real English words
scored = []
for sent in sentences[:15]:
    en_ipa = tts(sent, "en-us")
    en_words_ipa = [w.replace(" ","") for w in en_ipa.split() if len(w.replace(" ","")) >= 2]
    total_score = 0
    mapped_count = 0
    for chunk in en_words_ipa:
        cl = len(chunk)
        best = 0
        for length in range(max(2, cl-2), min(20, cl+3)):
            for _, en_w_ipa in en_by_len.get(length, []):
                s = ndice(chunk, en_w_ipa)
                if s > best: best = s
        total_score += best
        if best >= 0.35: mapped_count += 1
    scored.append((total_score/max(1, len(en_words_ipa)), mapped_count, sent, en_ipa))

scored.sort(reverse=True)
for avg_score, mapped, sent, en_ipa in scored[:5]:
    print(f"  score={avg_score:.3f}  mapped={mapped} words  \"{sent[:80]}...\"")
    print(f"    EN ear: [{en_ipa[:100]}]")
    print()

print(f"\n{'='*65}")
print(f"This is the data that a real Whisper ASR would produce at scale.")
print(f"espeak-ng simulates it; Whisper on actual French audio would")
print(f"discover homophonic translations directly from speech.")
