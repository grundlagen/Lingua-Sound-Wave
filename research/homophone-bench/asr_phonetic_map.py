#!/usr/bin/env python3
"""
LARGE-SCALE WRONG-LANGUAGE ASR — French → English phonetic mapping.
No literal translations. Pure French text → EN voice TTS → English-sounding output.

Processes full French corpus (Candide) through English TTS voice,
building a fine-grained map of "what French sounds like to an English ear."

This IS the homophone generation engine, reversed:
  French word → EN voice mispronunciation → closest English word

Run: python asr_phonetic_map.py
"""

import subprocess, sys
from collections import defaultdict, Counter

# ── Load French words with IPA ──
def load_fr_vocab():
    words = []
    for i,line in enumerate(open("fr-word-ipa.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=2 and p[1] and "(en)" not in p[0]:
            words.append(p[0].lower())
    return words

# ── Load French sentences from corpus ──
def load_fr_sentences(n=500):
    sentences = []
    for fp in ["/tmp/fr-candide.txt","/tmp/fr-monte-cristo.txt","/tmp/fr-les-mis.txt","/tmp/fr-vingt-mille.txt"]:
        try:
            txt = open(fp,encoding="utf-8",errors="ignore").read()
            # Extract sentences (roughly)
            import re
            sents = re.split(r'[.!?]+', txt)
            for s in sents:
                s = s.strip()
                if 20 <= len(s) <= 120 and not s.startswith("***"):
                    sentences.append(s)
                    if len(sentences) >= n:
                        return sentences
        except FileNotFoundError:
            continue
    return sentences

# ── English TTS reading French (the "bad ASR") ──
_CACHE = {}
def en_voice_hears_fr(fr_text):
    key = fr_text[:80]
    if key in _CACHE: return _CACHE[key]
    r = subprocess.run(["espeak-ng","-q","--ipa","-v","en-us",fr_text],
                       capture_output=True, text=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    _CACHE[key] = ipa
    return ipa

# ── Extract "English-sounding words" from the IPA ──
def extract_phonetic_words(ipa_stream):
    """Split English-accented IPA into word-like chunks (by stress/spaces)."""
    # The EN voice inserts spaces between words — use them
    words = ipa_stream.split()
    cleaned = []
    for w in words:
        w = w.replace(" ","").replace(".","")
        if len(w) >= 2 and not w.startswith("("):
            cleaned.append(w)
    return cleaned

# ── Load English word-IPA dictionary for reverse lookup ──
def load_en_vocab():
    en_ipa = {}
    try:
        for i,line in enumerate(open("en-word-ipa.tsv",encoding="utf-8")):
            if i==0: continue
            p = line.rstrip("\n").split("\t")
            if len(p)>=2 and p[1] and "(fr)" not in p[0]:
                en_ipa[p[0].lower()] = p[1]
    except:
        # Build a tiny fallback
        en_ipa = {"the":"ðə","a":"ə","an":"æn","of":"ʌv","to":"tuː",
                  "in":"ɪn","on":"ɒn","at":"æt","for":"fɔːɹ","and":"ænd",
                  "or":"ɔːɹ","is":"ɪz","are":"ɑːɹ","was":"wʌz","be":"biː",
                  "it":"ɪt","he":"hiː","she":"ʃiː","we":"wiː","they":"ðeɪ",
                  "my":"maɪ","his":"hɪz","her":"hɜː","our":"aʊɚ",
                  "this":"ðɪs","that":"ðæt","not":"nɒt","but":"bʌt",
                  "so":"soʊ","as":"æz","by":"baɪ","with":"wɪð",
                  "all":"ɔːl","some":"sʌm","more":"mɔːɹ","very":"vɛɹi"}
    return en_ipa

# ── Build French → English phonetic mapping ──
def build_fr_en_map(fr_vocab, n=300):
    """For each French word, get its English-ear pronunciation."""
    mapping = {}
    for i, fr_word in enumerate(fr_vocab[:n]):
        if i % 50 == 0: print(f"  {i}/{n}...")
        en_pron = en_voice_hears_fr(fr_word).replace(" ","")
        fr_native = ""
        try:
            r = subprocess.run(["espeak-ng","-q","--ipa","-v","fr",fr_word],
                               capture_output=True, text=True)
            fr_native = r.stdout.strip()
            for c in "ˈˌ ": fr_native = fr_native.replace(c,"")
        except: pass
        mapping[fr_word] = (en_pron, fr_native)
    return mapping

# ── Find closest English words ──
def ndice(a,b,n=2):
    def ng(s): return {s[i:i+n] for i in range(len(s)-n+1)} if len(s)>=n else {s}
    A,B=ng(a),ng(b); return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

def find_english_matches(fr_word, en_pron, en_vocab, top=5):
    """Find English words whose IPA is closest to the EN-ear pronunciation of the French word."""
    matches = []
    for en_word, en_ipa in en_vocab.items():
        score = ndice(en_pron, en_ipa.replace(" ",""))
        if score >= 0.30:
            matches.append((score, en_word, en_ipa))
    matches.sort(reverse=True)
    return matches[:top]

# ═══════════════════════════════════════════════════════════════════
print("LARGE-SCALE WRONG-LANGUAGE ASR — French → English Phonetic Map")
print("="*65)

print("Loading French vocabulary...")
fr_vocab = load_fr_vocab()
print(f"  {len(fr_vocab)} French words")

print("Loading English vocabulary...")
en_vocab = load_en_vocab()
print(f"  {len(en_vocab)} English words")

print(f"\nProcessing {min(300, len(fr_vocab))} French words through English TTS...")
fr_en_map = build_fr_en_map(fr_vocab, n=200)
print(f"  {len(fr_en_map)} mapped")

# ── Show the mapping ──
print(f"\n{'='*65}")
print(f"FRENCH → ENGLISH PHONETIC DEFORMATION (what the EN ear hears)")
print(f"{'='*65}")
print(f"  {'FR word':15s} {'FR native':20s} {'EN ear':20s} {'closest EN words'}")
print(f"  {'─'*15} {'─'*20} {'─'*20} {'─'*30}")

interesting = []
for fr_word, (en_pron, fr_native) in list(fr_en_map.items())[:80]:
    matches = find_english_matches(fr_word, en_pron, en_vocab, top=3)
    if matches:
        match_str = ", ".join(f"{w}({s:.2f})" for s,w,_ in matches[:3])
        shift = ndice(fr_native, en_pron) if fr_native else 0
        print(f"  {fr_word:15s} [{fr_native:18s}] [{en_pron:18s}] {match_str}")
        interesting.append((shift, fr_word, fr_native, en_pron, matches))

# ── Most deformed (biggest gap between FR native and EN ear) ──
print(f"\n{'='*65}")
print(f"MOST DEFORMED: French words that sound MOST different to the English ear")
print(f"{'='*65}")
interesting.sort(key=lambda x: x[0])
for shift, fr_word, fr_native, en_pron, matches in interesting[:20]:
    gap = 1.0 - shift
    match_str = ", ".join(f"{w}({s:.2f})" for s,w,_ in matches[:3])
    print(f"  gap={gap:.3f}  {fr_word:15s} [{fr_native:18s}] → [{en_pron:18s}] → {match_str}")

# ── Process sentences at scale ──
print(f"\n{'='*65}")
print(f"SENTENCE-LEVEL ASR: French sentences through English ear")
print(f"{'='*65}")

sentences = load_fr_sentences(n=30)
print(f"  {len(sentences)} French sentences\n")

for s in sentences[:15]:
    en_heard = en_voice_hears_fr(s)
    phonetic_words = extract_phonetic_words(en_heard)
    # Clean up
    clean_words = [w for w in phonetic_words if len(w)>=2 and not any(c in "()[]" for c in w)]
    fr_native = ""
    try:
        r = subprocess.run(["espeak-ng","-q","--ipa","-v","fr",s],
                           capture_output=True, text=True)
        fr_native = r.stdout.strip()[:60]
        for c in "ˈˌ": fr_native = fr_native.replace(c,"")
    except: pass
    print(f"  FR: {s[:80]}...")
    print(f"  FR: [{fr_native}]")
    print(f"  EN: [{en_heard[:100]}]")
    print(f"  EN words: {' '.join(clean_words[:12])}")
    print()

print(f"\n{'='*65}")
print(f"INSIGHT: French → EN TTS → English-sounding word sequences")
print(f"These are the raw material for homophonic translation generation.")
print(f"At scale (full French corpus × EN TTS), this builds a complete")
print(f"FR→EN homophone dictionary with phonetic deformation rules.")
