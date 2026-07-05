#!/usr/bin/env python3
"""
WRONG-LANGUAGE ASR HOMOPHONE GENERATOR

THE CONCEPT:
  A French film has French dialogue and English subtitles.
  Feed the French audio through an ENGLISH speech recognizer (badly configured).
  The ASR "hears" English words in the French sounds — its hallucinations
  ARE homophonic translations.

  We simulate this with espeak-ng:
    French text → EN voice TTS → English-accented IPA
    This IPA is what the "badly configured English ASR" would output.
    The English words it contains are the homophonic translation.

  Then compare to the actual English subtitle — how close is the ASR hallucination?

DEMO: French film dialogue → "bad EN ASR" → English homophone output

Run: python wrong_asr_homophone.py
"""

import subprocess
from collections import defaultdict

# ── Simulate "badly configured English ASR" on French ──
def en_asr_hears_french(fr_text):
    """
    What would an English ASR hear if fed French audio?
    We simulate: read French text with English TTS voice.
    The result is English-accented French IPA.
    Then we convert IPA back to English-sounding words.
    """
    # English voice reading French text (the "badly configured TTS")
    r = subprocess.run(["espeak-ng","-q","--ipa","-v","en-us",fr_text],
                       capture_output=True, text=True)
    en_ipa = r.stdout.strip()
    for c in "ˈˌ": en_ipa = en_ipa.replace(c,"")
    return en_ipa

def fr_native_tts(fr_text):
    """French voice reading French text (correct)."""
    r = subprocess.run(["espeak-ng","-q","--ipa","-v","fr",fr_text],
                       capture_output=True, text=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    return ipa

def en_tts(en_text):
    """English voice reading English text."""
    r = subprocess.run(["espeak-ng","-q","--ipa","-v","en-us",en_text],
                       capture_output=True, text=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    return ipa

def ndice(a,b):
    def ng(s,n=2): return {s[i:i+n] for i in range(len(s)-n+1)} if len(s)>=n else {s}
    A,B=ng(a),ng(b); return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

# ── FRENCH FILM SCENES WITH ENGLISH SUBTITLES ──
SCENES = [
    # French film dialogue              English subtitle
    ("Je ne sais pas quoi faire",       "I don't know what to do"),
    ("Où est la gare",                  "Where is the station"),
    ("Je t'aime de tout mon coeur",     "I love you with all my heart"),
    ("Il fait beau aujourd'hui",        "It's beautiful today"),
    ("Qu'est-ce que tu veux",           "What do you want"),
    ("Je suis fatigué",                 "I am tired"),
    ("Regarde le ciel",                 "Look at the sky"),
    ("Viens avec moi ce soir",          "Come with me tonight"),
    ("C'est la vie",                    "That's life"),
    ("Je me souviens de toi",           "I remember you"),
    ("Le chat dort sur le lit",         "The cat sleeps on the bed"),
    ("J'ai faim",                       "I am hungry"),
    ("Il pleut beaucoup",               "It's raining a lot"),
    ("Tu me manques",                   "I miss you"),
    ("Fermez la porte",                 "Close the door"),
    ("Un petit café s'il vous plaît",   "A small coffee please"),
    ("Où sont les toilettes",           "Where is the bathroom"),
    ("Je ne comprends pas",             "I don't understand"),
    ("Bonne nuit mon amour",            "Good night my love"),
    ("À demain matin",                  "See you tomorrow morning"),
]

print("WRONG-LANGUAGE ASR HOMOPHONE GENERATOR")
print("="*72)
print("French film dialogue → badly-configured English ASR → English homophones\n")

print(f"  {'FRENCH dialogue':30s} {'EN ASR hears':35s} {'EN subtitle':30s} {'ASR↔sub':>7s}")
print(f"  {'─'*30} {'─'*35} {'─'*30} {'─'*7}")

best_pairs = []

for fr_line, en_sub in SCENES:
    # The "badly configured ASR": EN voice reading FR dialogue
    asr_hears = en_asr_hears_french(fr_line)
    # The actual English subtitle
    sub_ipa = en_tts(en_sub)
    # Native French for reference
    fr_ipa = fr_native_tts(fr_line)
    
    # How close is the ASR hallucination to the real subtitle?
    asr_sub_match = ndice(asr_hears.replace(" ",""), sub_ipa.replace(" ",""))
    # How close is the ASR hallucination to the native French?
    asr_fr_match = ndice(asr_hears.replace(" ",""), fr_ipa.replace(" ",""))
    
    best_pairs.append((asr_sub_match, fr_line, en_sub, asr_hears, sub_ipa))
    
    marker = " ★" if asr_sub_match >= 0.25 else ""
    print(f"  {fr_line:30s} [{asr_hears:33s}] {en_sub:30s} {asr_sub_match:7.3f}{marker}")

print(f"\n{'='*72}")
print("BEST ASR→SUBTITLE MATCHES (the ASR 'hallucinated' close to real subtitle):")
best_pairs.sort(reverse=True)
for match, fr, en, asr_ipa, sub_ipa in best_pairs[:8]:
    print(f"  ASR↔sub={match:.3f}")
    print(f"    FR:   {fr}")
    print(f"    ASR:  [{asr_ipa}]")
    print(f"    SUB:  {en}  [{sub_ipa}]")
    print()

# ── THE KEY INSIGHT: Generate English from French via wrong ASR ──
print(f"{'='*72}")
print("GENERATION DEMO: French → wrong EN ASR → homophonic English")
print(f"{'='*72}")

# Take a French phrase, run through EN TTS, show what English it sounds like
demo = [
    "Un petit d'un petit s'étonne aux Halles",
    "Et qui rit des curés d'Oc",
    "Reine reine gueux éveille",
    "Jacques s'apprête coulis de nos fête",
]

for fr_text in demo:
    en_hears = en_asr_hears_french(fr_text)
    fr_native = fr_native_tts(fr_text)
    # The phonetic gap between "what French sounds like" and "what EN ASR hears"
    gap = ndice(en_hears.replace(" ",""), fr_native.replace(" ",""))
    print(f"\n  FRENCH:   {fr_text}")
    print(f"  FR IPA:   [{fr_native}]")
    print(f"  EN ASR:   [{en_hears}]")
    print(f"  FR↔ASR:   {gap:.3f} (how much French sound survives the English ASR)")
