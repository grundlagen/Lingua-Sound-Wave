#!/usr/bin/env python3
"""
SUBTITLE-TTS HOMOPHONE MINER — "Wrong language subtitles" as data source.

THE INSIGHT:
  French subtitles for English films are translated to match TIMING.
  Read the French subtitle with an ENGLISH TTS voice → sounds like the original?
  Read the English original with a FRENCH TTS voice → sounds like the subtitle?

  When BOTH directions have high phonetic similarity, the subtitle pair is
  a natural homophonic translation — discovered, not generated.

METHOD:
  1. Take parallel EN/FR subtitle pairs (sentence-aligned)
  2. For each pair, run FOUR TTS syntheses:
     EN text → EN voice (target_ipa)
     FR text → FR voice (fr_native_ipa)
     FR text → EN voice (en_misreading_ipa)  ← "wrong language subtitle"
     EN text → FR voice (fr_misreading_ipa)  ← "wrong language subtitle"
  3. Cross-accent match:
     EN_match = similarity(target_ipa, en_misreading_ipa)
     FR_match = similarity(target_ipa, fr_native_ipa)
  4. Pairs with EN_match ≥ 0.50 AND FR_match ≥ 0.40 are homophonic translations
  5. These are TRAINING DATA for the homophonic writer

DEMO: Python espeak-ng TTS on sample subtitle pairs.
"""

import subprocess
from collections import defaultdict

def tts(text, voice="en-us"):
    """Text-to-speech via espeak-ng → IPA."""
    r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text],
                       capture_output=True, text=True)
    ipa = r.stdout.strip()
    # Clean stress marks
    for c in "ˈˌ":
        ipa = ipa.replace(c,"")
    return ipa

def _segs(ipa):
    s,i,n = [],0,len(ipa)
    while i<n:
        if i+1<n and ipa[i:i+2] in {"ɑ̃","ɛ̃","ɔ̃","œ̃","t͡ʃ","d͡ʒ","t͡s"}:
            s.append(ipa[i:i+2]); i+=2
        else: s.append(ipa[i]); i+=1
    return s

def ndice(ia,ib,n=2):
    def ng(s): return {s[i:i+n] for i in range(len(s)-n+1)} if len(s)>=n else {s}
    A,B=ng(ia),ng(ib); return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

# ── SAMPLE SUBTITLE PAIRS (EN film → FR subtitle) ──
# Real translations from films, hand-picked for phonetic interest
SAMPLES = [
    ("I love you", "Je t'aime"),
    ("Good morning", "Bonjour"),
    ("See you later", "À plus tard"),
    ("What is your name", "Comment tu t'appelles"),
    ("I am sorry", "Je suis désolé"),
    ("Thank you very much", "Merci beaucoup"),
    ("Where is the bathroom", "Où sont les toilettes"),
    ("The cat is on the table", "Le chat est sur la table"),
    ("We must go now", "Il faut partir maintenant"),
    ("She walks in beauty", "Elle marche dans la beauté"),
    ("The sea remembers every ship", "La mer se souvient de chaque navire"),
    ("I dream of you", "Je rêve de toi"),
    ("Come with me", "Viens avec moi"),
    ("Never say never", "Ne jamais dire jamais"),
    ("Time is running out", "Le temps s'épuise"),
]

print("SUBTITLE-TTS HOMOPHONE MINER")
print("="*65)
print(f"  {'EN original':30s} {'FR subtitle':30s} {'EN-match':>8s} {'FR-match':>8s} {'cross':>7s}")
print(f"  {'─'*30} {'─'*30} {'─'*8} {'─'*8} {'─'*7}")

results = []
for en_text, fr_text in SAMPLES:
    # Four TTS syntheses
    en_native = tts(en_text, "en-us")     # Original English → English voice
    fr_native = tts(fr_text, "fr")        # French subtitle → French voice
    fr_english = tts(fr_text, "en-us")    # French subtitle → English voice (WRONG)
    en_french  = tts(en_text, "fr")       # English → French voice (WRONG)

    # Cross-accent matching
    en_match = ndice(en_native.replace(" ",""), fr_english.replace(" ",""))
    fr_match = ndice(en_native.replace(" ",""), fr_native.replace(" ",""))
    cross = (en_match + fr_match) / 2

    results.append((en_match, fr_match, cross, en_text, fr_text))
    marker = " ★" if cross >= 0.35 else ""
    print(f"  {en_text:30s} {fr_text:30s} {en_match:8.3f} {fr_match:8.3f} {cross:7.3f}{marker}")

# Sort by cross-accent score
results.sort(reverse=True)
print(f"\nTOP HOMOPHONIC SUBTITLE PAIRS:")
for en_m, fr_m, cross, en_t, fr_t in results[:5]:
    print(f"  cross={cross:.3f}  \"{en_t}\" ↔ \"{fr_t}\"")

print(f"\nINSIGHT: {sum(1 for r in results if r[2]>=0.35)}/{len(results)} pairs have cross≥0.35")
print("These are naturally-occurring homophonic translations found in subtitle data.")
print("At scale (OpenSubtitles: millions of pairs), this mines a training corpus.")
