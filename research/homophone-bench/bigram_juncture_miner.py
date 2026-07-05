#!/usr/bin/env python3
"""
BIGRAM JUNCTURE MINER — Wrong-language TTS on word transitions.

THE INSIGHT:
  Word boundaries are where homophonic translation lives or dies.
  "petit ami" run through EN voice ≠ "petit" + "ami" run separately.
  The difference IS the juncture effect: elision, liaison, connected speech.

  By running French bigrams through English TTS and comparing with
  individual words, we extract the JUNCTURE DEFORMATION RULES that
  govern how French sound flows across word boundaries to an English ear.

  This data directly improves the whole-line carve decoder's LIAISON feature.

ALSO: English bigrams → French TTS for the reverse direction.

Run: python bigram_juncture_miner.py
"""

import subprocess, os, re, json
from collections import defaultdict, Counter
import numpy as np

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ── TTS ──
def tts(text, voice):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text],
                       capture_output=True, text=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    return ipa

def ndice(a,b,n=2):
    def ng(s): return {s[i:i+n] for i in range(len(s)-n+1)} if len(s)>=n else {s}
    A,B=ng(a),ng(b); return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

# ── Extract common bigrams from French corpus ──
def extract_bigrams(n=500):
    """Extract common word bigrams from the French corpus."""
    bigrams = Counter()
    
    for fp in ["/tmp/fr-candide.txt","/tmp/fr-monte-cristo.txt",
               "/tmp/fr-les-mis.txt","/tmp/fr-vingt-mille.txt"]:
        try:
            txt = open(fp,encoding="utf-8",errors="ignore").read()
            txt = re.sub(r'\*\*\*.*?\*\*\*', '', txt, flags=re.S)
            words = [w.lower().strip(".,;:!?'\"()[]") for w in txt.split() 
                    if w.strip(".,;:!?'\"()[]") and len(w) >= 2]
            for a,b in zip(words, words[1:]):
                bigrams[(a,b)] += 1
        except FileNotFoundError: continue
    
    # Get the most frequent
    return bigrams.most_common(n)

# ── Juncture analysis ──
def analyze_juncture(fr_bigrams, en_bigrams, n=200):
    """
    For each bigram, run:
      1. Full bigram through EN voice (connected speech)
      2. Word A alone through EN voice
      3. Word B alone through EN voice
      4. Concatenate A+B IPAs (no juncture)
      5. Compare: full vs concatenated = juncture effect
      6. Also compare to FR native for reference
    """
    results = []
    
    print(f"  Analyzing {n} FR bigrams + {n} EN bigrams...")
    
    # ── FR→EN direction ──
    for i, ((a,b), count) in enumerate(fr_bigrams[:n]):
        if i % 30 == 0 and i > 0: print(f"    FR→EN: {i}/{n}...")
        
        # Full bigram through EN voice
        bigram_text = f"{a} {b}"
        bigram_en = tts(bigram_text, "en-us").replace(" ","")
        
        # Individual words through EN voice
        a_en = tts(a, "en-us").replace(" ","")
        b_en = tts(b, "en-us").replace(" ","")
        concatenated = a_en + b_en
        
        # Full bigram through FR voice (native reference)
        bigram_fr = tts(bigram_text, "fr").replace(" ","")
        
        # Juncture effect: difference between connected and concatenated
        juncture_gap = ndice(bigram_en, concatenated) if concatenated else 0
        en_match = ndice(bigram_fr, bigram_en) if bigram_fr else 0
        
        if len(bigram_en) >= 4 and len(concatenated) >= 4:
            results.append({
                "fr_bigram": bigram_text,
                "count": count,
                "bigram_en": bigram_en,
                "concatenated": concatenated,
                "bigram_fr": bigram_fr,
                "juncture_gap": round(juncture_gap, 3),
                "en_match": round(en_match, 3),
                "a_en": a_en, "b_en": b_en,
            })
    
    # ── EN→FR direction ──
    en_results = []
    for i, ((a,b), count) in enumerate(en_bigrams[:n]):
        if i % 30 == 0 and i > 0: print(f"    EN→FR: {i}/{n}...")
        
        bigram_text = f"{a} {b}"
        bigram_fr = tts(bigram_text, "fr").replace(" ","")
        a_fr = tts(a, "fr").replace(" ","")
        b_fr = tts(b, "fr").replace(" ","")
        concatenated = a_fr + b_fr
        bigram_en = tts(bigram_text, "en-us").replace(" ","")
        
        juncture_gap = ndice(bigram_fr, concatenated) if concatenated else 0
        
        if len(bigram_fr) >= 4 and len(concatenated) >= 4:
            en_results.append({
                "en_bigram": bigram_text,
                "count": count,
                "bigram_fr": bigram_fr,
                "concatenated": concatenated,
                "bigram_en": bigram_en,
                "juncture_gap": round(juncture_gap, 3),
                "a_fr": a_fr, "b_fr": b_fr,
            })
    
    return results, en_results

# ── Find juncture-heavy bigrams ──
def find_juncture_rules(results):
    """Extract patterns: bigrams with large juncture gaps reveal rules."""
    # Sort by juncture gap (largest = most juncture effect)
    by_gap = sorted(results, key=lambda x: -x["juncture_gap"])
    
    print(f"\n  TOP JUNCTURE BIGRAMS (most boundary effect):")
    for r in by_gap[:15]:
        gap = r["juncture_gap"]
        print(f"    gap={gap:.3f}  \"{r['fr_bigram']:30s}\"")
        print(f"      connected:   [{r['bigram_en']}]")
        print(f"      concatenated:[{r['concatenated']}]")
    
    # Group by phonological pattern
    print(f"\n  JUNCTURE PATTERNS (grouped by word ending):")
    
    # Pattern: word ending in vowel → next starting with vowel (elision possible)
    elision_like = [r for r in results if r["juncture_gap"] < 0.8
                    and r["a_en"] and r["b_en"]
                    and r["a_en"][-1] in "aeiouyəɚɜʌɒɔɛɪʊ"
                    and r["b_en"][0] in "aeiouyəɚɜʌɒɔɛɪʊ"]
    if elision_like:
        avg_gap = np.mean([r["juncture_gap"] for r in elision_like[:20]])
        print(f"    Vowel→Vowel ({len(elision_like)}): avg gap={avg_gap:.3f}")
        for r in elision_like[:5]:
            print(f"      \"{r['fr_bigram']:25s}\" gap={r['juncture_gap']:.3f} "
                  f"[{r['a_en']}]·[{r['b_en']}] → [{r['bigram_en']}]")
    
    # Pattern: word ending in consonant → next starting vowel (liaison possible)
    liaison_like = [r for r in results if r["juncture_gap"] < 0.8
                    and r["a_en"] and r["b_en"]
                    and r["a_en"][-1] not in "aeiouyəɚɜʌɒɔɛɪʊ"
                    and r["b_en"][0] in "aeiouyəɚɜʌɒɔɛɪʊ"]
    if liaison_like:
        avg_gap = np.mean([r["juncture_gap"] for r in liaison_like[:20]])
        print(f"    Consonant→Vowel ({len(liaison_like)}): avg gap={avg_gap:.3f}")
        for r in liaison_like[:5]:
            print(f"      \"{r['fr_bigram']:25s}\" gap={r['juncture_gap']:.3f} "
                  f"[{r['a_en']}]·[{r['b_en']}] → [{r['bigram_en']}]")
    
    return by_gap

# ── The EN ear's juncture model ──
def build_juncture_model(results, en_results):
    """Build a simple model: for each French word ending/start pattern,
       what does the EN ear do at the boundary?"""
    
    model = {"fr_to_en": defaultdict(list), "en_to_fr": defaultdict(list)}
    
    for r in results:
        # Get the ending of word A and start of word B in EN-ear IPA
        a_end = r["a_en"][-2:] if len(r["a_en"]) >= 2 else ""
        b_start = r["b_en"][:2] if len(r["b_en"]) >= 2 else ""
        # What the EN ear hears at the boundary (the difference)
        boundary_en = r["bigram_en"]
        boundary_cat = r["concatenated"]
        if boundary_en != boundary_cat:
            key = f"{a_end}·{b_start}"
            model["fr_to_en"][key].append({
                "bigram": r["fr_bigram"],
                "connected": boundary_en,
                "separate": boundary_cat,
                "gap": r["juncture_gap"],
            })
    
    # Summarize
    summary = []
    for key, examples in model["fr_to_en"].items():
        if len(examples) >= 2:
            avg_gap = np.mean([e["gap"] for e in examples])
            summary.append((avg_gap, key, examples))
    summary.sort()
    
    return summary, model

# ═══════════════════════════════════════════════════════════════════
print("BIGRAM JUNCTURE MINER — Wrong-language TTS on word transitions")
print("="*65)

print("\nExtracting common bigrams from French corpus...")
fr_bigrams = extract_bigrams(600)
print(f"  FR: {len(fr_bigrams)} bigrams (top: {fr_bigrams[0][0]} ×{fr_bigrams[0][1]})")

# Also get English bigrams from fr corpus (it contains some EN fragments)
# For now, use the French corpus as a proxy — the EN→FR direction
# will use the same bigrams reversed through FR voice
en_bigrams = [(("the","same"), 1), (("I","am"), 1), (("you","are"), 1),
              (("it","is"), 1), (("he","said"), 1), (("she","was"), 1),
              (("we","have"), 1), (("they","were"), 1), (("do","not"), 1),
              (("can","be"), 1), (("will","have"), 1), (("has","been"), 1)]
# Also extract some from the English corpus if available
en_words_list = []
try:
    txt = open("/tmp/fr-candide.txt",encoding="utf-8").read()[:50000]
    # The Candide text has some English in headers
except: pass

print(f"\nAnalyzing juncture effects...")
fr_results, en_results = analyze_juncture(fr_bigrams, en_bigrams, n=200)

# ── FR→EN Juncture ──
print(f"\n{'─'*65}")
print("FR→EN JUNCTURE (French bigrams → English ear)")
print(f"{'─'*65}")

# Connected speech score distribution
gaps = [r["juncture_gap"] for r in fr_results]
print(f"  Juncture gap: μ={np.mean(gaps):.3f} σ={np.std(gaps):.3f}")
print(f"  Close to 1.0 = no juncture effect (words act independently)")
print(f"  Below 0.8 = strong juncture effect (words blend)")
print(f"  N={len(fr_results)}")

fr_rules = find_juncture_rules(fr_results)

# ── EN→FR Juncture ──  
print(f"\n{'─'*65}")
print("EN→FR JUNCTURE (English bigrams → French ear)")
print(f"{'─'*65}")

if en_results:
    en_gaps = [r["juncture_gap"] for r in en_results]
    print(f"  Juncture gap: μ={np.mean(en_gaps):.3f} σ={np.std(en_gaps):.3f}")
    for r in sorted(en_results, key=lambda x: -x["juncture_gap"])[:5]:
        print(f"    gap={r['juncture_gap']:.3f}  \"{r['en_bigram']:25s}\"")
        print(f"      connected:   [{r['bigram_fr']}]")
        print(f"      concatenated:[{r['concatenated']}]")

# ── Build juncture model ──
print(f"\n{'─'*65}")
print("JUNCTURE MODEL (boundary rules for the decoder)")
print(f"{'─'*65}")

summary, model = build_juncture_model(fr_results, en_results)
print(f"  Boundary patterns with strong juncture effects:")
for avg_gap, key, examples in summary[:10]:
    samples = ", ".join(e["bigram"] for e in examples[:3])
    print(f"    gap={avg_gap:.3f}  [{key}]  e.g. {samples}")

# ── Show what this means for the whole-line carve ──
print(f"\n{'─'*65}")
print("APPLICATION: How this improves the whole-line carve")
print(f"{'─'*65}")

# Pick the top 3 juncture-heavy bigrams and show the carve improvement
for r in sorted(fr_results, key=lambda x: -x["juncture_gap"])[:3]:
    gap = 1.0 - r["juncture_gap"]
    print(f"\n  Bigram: \"{r['fr_bigram']}\" (juncture gap={r['juncture_gap']:.3f})")
    print(f"    Separate words:  {r['a_en']} + {r['b_en']} = [{r['concatenated']}]")
    print(f"    Connected speech: [{r['bigram_en']}]")
    print(f"    → The decoder should use [{r['bigram_en']}] as the target IPA,")
    print(f"      not [{r['concatenated']}]. Difference is {gap*100:.0f}% of the signal.")

# ── Save ──
output = {
    "fr_juncture": [{"bigram": r["fr_bigram"], "juncture_gap": r["juncture_gap"],
                     "en_match": r["en_match"], "bigram_en": r["bigram_en"],
                     "concatenated": r["concatenated"]}
                    for r in sorted(fr_results, key=lambda x: -x["juncture_gap"])[:300]],
    "en_juncture": [{"bigram": r["en_bigram"], "juncture_gap": r["juncture_gap"],
                     "bigram_fr": r["bigram_fr"], "concatenated": r["concatenated"]}
                    for r in en_results[:100]],
    "boundary_rules": [{"gap": float(avg_gap), "pattern": key, "examples": [e["bigram"] for e in ex[:5]]}
                       for avg_gap, key, ex in summary[:50]],
    "stats": {"fr_juncture_mean": float(np.mean(gaps)),
              "fr_juncture_std": float(np.std(gaps)),
              "n_fr_bigrams": len(fr_results)},
}

with open("bigram_juncture_map.json","w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n  Saved bigram_juncture_map.json ({len(fr_results)} FR + {len(en_results)} EN bigrams)")
print(f"  → Feed these boundary rules into phonetic_decoder LIAISON")
print(f"  → Use as bigram transitions weights in whole_line_carve beam search")
