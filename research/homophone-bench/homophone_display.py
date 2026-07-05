#!/usr/bin/env python3
"""
HOMOPHONE DISPLAY — Show what French words sound like to an English ear.
Also builds the correct-direction training corpus for the LLM.

The LLM must learn: "English text X" → "French spelling Y where Y sounds like X"
NOT French synonyms. The training data direction matters.

Run: python homophone_display.py
"""
import subprocess, os, json
import numpy as np

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

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

# Load the paragraph
d = json.load(open("full_dict_paragraph.json"))
pairs = d["pairs"]

# Load English vocab for reverse lookup
en_vocab = {}
for i,line in enumerate(open("en-word-ipa.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[1] and "(fr)" not in p[0]:
        en_vocab[p[0].lower()] = p[1].replace(" ","")

print("HOMOPHONE DISPLAY — What French sounds like to an English ear")
print("=" * 65)

# ── Show top 20: French word → what English ear hears → closest English word ──
results = []
for p in pairs[:80]:
    fr_w = p["fr"]
    en_source = p["en"]
    
    # What does the French word sound like to an English ear?
    en_ear_ipa = tts(fr_w, "en-us").replace(" ","")
    fr_native = tts(fr_w, "fr").replace(" ","")
    target_ipa = tts(en_source, "en-us").replace(" ","")
    
    # Find closest English word to the EN-ear IPA
    best_en, best_score = "", 0
    for en_w, en_ipa in en_vocab.items():
        s = ndice(en_ear_ipa, en_ipa)
        if s > best_score:
            best_score, best_en = s, en_w
    
    cross_score = ndice(en_ear_ipa, target_ipa)
    
    results.append({
        "fr": fr_w, "en_source": en_source,
        "sounds_like": best_en, "en_ear_ipa": en_ear_ipa,
        "cross": cross_score, "vocab_match": best_score,
    })

results.sort(key=lambda x: -(x["cross"] + x["vocab_match"]))
print(f"\n  {'FR word':15s} {'sounds like':15s} {'EN source':15s} {'cross':>6s} {'vocab':>6s}")
print(f"  {'─'*15} {'─'*15} {'─'*15} {'─'*6} {'─'*6}")
for r in results[:25]:
    print(f"  {r['fr']:15s} {r['sounds_like']:15s} {r['en_source']:15s} {r['cross']:6.3f} {r['vocab_match']:6.3f}")

# ── Show the stanzas with English-homophone annotations ──
print(f"\n{'='*65}")
print(f"STANZAS WITH ENGLISH HOMOPHONE ANNOTATIONS")
print(f"{'='*65}")

# Load the LLM-organized stanzas
STANZAS = [
    ["ouest","est","sève","terre","terres","terreau","cerises","chaudes","chaude","sol",
     "sols","soupe","soupes","miettes","choux","zeste","roue","rhône","houle","ouate"],
    ["sauter","sautent","sortent","sortes","tournent","traîne","trappe","tirs","tir","tendes",
     "tenir","transforment","transforme","transmettent","transport","transfert","skient","ski","slip","slip"],
    ["speech","dis","verrez","savez","savent","valent","verbes","vexes","toquent","toque",
     "toc","stoppe","stoppes","cède","cessent","déceler","laisses","opte","ouais","ouais"],
    ["technologie","système","script","sketch","spots","sport","stade","stores","square","strip",
     "strip","télésope","textes","textiles","pelle","pelle","tamis","tests","tester","test"],
    ["tristesse","victorieux","joli","folie","souhaite","stresse","stressent","déchaîne","rixe","torts",
     "toutefois","inacceptables","spécifique","strict","stricte","strict","stricte","série","trahi","soupir"],
    ["celtes","slaves","troupes","troupes","troupes","aides","aides","aides","volante","voler",
     "vider","vacciner","saine","diocèses","tipi","tels","ceci","cette","cette","ouah"],
    ["tête","os","tonte","veines","corps","live","sieste","semelle","nique","sic",
     "soc","tontes","simili","symphonie","tim","titi","tamis","tamis","sushis","sushi"],
    ["chaise","sacs","solde","semestre","tome","script","soupir","soupir","scie","sciure",
     "choses","chiffe","hard","chers","schiste","séquelles","chili","chics","chauve","chatte"],
    ["horde","ouate","n","oui","oies","citerne","round","acquis","reprit","daller",
     "step","stade","té hic","treize","très","tri","trique","trique","trous","tout"],
    ["vielle","val","vol","veines","verrez","valent","vestes","zooms","zeste","yang",
     "vais tri","vannes","valent","est","opte","saine","savent","savez","transforment","stresse"],
]

# Build reverse lookup: fr_word → (en_source, what_it_sounds_like)
sound_like = {}
for p in pairs:
    fr_w = p["fr"]
    en_ear = tts(fr_w, "en-us").replace(" ","")
    best_en, best_s = "", 0
    for en_w, en_ipa in list(en_vocab.items())[:3000]:
        s = ndice(en_ear, en_ipa)
        if s > best_s:
            best_s, best_en = s, en_w
    sound_like[fr_w] = (p["en"], best_en, best_s)

for i, stanza in enumerate(STANZAS):
    print(f"\n  [{i:2d}] FR: {'  '.join(stanza[:8])}")
    # Show what each word SOUNDS LIKE in English
    en_sounds = []
    for w in stanza[:8]:
        if w in sound_like:
            en_src, en_sound, score = sound_like[w]
            en_sounds.append(f"{en_sound}({score:.2f})" if score > 0.3 else f"[{w}]")
        else:
            en_sounds.append(f"[{w}]")
    print(f"       EN: {'  '.join(en_sounds)}")

# ── Build correct-direction training corpus ──
print(f"\n{'='*65}")
print(f"TRAINING CORPUS — Correct direction: EN input → FR homophone output")
print(f"{'='*65}")

training_rows = []
for p in pairs:
    fr_w = p["fr"]
    en_source = p["en"]
    # This is what the LLM needs to learn:
    # Input: English word
    # Output: French spelling that sounds like it
    training_rows.append({
        "input": f"English word: {en_source}",
        "output": fr_w,
        "sound_score": p["sound"],
    })

# Filter: only pairs where sound ≥ 0.70 and EN ≠ FR
filtered = [r for r in training_rows if r["sound_score"] >= 0.70]
print(f"  {len(filtered)} high-quality training pairs (sound ≥ 0.70, EN ≠ FR)")
print(f"  Sample:")
for r in filtered[:10]:
    print(f"    IN:  {r['input']}")
    print(f"    OUT: {r['output']}")
    print()

with open("train-homophonic-corrected.jsonl","w") as f:
    for r in filtered:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  Saved train-homophonic-corrected.jsonl ({len(filtered)} rows)")
print(f"  → Feed this to train_selflearn.py for LLM fine-tuning")
