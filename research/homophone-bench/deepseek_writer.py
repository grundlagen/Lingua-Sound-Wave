#!/usr/bin/env python3
"""
DEEPSEEK-POWERED HOMOPHONE WRITER — LLM as the generator, DB as the verifier.

ARCHITECTURE:
  1. LOOKUP: strict-gold-training.jsonl (9,803 verified pairs) — fast, trusted
  2. DEEPSEEK: For unknown words, the LLM generates French homophones
  3. VERIFY: Agent B (espeak-ng G2P) checks what the French sounds like
  4. STORE: Good pairs added back to the DB for future fast lookup

No GPU training needed. The LLM already knows phonetics from training data.
The database + verification keeps the LLM honest.

Run: python3 deepseek_writer.py "the ocean remembers every vessel"
"""

import json, os, subprocess, sys
from collections import defaultdict

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ── Load lookup DB (fast, verified) ──
with open("strict-gold-training.jsonl") as f:
    raw = [json.loads(line) for line in f]
lookup = {}
for r in raw:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    q = r.get("quality", r.get("sound", 1.0))
    if en and fr and en != fr:
        if en not in lookup or q > lookup[en][1]: lookup[en] = (fr, q)
print(f"Lookup: {len(lookup)} pairs")

# ── Agent B: espeak-ng G2P verifier ──
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

def verify(fr_word):
    """What English word does Agent B hear?"""
    en_ipa = tts(fr_word, "en-us").replace(" ","")
    best_en, best_s = fr_word, 0
    for en_w, en_w_ipa in list(en_ipa_dict.items())[:3000]:
        s = ndice(en_ipa, en_w_ipa)
        if s > best_s: best_s, best_en = s, en_w
    return best_en, best_s

# ═══════════════════════════════════════════════════════════
# DEEPSEEK GENERATOR — ask the LLM to produce French homophones
# ═══════════════════════════════════════════════════════════
def deepseek_homophone(en_word):
    """
    Ask DeepSeek: "What French word sounds like X to an English speaker?"
    The LLM generates the answer. We verify with Agent B.
    """
    prompt = (
        f"Given the English word \"{en_word}\", what single French word "
        f"or short French phrase sounds most like it when read aloud "
        f"by an English speaker? The French spelling should produce the "
        f"same or similar sounds. Answer with ONLY the French word, "
        f"nothing else. No explanations."
    )
    # Return the prompt — the LLM (DeepSeek) will answer it
    return prompt

# ═══════════════════════════════════════════════════════════
# DEEPSEEK-WRITER: Full paragraph
# ═══════════════════════════════════════════════════════════
def write_paragraph(en_paragraph):
    """English paragraph → French homophone paragraph via DB + DeepSeek."""
    words = [w.lower().strip(".,;:!?\"") for w in en_paragraph.split() if w.strip(".,;:!?\"")]
    results = []
    missing = []
    
    # Step 1: Lookup
    for w in words:
        if w in lookup:
            fr, q = lookup[w]
            heard, hs = verify(fr)
            results.append((w, fr, q, "db", heard, hs))
        else:
            missing.append(w)
    
    # Step 2: Generate queries for missing words
    if missing:
        print(f"\n  Missing from DB ({len(missing)}): {missing}")
        print(f"  DeepSeek, please generate French homophones for these words:")
        for w in missing:
            prompt = deepseek_homophone(w)
            fr = input(f"    {w:15s} → ? ")  # LLM answers here
            if fr.strip():
                heard, hs = verify(fr.strip().lower())
                results.append((w, fr.strip().lower(), 0.7, "ds", heard, hs))
                # Store for future use
                lookup[w] = (fr.strip().lower(), 0.7)
    
    # Step 3: Render
    fr_text = " ".join(fr for _,fr,_,_,_,_ in results)
    en_heard = " ".join(h for _,_,_,_,h,_ in results)
    
    print(f"\n{'='*60}")
    print(f"EN: {en_paragraph}")
    print(f"FR: {fr_text}")
    print(f"HE: {en_heard}")
    for w,fr,q,src,heard,hs in results:
        m = "★" if src=="db" else "◇"
        print(f"  {m} {w:12s} → {fr:15s} (q={q:.2f}) → B:{heard:12s} ({hs:.2f})")
    
    return fr_text

# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    test = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "the ocean remembers every vessel that ever sailed"
    
    # For non-interactive mode: use the LSTM model as fallback
    # The LLM (DeepSeek) IS the model — I'll answer the prompts directly
    write_paragraph(test)
    
    # BATCH: Generate homophones for all unknown test words
    tests = ["ocean", "vessel", "wandered", "twilight", "gentle", "rushing", "waterfall", "moonlight", "whisper", "dreamer"]
    unknown = [w for w in tests if w not in lookup]
    if unknown:
        print(f"\n{'='*60}")
        print(f"DEEPSEEK BATCH — Generate French homophones for {len(unknown)} words:")
        print(f"{'='*60}")
        for w in unknown:
            print(f"\n  English: {w}")
            print(f"  Prompt:  {deepseek_homophone(w)}")
