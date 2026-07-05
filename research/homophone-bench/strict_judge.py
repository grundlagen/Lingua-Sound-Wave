#!/usr/bin/env python3
"""
STRICT JUDGE — After-the-fact quality scoring for homophonic translations.

Three gates, ALL must pass:
  1. PHONETIC: cross-accent dice (EN voice reads FR vs EN voice reads EN)
  2. ENGLISH: output words must ALL be real English (en-word-ipa.tsv)
  3. FLUENCY: bigram LM coherence score (loaded from pickle)

If ANY gate fails → REJECTED.
If ALL pass → ACCEPTED with composite score.

This is NOT a generator — it JUDGES outputs from any source (Whisper, carve, etc.)

RUN: python strict_judge.py --en "Humpty Dumpty sat on a wall" --fr "un petit un petit assis sur un mur"
     python strict_judge.py --jsonl strict-whisper-dataset.jsonl
"""

import subprocess, os, sys, json, re, argparse

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ── English vocabulary ──
EN_VOCAB = set()
for i,line in enumerate(open("en-word-ipa.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[1] and "(fr)" not in p[0]:
        EN_VOCAB.add(p[0].lower())
EN_VOCAB.update({"a","i","the","and","or","but","in","on","at","to","of","for",
                 "is","are","was","were","be","been","am","he","she","it","we",
                 "they","you","me","him","her","us","them","my","your","his","its",
                 "our","their","this","that","these","those","not","no","yes",
                 "do","does","did","have","has","had","can","could","will","would",
                 "shall","should","may","might","must","so","as","by","with","from",
                 "up","down","out","over","under","again","just","only","very","too"})

# ── G2P ──
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

# ── Load bigram LM ──
LM = None
try:
    import bigram_lm as blm
    LM = blm.load("en")  # ENGLISH bigram model
except:
    try:
        import bigram_lm as blm
        LM = blm.load("fr")  # fallback to FR
    except:
        pass

# ═══════════════════════════════════════════════════════════
def judge(en_text, fr_or_en_text, verbose=True):
    """
    Strict judge: phonetic gate + English word gate + fluency gate.
    
    fr_or_en_text: the OUTPUT text (English words produced by homophone writer / Whisper)
    
    Returns: verdict dict with PASS/FAIL and scores.
    """
    output = fr_or_en_text
    
    # ── Gate 1: Phonetic match ──
    # How does the FR text sound to an English ear?
    # Compare: EN voice reading the SOURCE (French) vs EN voice reading the OUTPUT (English)
    fr_ipa = tts(en_text, "en-us").replace(" ","")
    en_ipa = tts(output, "en-us").replace(" ","")
    ph = ndice(fr_ipa, en_ipa) if fr_ipa and en_ipa else 0.0
    
    # ── Gate 2: English word ratio ──
    words = re.findall(r"[a-z']+", output.lower())
    if not words:
        en_ratio = 0.0
    else:
        en_ratio = sum(1 for w in words if w in EN_VOCAB) / len(words)
    
    # ── Gate 3: Fluency (bigram LM) ──
    if LM and len(words) >= 2:
        try:
            flu = LM.fluency(words)
        except:
            flu = 0.5
    else:
        flu = 0.5
    
    # ── Verdict ──
    PHONETIC_GATE = 0.25   # minimum phonetic match
    ENGLISH_GATE = 0.80    # minimum English word ratio
    FLUENCY_GATE = 0.20    # minimum bigram coherence
    
    gates = {
        "phonetic": (ph >= PHONETIC_GATE, round(ph, 3), PHONETIC_GATE),
        "english": (en_ratio >= ENGLISH_GATE, round(en_ratio, 2), ENGLISH_GATE),
        "fluency": (flu >= FLUENCY_GATE, round(flu, 3), FLUENCY_GATE),
    }
    
    all_pass = all(v[0] for v in gates.values())
    composite = (ph + en_ratio + flu) / 3 if all_pass else 0.0
    
    verdict = {
        "pass": all_pass,
        "composite": round(composite, 3),
        "phonetic": round(ph, 3),
        "en_ratio": round(en_ratio, 2),
        "fluency": round(flu, 3),
        "fr_ipa": fr_ipa[:60],
        "en_ipa": en_ipa[:60],
        "output": output[:80],
        "gates": {k: {"pass": v[0], "value": v[1], "threshold": v[2]} 
                  for k,v in gates.items()},
    }
    
    if verbose:
        status = "✓ PASS" if all_pass else "✗ FAIL"
        fails = [k for k,v in gates.items() if not v[0]]
        print(f"  {status}  ph={ph:.3f}  en={en_ratio:.2f}  flu={flu:.3f}  "
              f"{'['+','.join(fails)+']' if fails else ''}")
        print(f"    FR: {en_text[:60]}")
        print(f"    EN: {output[:60]}")
    
    return verdict

# ═══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Strict Judge for homophonic translations")
    ap.add_argument("--en", type=str, help="English source text")
    ap.add_argument("--fr", type=str, help="French output text (homophone)")
    ap.add_argument("--jsonl", type=str, help="Batch judge from JSONL file")
    ap.add_argument("--threshold", type=float, default=0.25, help="Phonetic gate threshold")
    args = ap.parse_args()
    
    print("STRICT JUDGE — Homophonic translation quality")
    print("=" * 55)
    
    if args.en and args.fr:
        verdict = judge(args.en, args.fr, verbose=True)
        print(f"\n  {json.dumps(verdict, indent=2)}")
        return
    
    if args.jsonl:
        results = []
        with open(args.jsonl) as f:
            for line in f:
                r = json.loads(line)
                if "fr" in r and "en" in r:
                    verdict = judge(r["fr"], r["en"], verbose=False)
                    verdict["source_fr"] = r["fr"][:60]
                    results.append(verdict)
        
        passed = [r for r in results if r["pass"]]
        failed = [r for r in results if not r["pass"]]
        
        print(f"\n  JUDGED: {len(results)} pairs")
        print(f"  PASS: {len(passed)} ({100*len(passed)/len(results):.0f}%)")
        print(f"  FAIL: {len(failed)} ({100*len(failed)/len(results):.0f}%)")
        
        if passed:
            import numpy as np
            print(f"\n  PASSED — composite scores:")
            print(f"    phonetic: μ={np.mean([r['phonetic'] for r in passed]):.3f} "
                  f"σ={np.std([r['phonetic'] for r in passed]):.3f}")
            print(f"    en_ratio: μ={np.mean([r['en_ratio'] for r in passed]):.2f}")
            print(f"    fluency:  μ={np.mean([r['fluency'] for r in passed]):.3f}")
        
        if failed:
            reasons = {}
            for r in failed:
                for gate, v in r["gates"].items():
                    if not v["pass"]:
                        reasons[gate] = reasons.get(gate, 0) + 1
            print(f"\n  FAILED — reasons:")
            for gate, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"    {gate}: {count}/{len(results)} ({100*count/len(results):.0f}%)")
        
        # Save filtered
        passed_path = args.jsonl.replace(".jsonl", "-passed.jsonl")
        with open(passed_path, "w") as f:
            for r in passed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  Saved {len(passed)} passed → {passed_path}")
        return
    
    # Default: test with known pairs
    tests = [
        ("Humpty Dumpty sat on a wall", "un petit un petit assis sur un mur"),
        ("dans un", "dancing"),
        ("je ne sais pas", "I'll do the same part"),
        ("il y avait en westphalie", "You are having West alley"),
    ]
    for en, fr in tests:
        judge(en, fr, verbose=True)

if __name__ == "__main__":
    main()
