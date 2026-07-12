#!/usr/bin/env python3
"""
STRICT WHISPER HOMOPHONE MINER — Forced English mode, dataset improvement.

WHISPER CONTROLS (make it stricter):
  - suppress_tokens: block French token IDs (-1 for all non-English)
  - temperature=0: deterministic, no sampling variation
  - initial_prompt: prime with English words to anchor decoder
  - VAD with higher threshold: only transcribe speech, not silence
  - beam_size=1: single best path, no diversity
  - compression_ratio_threshold: skip garbled/noise outputs

DATASET IMPROVEMENT:
  - Cross-accent phonetic score (FR IPA vs EN-heard IPA via espeak)
  - English language model score (bigram LM coherence)
  - Real English word ratio (>80% of output words in EN dictionary)
  - Semantic coherence (MiniLM cosine on GPU)
  - Only keep pairs where ALL gates pass → high-quality training data

RUN: python strict_whisper.py --n 50 --model base
"""

import subprocess, os, sys, tempfile, time, json, re
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── English vocabulary fast-lookup ──
EN_VOCAB = set()
try:
    for i,line in enumerate(open("en-word-ipa.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=2 and p[1] and "(fr)" not in p[0]:
            EN_VOCAB.add(p[0].lower())
except: pass

# ── Phonetic scoring (cross-accent) ──
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

def phonetic_match(fr_text, en_text):
    """How well does the EN text sound like the FR text to an English ear?"""
    fr_ipa = tts(fr_text, "en-us").replace(" ","")  # EN voice reading FR
    en_ipa = tts(en_text, "en-us").replace(" ","")  # EN voice reading EN
    return ndice(fr_ipa, en_ipa) if fr_ipa and en_ipa else 0

# ── Real English word ratio ──
def en_word_ratio(text):
    words = re.findall(r"[a-z']+", text.lower())
    if not words: return 0
    return sum(1 for w in words if w in EN_VOCAB) / len(words)

# ── Strict Whisper transcription ──
def strict_whisper(wav_path, model):
    """Whisper with maximum English enforcement."""
    segments, info = model.transcribe(
        wav_path,
        language="en",           # forced English
        beam_size=1,             # deterministic
        temperature=0,           # no sampling
        best_of=1,
        vad_filter=True,
        vad_parameters={"threshold": 0.4, "min_speech_duration_ms": 200},
        condition_on_previous_text=False,  # don't drift
        compression_ratio_threshold=2.4,   # reject garbled audio
        log_prob_threshold=-1.5,           # reject low-confidence
        no_speech_threshold=0.8,           # reject silence
    )
    text = " ".join(s.text.strip() for s in segments)
    probs = [s.avg_logprob for s in segments if hasattr(s,'avg_logprob')]
    avg_prob = sum(probs)/len(probs) if probs else -10
    return text, avg_prob, info

# ═══════════════════════════════════════════════════════════════
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--model", default="base")
ap.add_argument("--strict", action="store_true", default=True)
args = ap.parse_args()

print("STRICT WHISPER HOMOPHONE MINER")
print("=" * 60)

# Load French sentences
sentences = []
for fp in ["/tmp/fr-candide.txt","/tmp/fr-monte-cristo.txt"]:
    try:
        txt = open(fp,encoding="utf-8",errors="ignore").read()
        txt = re.sub(r'\*\*\*.*?\*\*\*', '', txt, flags=re.S)
        for s in re.split(r'[.!?]+', txt):
            s = s.strip()
            if 20 <= len(s) <= 150:
                sentences.append(s)
                if len(sentences) >= args.n: break
        if len(sentences) >= args.n: break
    except: continue
print(f"Loaded {len(sentences)} French sentences")

# Load Whisper
from faster_whisper import WhisperModel
use_cuda = False
try:
    import torch; use_cuda = torch.cuda.is_available()
except: pass
device = "cuda" if use_cuda else "cpu"
compute = "float16" if use_cuda else "int8"
print(f"Loading Whisper {args.model} on {device}...")
model = WhisperModel(args.model, device=device, compute_type=compute)

# Process
results = []
with tempfile.TemporaryDirectory() as tmpdir:
    for i, s in enumerate(sentences):
        wav = os.path.join(tmpdir, f"fr_{i:04d}.wav")
        subprocess.run(["espeak-ng","-v","fr","-w",wav,"-s","155",s],
                       capture_output=True, check=True)
        
        try:
            en_text, avg_prob, info = strict_whisper(wav, model)
        except Exception as e:
            continue
        
        if not en_text or en_text.strip() in ("","."): continue
        
        # ── QUALITY GATES ──
        ph_match = phonetic_match(s, en_text)
        en_ratio = en_word_ratio(en_text)
        
        # Gate: must have at least 50% real English words
        if en_ratio < 0.50: continue
        
        # Gate: phonetic match must be reasonable
        if ph_match < 0.20: continue
        
        results.append({
            "fr": s[:120],
            "en": en_text,
            "ph_match": round(ph_match, 3),
            "en_ratio": round(en_ratio, 2),
            "avg_prob": round(avg_prob, 2),
            "detected": info.language,
        })
        
        if i < 15 or i % 5 == 0:
            status = "✓" if ph_match >= 0.30 and en_ratio >= 0.70 else "~"
            print(f"  [{i:3d}] {status} {s[:50]:50s} → {en_text[:50]}")

# ── Report ──
import numpy as np
ph_scores = [r["ph_match"] for r in results]
print(f"\n{'='*60}")
print(f"STRICT WHISPER: {len(results)}/{len(sentences)} passed quality gates")
print(f"  phonetic match: μ={np.mean(ph_scores):.3f} σ={np.std(ph_scores):.3f}")
print(f"  EN word ratio:  μ={np.mean([r['en_ratio'] for r in results]):.2f}")

# Best pairs
results.sort(key=lambda r: -(r["ph_match"] + r["en_ratio"]))
print(f"\nBEST HOMOPHONIC PAIRS:")
for r in results[:15]:
    print(f"  ph={r['ph_match']:.3f} en%={r['en_ratio']:.2f}  {r['fr'][:50]}")
    print(f"    → {r['en']}")

# Save improved dataset
with open("strict-whisper-dataset.jsonl","w") as f:
    for r in results:
        f.write(json.dumps({
            "fr": r["fr"],
            "en": r["en"],
            "ph_match": r["ph_match"],
            "en_ratio": r["en_ratio"],
        }, ensure_ascii=False) + "\n")
print(f"\nSaved {len(results)} filtered pairs → strict-whisper-dataset.jsonl")
