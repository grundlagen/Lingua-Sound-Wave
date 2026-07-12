#!/usr/bin/env python3
"""
WHISPER HOMOPHONE MINER — Full-scale French → English via Whisper ASR.
Every Whisper misrecognition IS a homophonic translation candidate.

Uses faster-whisper with English language forced.
Processes French sentences, collects English transcriptions as training data.

RUN: python whisper_homophone_miner.py --n 100
"""

import subprocess, os, sys, tempfile, time, json, re
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_fr_sentences(n=200):
    """Load French sentences from corpus."""
    sentences = []
    for fp in ["/tmp/fr-candide.txt","/tmp/fr-monte-cristo.txt",
               "/tmp/fr-les-mis.txt","/tmp/fr-vingt-mille.txt"]:
        try:
            txt = open(fp,encoding="utf-8",errors="ignore").read()
            txt = re.sub(r'\*\*\*.*?\*\*\*', '', txt, flags=re.S)
            for s in re.split(r'[.!?]+', txt):
                s = s.strip()
                if 20 <= len(s) <= 200 and not s.startswith("***"):
                    sentences.append(s)
                    if len(sentences) >= n: return sentences
        except: continue
    return sentences

def generate_wav_batch(sentences, tmpdir):
    """Generate WAV files for all sentences."""
    paths = []
    for i, s in enumerate(sentences):
        wav = os.path.join(tmpdir, f"fr_{i:04d}.wav")
        subprocess.run(["espeak-ng","-v","fr","-w",wav,"-s","160",s],
                       capture_output=True, check=True)
        paths.append(wav)
    return paths

def transcribe_batch(wav_paths, model_size="small"):
    """Transcribe all WAVs through Whisper with lang=en."""
    from faster_whisper import WhisperModel
    
    use_cuda = False
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except: pass
    
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"
    
    print(f"  Loading Whisper {model_size} on {device} ({compute})...")
    model = WhisperModel(model_size, device=device, compute_type=compute)
    
    results = []
    for i, wav in enumerate(wav_paths):
        if i % 10 == 0: print(f"  {i}/{len(wav_paths)}...")
        try:
            segments, info = model.transcribe(wav, language="en", beam_size=5,
                                              vad_filter=True,
                                              vad_parameters={"threshold": 0.3})
            text = " ".join(s.text.strip() for s in segments)
            results.append({
                "fr_text": sentences[i],
                "en_text": text,
                "detected": info.language,
                "prob": info.language_probability,
            })
        except Exception as e:
            results.append({"fr_text": sentences[i], "en_text": "", "error": str(e)})
    
    return results

# ═══════════════════════════════════════════════════════════════
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100, help="Number of sentences")
ap.add_argument("--model", default="small", help="Whisper model (tiny/base/small/medium)")
args = ap.parse_args()

print("WHISPER HOMOPHONE MINER")
print("=" * 60)

sentences = load_fr_sentences(args.n)
print(f"Loaded {len(sentences)} French sentences")

with tempfile.TemporaryDirectory() as tmpdir:
    print("Generating speech WAVs...")
    wavs = generate_wav_batch(sentences, tmpdir)
    print(f"Generated {len(wavs)} WAVs")
    
    print("Transcribing with Whisper (lang=en forced)...")
    results = transcribe_batch(wavs, args.model)
    
    # ── Show results ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(results)} French → English transcriptions")
    print(f"{'='*60}")
    
    for r in results[:30]:
        en = r.get("en_text", "(error)")
        fr = r["fr_text"][:80]
        print(f"\nFR: {fr}")
        print(f"EN: {en}")
    
    # ── Stats ──
    valid = [r for r in results if r.get("en_text")]
    import numpy as np
    word_counts = [len(r["en_text"].split()) for r in valid]
    print(f"\n{'='*60}")
    print(f"STATS: {len(valid)}/{len(results)} valid transcriptions")
    print(f"  EN words/sentence: {np.mean(word_counts):.1f} ± {np.std(word_counts):.1f}")
    print(f"  Empty: {sum(1 for r in results if not r.get('en_text'))}")
    
    # ── Save training corpus ──
    output = []
    for r in valid:
        output.append({
            "instruction": f"Translate this English text to French-sounding English (homophonic translation)",
            "input": r["fr_text"],
            "output": r["en_text"],
            "source": "whisper_asr_misrecognition",
        })
    
    path = "train-whisper-asr.jsonl"
    with open(path, "w") as f:
        for entry in output:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n  Saved {len(output)} rows → {path}")
    print(f"  → Self-learn training data from REAL Whisper misrecognitions")
