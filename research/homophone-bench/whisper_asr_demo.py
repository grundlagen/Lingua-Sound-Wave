#!/usr/bin/env python3
"""
WHISPER WRONG-LANGUAGE ASR — Real speech recognition on French audio.

Generates French speech via espeak-ng WAV, then runs faster-whisper
with English language setting to see what English text it "hears."

This is the real deal — not simulation. Whisper is a production ASR
that models connected speech, and when fed French audio with lang=en,
it will output its best English-guess transcription.

RUN: python whisper_asr_demo.py
"""

import subprocess, os, sys, tempfile, time

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ── French test phrases ──
FRENCH_TESTS = [
    # Single words (should produce clear English guesses)
    "petit",
    "dans un",
    "vous êtes",
    "tout le monde",
    # Phrases
    "un petit d'un petit",
    "je ne sais pas",
    "c'est la vie",
    "merci beaucoup",
    # Whole sentences from Candide
    "il y avait en westphalie un jeune garçon",
    "tout est pour le mieux dans le meilleur des mondes possibles",
    "il faut cultiver notre jardin",
    # Edge cases: what does English Whisper hear?
    "pain au chocolat",
    "voulez vous coucher avec moi ce soir",
    "la plume de ma tante",
]

def generate_wav(french_text, out_path):
    """Generate French speech WAV via espeak-ng."""
    subprocess.run([
        "espeak-ng", "-v", "fr", "-w", out_path,
        "-s", "150",  # speed (words per minute)
        french_text
    ], capture_output=True, check=True)
    return os.path.getsize(out_path)

def run_whisper(wav_path, model_size="tiny"):
    """Run faster-whisper with English language on the WAV."""
    from faster_whisper import WhisperModel
    
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(wav_path, language="en", beam_size=5)
    
    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "tokens": [],
            "avg_logprob": segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0,
        })
    
    return results, info

# ═══════════════════════════════════════════════════════════════
print("WHISPER WRONG-LANGUAGE ASR — French speech → English transcription")
print("=" * 65)

# Wait for faster-whisper to be installed
print("Waiting for faster-whisper...")
for _ in range(30):
    try:
        import faster_whisper
        print("  faster-whisper ready.")
        break
    except ImportError:
        time.sleep(2)
else:
    print("  faster-whisper not installed. Trying alternative...")
    # Try openai-whisper
    try:
        import whisper
        print("  openai-whisper available.")
    except ImportError:
        print("  No whisper available. Install: pip3 install faster-whisper")
        sys.exit(1)

with tempfile.TemporaryDirectory() as tmpdir:
    for phrase in FRENCH_TESTS:
        wav_path = os.path.join(tmpdir, "fr_audio.wav")
        
        # Generate speech
        try:
            size = generate_wav(phrase, wav_path)
        except Exception as e:
            print(f"  SKIP: {phrase[:40]} — {e}")
            continue
        
        # Run Whisper with English language
        try:
            segments, info = run_whisper(wav_path, "tiny")
        except Exception as e:
            print(f"  WHISPER ERROR: {e}")
            continue
        
        # Get transcription
        en_text = " ".join(s["text"] for s in segments)
        
        print(f"\n{'─'*65}")
        print(f"FR: {phrase}")
        print(f"EN (Whisper hears): \"{en_text}\"")
        print(f"  detected lang: {info.language} (prob={info.language_probability:.3f})")
        if segments:
            print(f"  segments: {' | '.join(s['text'] for s in segments)}")
        print(f"  audio: {size/1000:.1f}KB, {info.duration:.1f}s")

print(f"\n{'='*65}")
print("DONE — These are the RAW English transcriptions of French speech.")
print("This is what a badly-configured Whisper produces when it thinks")
print("French is English. Every output word IS a homophonic translation.")
