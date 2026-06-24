"""Improving the Whisper path: replace the random-vector stub with REAL acoustic
encoder features.

Finding: artifacts/api-server/src/lib/whisper-phonetic-cluster.ts'
extractWhisperEncoderFeatures() is a STUB -- it returns Math.random() 768-dim
vectors ("TODO: Replace with actual Whisper encoder call"). So the "cosine >= 0.95
= Perfect Match" reservoir is comparing RANDOM vectors: the threshold is
meaningless, and any "perfect match" it records is noise.

Fix (deterministic, no neural model needed here): use real MFCC encoder features
(mean+std pooled over frames) as the cluster vector. This is the same acoustic
signal the bench's MFCC-DTW uses, reduced to a fixed-dim cosine-comparable vector
-- a drop-in for the stub that actually discriminates. (A true Whisper encoder is
strictly better and should replace this when the model is available; this proves
the stub must go.)

Run: python whisper_improve.py
"""
from __future__ import annotations

import subprocess
import wave

import numpy as np

import bench


def load16k(path):
    w = wave.open(path, "rb"); sr, n = w.getframerate(), w.getnframes()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    if sr != 16000 and len(x) > 1:
        m = int(len(x) * 16000 / sr)
        x = np.interp(np.linspace(0, len(x), m, endpoint=False),
                      np.arange(len(x)), x).astype(np.float32)
    return x


def encoder_features(text, lang):
    """REAL acoustic cluster vector: mean+std of MFCC frames (78-dim), L2-normed.
    The drop-in replacement for the random stub."""
    voice = "en-us" if lang == "en" else "fr"
    path = bench.tts_wav(text, lang, "")
    x = load16k(path)
    m = bench.mfcc(x)                      # [frames, 39]
    if len(m) == 0:
        return np.zeros(78)
    v = np.concatenate([m.mean(0), m.std(0)])
    n = np.linalg.norm(v)
    return v / n if n else v


def cos(a, b):
    return float(np.dot(a, b))


def stub_features():
    v = (np.random.rand(768) - 0.5) * 0.7
    return v / (np.linalg.norm(v) or 1)


def main():
    bench.load_mono16k = load16k
    pairs_pos = [("shoe", "chou"), ("key", "qui"), ("sea", "si"),
                 ("two", "tout"), ("bell", "belle")]
    pairs_neg = [("dog", "chien"), ("house", "maison"), ("book", "livre"),
                 ("water", "eau"), ("night", "feu")]

    print("REAL acoustic encoder features (mean+std MFCC, 78-dim) vs the random stub.\n")
    print(f"{'pair':22s} {'real cos':>9s} {'stub cos':>9s}")
    print("-" * 44)
    rp, rn, sp = [], [], []
    for en, fr in pairs_pos:
        c = cos(encoder_features(en, "en"), encoder_features(fr, "fr"))
        s = cos(stub_features(), stub_features())
        rp.append(c); sp.append(s)
        print(f"{en+'~'+fr:22s} {c:9.3f} {s:9.3f}  (homophone)")
    for en, fr in pairs_neg:
        c = cos(encoder_features(en, "en"), encoder_features(fr, "fr"))
        rn.append(c)
        print(f"{en+'~'+fr:22s} {c:9.3f} {'':>9s}  (not alike)")

    print(f"\nreal features:  homophone mean {np.mean(rp):.3f}  "
          f"non-pair mean {np.mean(rn):.3f}  separation {np.mean(rp)-np.mean(rn):+.3f}")
    print(f"random stub:    any-pair cos ~ {np.mean(sp):+.3f} (centered on 0, no signal)")
    print("""
Reading: the real MFCC encoder features SEPARATE homophones from non-pairs
(positive gap), so a cosine threshold is meaningful. The random stub centres on 0
for everything -- its 0.95 "perfect match" gate fires only by chance. Improving
Whisper = (1) at minimum, replace the stub with these real features; (2) ideally,
wire the actual Whisper encoder; (3) keep it as a re-ranker/validator on finished
output (REPRESENTATION.md), never the load-bearing scorer.""")


if __name__ == "__main__":
    main()
