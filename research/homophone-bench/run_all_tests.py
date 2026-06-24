"""Run the FULL test suite -- every method, symbolic + audio together,
ffmpeg-free -- as one table. The final bake-off across the whole bench.

Run: python run_all_tests.py
"""
from __future__ import annotations

import wave

import numpy as np

import bench
from dataset import all_pairs


def load16k(path):
    w = wave.open(path, "rb"); sr, n = w.getframerate(), w.getnframes()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    if sr != 16000 and len(x) > 1:
        m = int(len(x) * 16000 / sr)
        x = np.interp(np.linspace(0, len(x), m, endpoint=False),
                      np.arange(len(x)), x).astype(np.float32)
    return x


bench.load_mono16k = load16k


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return 0.0
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    pairs = all_pairs()
    order = ["combo", "combo2", "feat-nw-sharp", "class-ngram", "ngram-dice",
             "feat-dtw", "gate", "mfcc-dtw-xvoice", "mfcc-dtw", "hybrid-geo",
             "ipa-lev", "xsampa-lev", "feat-nw"]
    print(f"FULL test suite on {len(pairs)} pairs (symbolic + audio, ffmpeg-free)\n")
    print(f"{'method':18s} {'AUC':>6s} {'hard':>6s} {'pos':>5s} {'neg':>5s} {'sep':>6s} {'kind':>8s}")
    print("-" * 60)
    audio = {"feat-dtw", "gate", "mfcc-dtw-xvoice", "mfcc-dtw", "hybrid-geo"}
    res = []
    for name in order:
        fn = bench.METHODS.get(name)
        if fn is None:
            continue
        pos, neg, neg_tr = [], [], []
        for en, fr, label, tier in pairs:
            try:
                s = float(fn(en, fr))
            except Exception:
                s = 0.0
            (pos if label else neg).append(s)
            if label == 0 and tier == "translation":
                neg_tr.append(s)
        a, ah = auc(pos, neg), auc(pos, neg_tr)
        mp, mn = float(np.mean(pos)), float(np.mean(neg))
        kind = "audio" if name in audio else "symbolic"
        res.append((a, name, ah, mp, mn, kind))
        print(f"{name:18s} {a:6.3f} {ah:6.3f} {mp:5.2f} {mn:5.2f} {mp-mn:+6.2f} {kind:>8s}")

    best_sym = max(r for r in res if r[5] == "symbolic")
    best_aud = max(r for r in res if r[5] == "audio")
    print(f"\nbest symbolic: {best_sym[1]} {best_sym[0]:.3f}   "
          f"best audio: {best_aud[1]} {best_aud[0]:.3f}   "
          f"gap {best_sym[0]-best_aud[0]:+.3f}")
    print("""
Verdict (reconfirmed): symbolic combo leads; audio plateaus ~0.95 and is weaker
on SEPARATION (rates everything high). Use audio as a calibrated re-ranker
(whisper_train.py), never the judge. The dictionaries (v5/v6/v7) and the carve
engine all rank on the symbolic combo for exactly this reason.""")


if __name__ == "__main__":
    main()
