"""Run the older Lingua Weaver SPEECH/WAV tests (bench.py audio methods) without
ffmpeg -- espeak already writes WAV, and stdlib `wave` reads it.

Patches bench.load_mono16k to decode espeak's 22050 Hz mono int16 WAV with the
`wave` module + a NumPy linear resample to 16 kHz (what mfcc() assumes), then runs
every audio method on the 105-pair benchmark and reports ROC-AUC beside the
symbolic combo. This is the acoustic route the production app also runs via
Whisper/wav2vec (artifacts/api-server/src/lib/{whisper-phonetic-cluster,wav2vec}.ts);
those need the TS app + neural models, so here we run the deterministic MFCC-DTW
family that the same bench already defines.

Run: python run_wav_tests.py
"""
from __future__ import annotations

import wave

import numpy as np

import bench


def load_mono16k_stdlib(path: str) -> np.ndarray:
    w = wave.open(path, "rb")
    sr, n = w.getframerate(), w.getnframes()
    raw = w.readframes(n)
    w.close()
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000 and len(x) > 1:                     # linear resample to 16 kHz
        m = int(len(x) * 16000 / sr)
        x = np.interp(np.linspace(0, len(x), m, endpoint=False),
                      np.arange(len(x)), x).astype(np.float32)
    return x


bench.load_mono16k = load_mono16k_stdlib            # the ffmpeg-free patch


def main():
    # sanity: can we synth + load a wav at all?
    try:
        s = bench.load_mono16k(bench.tts_wav("test", "en", ""))
        print(f"wav loader OK: {len(s)} samples @16k (stdlib wave, no ffmpeg)\n")
    except Exception as e:
        print(f"wav loader FAILED: {e}")
        return

    pairs = bench.all_pairs()
    audio_methods = ["feat-dtw", "mfcc-dtw", "mfcc-dtw-xvoice", "hybrid-geo", "gate"]
    print(f"running {len(audio_methods)} speech/wav methods on {len(pairs)} pairs "
          f"+ symbolic combo for reference\n")
    print(f"{'method':18s} {'AUC':>6s} {'AUC_hard':>9s} {'pos':>5s} {'neg':>5s} {'sep':>6s}")
    print("-" * 56)

    for name in audio_methods + ["combo"]:
        fn = bench.METHODS[name]
        pos, neg, neg_tr = [], [], []
        ok = True
        for en, fr, label, tier in pairs:
            try:
                s = float(fn(en, fr))
            except Exception as e:
                print(f"{name:18s} FAILED on ({en!r},{fr!r}): {type(e).__name__}: {e}")
                ok = False
                break
            (pos if label else neg).append(s)
            if label == 0 and tier == "translation":
                neg_tr.append(s)
        if not ok:
            continue
        auc = bench.auc(pos, neg)
        auc_h = bench.auc(pos, neg_tr)
        mp, mn = float(np.mean(pos)), float(np.mean(neg))
        print(f"{name:18s} {auc:6.3f} {auc_h:9.3f} {mp:5.2f} {mn:5.2f} {mp-mn:+6.2f}")

    print("""
Reading: these are the acoustic (synthesized-speech MFCC-DTW) homophone scorers
from the older bench, now runnable here. Compare their AUC to symbolic combo
(~0.99). RESULTS.md's finding was that audio scores ~0.93-0.94 and far worse on
loose positives -- this run lets you re-confirm it on the spot. The production
Whisper/wav2vec path is the neural version of the same acoustic idea and needs the
TS app + models (not in this pure-Python env).""")


if __name__ == "__main__":
    main()
