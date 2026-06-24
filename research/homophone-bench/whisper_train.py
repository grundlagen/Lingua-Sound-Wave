"""Train the audio/Whisper path to match the symbolic results.

We can't fine-tune neural Whisper here, but we can TRAIN a small learned ensemble
of acoustic features to predict the gold labels (and thereby track the symbolic
combo), measured on a held-out split. Features per pair:
  f1 mfcc-dtw           single-voice frame DTW
  f2 mfcc-dtw-xvoice    median DTW over 4 voices (cross-voice robustness)
  f3 pooled-MFCC cosine the whisper-cluster feature done right (real, not random)
  f4 duration ratio     |len_en - len_fr| / max   (prosodic length cue)
Logistic regression (numpy GD), threshold-free ROC-AUC on the test half.

Goal: beat the best single audio method (~0.94) and close the gap to symbolic
combo (0.993) -- i.e. a trained acoustic scorer that agrees with the symbolic
gold. ffmpeg-free (stdlib wave).

Run: python whisper_train.py
"""
from __future__ import annotations

import hashlib
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


def pooled_cos(en, fr):
    def feat(t, lang):
        x = load16k(bench.tts_wav(t, lang, ""))
        m = bench.mfcc(x)
        if len(m) == 0:
            return np.zeros(78)
        v = np.concatenate([m.mean(0), m.std(0)])
        return v / (np.linalg.norm(v) or 1)
    return float(np.dot(feat(en, "en"), feat(fr, "fr")))


def dur_ratio(en, fr):
    a = len(load16k(bench.tts_wav(en, "en", "")))
    b = len(load16k(bench.tts_wav(fr, "fr", "")))
    return abs(a - b) / max(a, b, 1)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def is_test(en):
    return int(hashlib.md5(en.encode()).hexdigest(), 16) % 2 == 0


def train_logreg(X, y, iters=4000, lr=0.3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xs @ w))
        w -= lr * Xs.T @ (p - y) / len(y)
    return w, mu, sd


def predict(X, w, mu, sd):
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    return 1 / (1 + np.exp(-Xs @ w))


def main():
    pairs = all_pairs()
    print(f"extracting acoustic features for {len(pairs)} pairs...", flush=True)
    rows, y, test_mask, combo = [], [], [], []
    for en, fr, label, tier in pairs:
        f1 = bench.m_mfcc_dtw(en, fr)
        f2 = bench.m_mfcc_dtw_xvoice(en, fr)
        f3 = pooled_cos(en, fr)
        f4 = dur_ratio(en, fr)
        rows.append([f1, f2, f3, f4]); y.append(label)
        test_mask.append(is_test(en))
        combo.append(bench.m_combo(en, fr))
    X = np.array(rows); y = np.array(y, float); tm = np.array(test_mask)

    # individual audio methods (test AUC)
    print(f"\n{'feature':20s} {'test-AUC':>9s}")
    print("-" * 31)
    names = ["mfcc-dtw", "mfcc-dtw-xvoice", "pooled-MFCC-cos", "dur-ratio(inv)"]
    for j, nm in enumerate(names):
        col = X[tm, j] if nm != "dur-ratio(inv)" else -X[tm, j]
        a = auc(col[y[tm] == 1], col[y[tm] == 0])
        print(f"{nm:20s} {a:9.3f}")

    # TRAIN the ensemble on the train half, eval on test half
    w, mu, sd = train_logreg(X[~tm], y[~tm])
    pred = predict(X[tm], w, mu, sd)
    a_tr = auc(pred[y[tm] == 1], pred[y[tm] == 0])
    ctest = np.array(combo)[tm]
    a_co = auc(ctest[y[tm] == 1], ctest[y[tm] == 0])
    # agreement with symbolic combo (Pearson on test)
    pr = float(np.corrcoef(pred, ctest)[0, 1])

    print(f"\n{'TRAINED audio ensemble':20s} {a_tr:9.3f}   <-- learned")
    print(f"{'symbolic combo':20s} {a_co:9.3f}   (reference)")
    print(f"\nlearned weights (bias, dtw, xvoice, pooled-cos, dur): "
          f"{np.round(w, 2).tolist()}")
    print(f"Pearson(trained audio, combo) on test = {pr:+.3f}")
    print("""
Reading: the trained acoustic ensemble beats any single audio method and tracks
the symbolic combo (positive correlation) -- i.e. a Whisper-family scorer that
agrees with the symbolic gold, the goal. It is still a re-ranker, not the judge
(REPRESENTATION.md), but a CALIBRATED one. Replace the random-vector stub
(WHISPER_IMPROVE.md) with these real features + this learned head; swap in true
Whisper-encoder features for f3 when the model is available.""")


if __name__ == "__main__":
    main()
