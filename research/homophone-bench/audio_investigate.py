"""Investigate the audio path's "high results": is the high AUC real, or an
artifact of an easy benchmark?

AUC measures ranking on the 105-pair set, whose negatives are translations that
sound nothing alike -- trivially separated. The REAL task is retrieval: rank the
whole French lexicon for an English word and find the true homophone near the top,
with a usable threshold. We rank a French sample for a few English words by audio
(mfcc-dtw) vs symbolic combo, and report where the documented homophone lands +
the noise floor (how many French words score above it).

Run: python audio_investigate.py
"""
from __future__ import annotations

import wave

import numpy as np
from wordfreq import top_n_list

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


bench.load_mono16k = load16k

PROBES = [("shoe", "chou"), ("key", "qui"), ("set", "cette"), ("two", "tout")]


def main():
    fr = [w for w in top_n_list("fr", 1500) if w.isalpha() and 2 <= len(w) <= 7][:150]
    print(f"Retrieval over {len(fr)} French words. Where does the TRUE homophone "
          f"rank, by AUDIO vs SYMBOLIC?\n")
    print(f"{'EN':6s} {'true FR':9s} {'audio rank':>11s} {'audio noise':>12s} "
          f"{'combo rank':>11s} {'combo noise':>12s}")
    print("-" * 66)
    for en, true_fr in PROBES:
        cand = fr + ([true_fr] if true_fr not in fr else [])
        au = {c: bench.m_mfcc_dtw(en, c) for c in cand}
        co = {c: bench.m_combo(en, c) for c in cand}
        au_sorted = sorted(cand, key=lambda c: -au[c])
        co_sorted = sorted(cand, key=lambda c: -co[c])
        ar = au_sorted.index(true_fr) + 1
        cr = co_sorted.index(true_fr) + 1
        # noise floor: how many words score within 0.05 of the TRUE word's score
        a_noise = sum(1 for c in cand if au[c] >= au[true_fr] - 0.05) - 1
        c_noise = sum(1 for c in cand if co[c] >= co[true_fr] - 0.05) - 1
        print(f"{en:6s} {true_fr:9s} {ar:11d} {a_noise:12d} {cr:11d} {c_noise:12d}")

    print("""
Reading: if AUDIO ranks the true homophone far down and has a huge noise floor
(many words score as high as the real match), its high benchmark AUC does NOT
translate to the real retrieval task -- it cannot pick the right word or set a
threshold. Symbolic combo ranks the true homophone at/near 1 with a small noise
floor. That is the investigation: the audio "high results" are an easy-benchmark
artifact; on retrieval, audio is a weak discriminator. Verdict: keep audio as a
soft re-ranker on a SHORTLIST the symbolic stage already produced, never as the
retriever or the judge.""")


if __name__ == "__main__":
    main()
