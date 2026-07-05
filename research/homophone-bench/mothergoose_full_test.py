"""Unified Mother Goose test: every method, no cutting.

For each public-domain Mother Goose line we GENERATE a French carve two ways
(frontier whole-line carve + baseline decoder), then score each carve with ALL
the scorers we have, side by side:
  - symbolic combo   (matcher, AUC 0.993) -- sound
  - MFCC-DTW audio   (the acoustic / Whisper-family scorer, ffmpeg-free)
  - bigram coherence (L2 fluency)
This is REPRESENTATION.md's recommended use of the audio path: a re-ranker on
finished output, run beside the symbolic score, never load-bearing -- now actually
run on the generated carves rather than dropped.

Round Rabbit (Fable's lattice) themes the rhyme's content words separately: the
already-homophonic neighbours that make good carve material.

Public-domain source only (The Real Mother Goose, 1916). French is generated.

Run: python mothergoose_full_test.py
"""
from __future__ import annotations

import subprocess
import wave

import numpy as np

import matcher
import phonetic_decoder as pd
import poetry_mode as pm
from lexicon_g2p import clean_ipa

try:
    import bigram_lm
    LM = bigram_lm.load("fr")
except Exception:
    LM = None

import bench                                    # for the MFCC-DTW audio scorer


def load_mono16k_stdlib(path):                  # ffmpeg-free wav loader
    w = wave.open(path, "rb"); sr, n = w.getframerate(), w.getnframes()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    if sr != 16000 and len(x) > 1:
        m = int(len(x) * 16000 / sr)
        x = np.interp(np.linspace(0, len(x), m, endpoint=False),
                      np.arange(len(x)), x).astype(np.float32)
    return x


bench.load_mono16k = load_mono16k_stdlib

_OC, _OG = dict(matcher.CHEAP_GAP), matcher.GAP


def en_ipa(t):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", t],
                       capture_output=True, text=True, check=True)
    return clean_ipa(r.stdout.strip())


def coherence(fr):
    toks = [w.lower() for w in fr.replace("'", " ").split() if w]
    return LM.fluency(toks) if (LM and toks) else 0.0


def scores(line, fr):
    if not fr:
        return 0.0, 0.0, 0.0
    combo = matcher.homophone_score(line, "en", fr, "fr")["score"]
    try:
        audio = float(bench.m_mfcc_dtw(line, fr))
    except Exception:
        audio = float("nan")
    return combo, audio, coherence(fr)


def baseline(line, root):
    matcher.CHEAP_GAP.clear(); matcher.CHEAP_GAP.update(_OC); matcher.GAP = _OG
    pd._sub.cache_clear()
    pd.MIN_WORD_SEGS = 2; pd.WORD_PENALTY = 0.18; pd.BEAM = 250
    c = pd.decode(en_ipa(line), root, top_n=8, max_words=10,
                  lm=LM, lm_weight=pd.LM_BEAM_WEIGHT if LM else 0.0)
    return c[0]["fr"] if c else ""


def frontier(line, root):
    """Whole-line carve with a small learned-style scale sweep (best dual)."""
    pd.MIN_WORD_SEGS = 1; pd.WORD_PENALTY = 0.04; pd.BEAM = 420
    nw = len(line.split()); best, bestdual = "", -1
    for scale in (1.0, 1.8, 2.6):
        for k, v in _OC.items():
            matcher.CHEAP_GAP[k] = v if k == "h" else min(0.95, v * scale)
        matcher.GAP = min(0.95, _OG * scale); pd._sub.cache_clear()
        for c in pd.decode(en_ipa(line), root, top_n=25, max_words=nw + 3,
                           lm=LM, lm_weight=pd.LM_BEAM_WEIGHT if LM else 0.0):
            if c["coverage"] < 0.7:
                continue
            cb = matcher.homophone_score(line, "en", c["fr"], "fr")["score"]
            d = cb * coherence(c["fr"])
            if d > bestdual:
                best, bestdual = c["fr"], d
    return best


LINES = ["Humpty Dumpty", "Humpty Dumpty sat on a wall", "Jack and Jill",
         "Hickory dickory dock", "Pat a cake"]


def main():
    base_root = pd.build_trie(min_zipf=2.2, lang="fr")
    poet_root = pm.build_poetry_trie(min_zipf=2.0)
    print("Unified Mother Goose test -- carve + score with ALL methods.\n")
    print(f"{'line / strategy':34s} {'FR carve':26s} {'snd':>5s} {'audio':>6s} {'coh':>5s}")
    print("-" * 84)
    for line in LINES:
        for label, fr in (("baseline", baseline(line, base_root)),
                          ("frontier", frontier(line, poet_root))):
            cb, au, co = scores(line, fr)
            au_s = f"{au:.2f}" if au == au else "  - "
            head = f"{line[:22]:22s} {label:9s}"
            print(f"{head:34s} {fr[:26]:26s} {cb:5.2f} {au_s:>6s} {co:5.2f}")
        print()
    print("""Reading: each carve is scored by the symbolic matcher (sound), the MFCC-DTW
acoustic scorer (the Whisper-family audio path, run as a re-ranker per
REPRESENTATION.md), and bigram coherence. The audio column tends to agree with
sound on tight matches and is noisier on multi-word carves (synthesized-speech
prosody) -- which is exactly why it is a soft re-ranker, not the judge. Combine
with Round Rabbit (round_rabbit_run.py) for themed content selection.""")


if __name__ == "__main__":
    main()
