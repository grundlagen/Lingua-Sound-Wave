"""REAL-AUDIO French G2P -- the Common Voice track, running HERE (no Colab).

The Drive plan (Untitled7.ipynb) needed Common Voice (HF-gated) + torchaudio
(absent). Both dodged:
  audio      Tatoeba French clips (CC, per-sentence mp3, curl-able) with
             known transcripts -- real human speech, no gate.
  loading    ffmpeg -> numpy (bench.load_mono16k); no torchaudio.
  phonemizer Cnam-LMSSC/wav2vec2-french-phonemizer (HF, ungated), CPU.

What it does:
  1. download N French clips + transcripts;
  2. phonemize the REAL AUDIO;
  3. compare against espeak's G2P of the transcript (our whole stack's
     assumption) -- per-segment agreement, and an espeak-correction tally of
     the substitutions real speech actually makes;
  4. write espeak-corrections.tsv (feeds matcher EQUIV like learned-costs).

Run: python real_audio_g2p.py [--n 40]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

import bench
import matcher

AUDIO_URL = "https://tatoeba.org/audio/download/{aid}"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "tatoeba")


def load_index(n):
    """(sentence_id, text) for French sentences that have audio."""
    aid_of = {}
    for line in open("/tmp/sentences_with_audio.csv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1] not in aid_of:
            aid_of[p[1]] = p[0]
    out = []
    for line in open("/tmp/fra_s.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0] in aid_of and 4 <= len(p[2].split()) <= 10:
            out.append((aid_of[p[0]], p[2]))
        if len(out) >= n * 3:          # headroom for failed downloads
            break
    return out


def fetch(sid):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{sid}.mp3")
    if not os.path.exists(path):
        r = subprocess.run(["curl", "-sL", "--max-time", "25", "-o", path,
                            AUDIO_URL.format(aid=sid)], capture_output=True)
        if r.returncode or os.path.getsize(path) < 2000:
            try:
                os.remove(path)
            except OSError:
                pass
            return None
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForCTC
    import torch
    name = "Cnam-LMSSC/wav2vec2-french-phonemizer"
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForCTC.from_pretrained(name)
    model.eval()
    print("phonemizer loaded (CPU)", file=sys.stderr)

    idx = load_index(args.n)
    agree = []
    sub = defaultdict(Counter)
    done = 0
    for sid, text in idx:
        if done >= args.n:
            break
        path = fetch(sid)
        if not path:
            continue
        try:
            wav = bench.load_mono16k(path)
        except Exception:
            continue
        if len(wav) < 4000:
            continue
        with torch.no_grad():
            logits = model(torch.tensor(wav[None, :])).logits
        ids = logits.argmax(-1)[0]
        real_ipa = proc.batch_decode(ids[None, :])[0].replace(" ", "")
        esp_ipa = matcher._canonical(matcher.g2p(text, "fr"))
        real_ipa = matcher._canonical(real_ipa)
        sa, sb = matcher._segs(esp_ipa), matcher._segs(real_ipa)
        if not sa or not sb:
            continue
        # align espeak vs real-speech phonemes; tally what real speech does
        from difflib import SequenceMatcher
        sm = SequenceMatcher(None, sa, sb)
        agree.append(sm.ratio())
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "replace" and (i2 - i1) == (j2 - j1):
                for a, b in zip(sa[i1:i2], sb[j1:j2]):
                    sub[a][b] += 1
            elif op == "delete":
                for a in sa[i1:i2]:
                    sub[a][""] += 1
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{args.n}  mean espeak~real agreement "
                  f"{np.mean(agree):.2f}", file=sys.stderr)

    print(f"\nREAL French speech vs espeak G2P on {done} Tatoeba clips")
    print(f"mean segment agreement: {np.mean(agree):.2f}  "
          f"(1.0 = espeak is perfect; the gap is what real speech changes)")
    print("\ntop espeak->real substitutions (what French mouths actually do):")
    rows = []
    for a, c in sorted(sub.items(), key=lambda kv: -sum(kv[1].values()))[:14]:
        tot = sum(c.values())
        top = "  ".join(f"{(b or '∅')}×{n}" for b, n in c.most_common(3))
        print(f"  /{a}/ ({tot:3d}) -> {top}")
        for b, n in c.most_common():
            rows.append((a, b, n))
    with open("espeak-corrections.tsv", "w", encoding="utf-8") as f:
        f.write("espeak\treal\tcount\n")
        for a, b, n in rows:
            f.write(f"{a}\t{b}\t{n}\n")
    print("\nwrote espeak-corrections.tsv -- real-speech substitution evidence; "
          "feed the strong ones into matcher EQUIV/CHEAP_GAP exactly like "
          "learned-costs.json (validate on the frozen benchmark first).")


if __name__ == "__main__":
    main()
