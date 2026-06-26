"""Method tournament: compete the sound-matching methods on the labelled
benchmark (homophone pairs vs non-homophones) and rank them by AUC + separation,
so we can integrate the winner. Includes:
  - combo               (current matcher default)
  - ngram_dice          (exact-bigram channel)
  - feat_nw_sharp       (sharpened featural alignment)
  - prosody             (stress-weighted, EN/FR-diverged -- this session)
  - drive_equiv_seqmatch (the user's Drive phoneme_mapping_reference approach:
                          normalise both IPA strings by phoneme-equivalence class,
                          then SequenceMatcher ratio)

Run: python tournament.py
"""
from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np

import bench
import matcher
import prosody
from dataset import all_pairs

import drive_phoneme_map as dpm

# reverse the user's equivalence map: each phoneme -> a canonical class key
_CANON = {}
for k, vs in dpm.phoneme_mapping.items():
    for v in vs:
        _CANON.setdefault(v, k)


def _norm(ipa: str) -> str:
    return "".join(_CANON.get(s, s) for s in matcher._segs(ipa))


def drive_equiv_seqmatch(en: str, fr: str) -> float:
    try:
        ei = bench.g2p_ipa(en, "en")
        fi = bench.g2p_ipa(fr, "fr")
        return SequenceMatcher(None, _norm(ei), _norm(fi)).ratio()
    except Exception:
        return 0.0


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return 0.0
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    methods = {
        "combo": bench.m_combo,
        "ngram_dice": bench.m_ngram_dice,
        "feat_nw_sharp": bench.m_feat_nw_sharp,
        "prosody": prosody.prosodic_score,
        "drive_equiv_seqmatch": drive_equiv_seqmatch,
    }
    pairs = all_pairs()
    print(f"tournament on {len(pairs)} labelled pairs\n")
    print(f"{'method':22s} {'AUC':>6s} {'pos':>6s} {'neg':>6s} {'sep':>6s}")
    print("-" * 50)
    res = []
    for name, fn in methods.items():
        pos, neg = [], []
        for en, fr, label, tier in pairs:
            try:
                s = float(fn(en, fr))
            except Exception:
                s = 0.0
            (pos if label else neg).append(s)
        a = auc(pos, neg)
        mp, mn = float(np.mean(pos)), float(np.mean(neg))
        res.append((a, mp - mn, name))
        print(f"{name:22s} {a:6.3f} {mp:6.2f} {mn:6.2f} {mp-mn:+6.2f}")
    res.sort(reverse=True)
    print(f"\nWINNER (by AUC): {res[0][2]}  AUC {res[0][0]:.3f}")
    print("Integration: keep the top method as the sound channel; a calibrated "
          "blend of the top two often beats either (combo's exact-match precision "
          "+ prosody's perceptual recall). Promote by a deliberate edit; judge "
          "unchanged until then.")


if __name__ == "__main__":
    main()
