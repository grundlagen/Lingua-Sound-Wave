"""Reference implementation of the benchmark-winning homophone matcher.

Method: `combo` — the synthesis that won the bench.py comparison
(AUC 0.993, hard-negative AUC 0.992, loose-positive AUC 0.997 on the
105-pair EN<->FR set). It is the unweighted mean of two symbolic channels:

  1. ngram-dice    — Dice coefficient over exact phoneme bigrams. High
                     precision: hard, order-aware evidence that two phrases
                     share consonant/vowel transitions.
  2. feat-nw-sharp — Needleman-Wunsch alignment with *sharpened* panphon
                     articulatory-feature substitution costs and rule-based
                     pronunciation variants (schwa drop, diphthong
                     smoothing, nasal-vowel splits, rhotic equivalence).
                     High recall: tolerates near-equivalent segments that
                     differ across languages.

Their errors are independent — n-grams miss loose matches that survive a
segment swap; featural alignment over-credits unrelated phrases that happen
to share broad articulatory shape. Averaging keeps n-gram precision and
feat recall, beating every single method and the acoustic/neural ones by a
wide margin (see RESULTS.md).

Deterministic, no API keys, no neural weights. Needs espeak-ng on PATH and
panphon installed.
"""
from __future__ import annotations

import subprocess
import unicodedata
from functools import lru_cache

import numpy as np
import panphon

FT = panphon.FeatureTable()
N_FEATURES = 24
SHARPEN = 0.35   # featural distance at which a substitution saturates to 1.0
GAP = 0.42       # NW insertion/deletion cost

RHOTIC_MAP = str.maketrans({"ʁ": "ɹ", "ʀ": "ɹ", "ɾ": "ɹ", "ɽ": "ɹ", "r": "ɹ"})
NASAL_SPLIT = {"ɑ̃": "ɑn", "ɛ̃": "ɛn", "ɔ̃": "ɔn", "œ̃": "œn"}
DIPHTHONG_SMOOTH = {"eɪ": "e", "oʊ": "o", "əʊ": "o", "aɪ": "a", "aʊ": "a", "ɔɪ": "ɔ"}


def _normalize_ipa(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    drop = {"ˈ", "ˌ", "‿", ".", "|", "‖", " ", "\n", "\t"}
    return "".join(ch for ch in s if ch not in drop)


@lru_cache(maxsize=8192)
def g2p(text: str, lang: str) -> str:
    """Grapheme-to-IPA via espeak-ng. lang in {'en','fr',...espeak voice}."""
    voice = {"en": "en-us", "fr": "fr"}.get(lang, lang)
    out = subprocess.run(
        ["espeak-ng", "-q", "--ipa", "-v", voice, text],
        capture_output=True, text=True, check=True)
    return _normalize_ipa(out.stdout.strip())


def _canonical(ipa: str) -> str:
    return ipa.replace("ː", "").translate(RHOTIC_MAP)


def _variants(ipa: str) -> list[str]:
    base = _canonical(ipa)
    out = {base}
    s = base
    for k, v in NASAL_SPLIT.items():
        s = s.replace(unicodedata.normalize("NFD", k), v)
    out.add(s)
    s2 = base
    for k, v in DIPHTHONG_SMOOTH.items():
        s2 = s2.replace(k, v)
    out.add(s2)
    s3 = s2
    for k, v in NASAL_SPLIT.items():
        s3 = s3.replace(unicodedata.normalize("NFD", k), v)
    out.add(s3)
    out |= {v[:-1] for v in list(out) if v.endswith("ə")}
    return list(out)[:8]


@lru_cache(maxsize=8192)
def _vecs(ipa: str) -> np.ndarray:
    segs = FT.ipa_segs(ipa)
    if not segs:
        return np.zeros((0, N_FEATURES))
    return np.array(FT.word_to_vector_list("".join(segs), numeric=True), dtype=float)


@lru_cache(maxsize=8192)
def _segs(ipa: str) -> tuple:
    return tuple(FT.ipa_segs(ipa))


def _nw_sim(va: np.ndarray, vb: np.ndarray) -> float:
    n, m = len(va), len(vb)
    if n == 0 or m == 0:
        return 0.0
    sub = np.minimum(1.0, (np.abs(va[:, None, :] - vb[None, :, :]).sum(axis=2) / (2.0 * N_FEATURES)) / SHARPEN)
    cost = np.zeros((n + 1, m + 1))
    length = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, :] = np.arange(m + 1) * GAP
    cost[:, 0] = np.arange(n + 1) * GAP
    length[0, :] = np.arange(m + 1)
    length[:, 0] = np.arange(n + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = (
                (cost[i - 1, j - 1] + sub[i - 1, j - 1], length[i - 1, j - 1]),
                (cost[i - 1, j] + GAP, length[i - 1, j]),
                (cost[i, j - 1] + GAP, length[i, j - 1]),
            )
            c, l = min(opts, key=lambda t: t[0])
            cost[i, j] = c
            length[i, j] = l + 1
    return 1.0 - cost[n, m] / max(1, length[n, m])


def _feat_channel(ipa_a: str, ipa_b: str) -> float:
    va = _variants(ipa_a)
    vb = _variants(ipa_b)
    return max(_nw_sim(_vecs(a), _vecs(b)) for a in va for b in vb)


def _ngram_channel(ipa_a: str, ipa_b: str) -> float:
    sa = ("#",) + _segs(_canonical(ipa_a)) + ("#",)
    sb = ("#",) + _segs(_canonical(ipa_b)) + ("#",)
    ga = {sa[i] + sa[i + 1] for i in range(len(sa) - 1)}
    gb = {sb[i] + sb[i + 1] for i in range(len(sb) - 1)}
    if not ga or not gb:
        return 0.0
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def homophone_score(text_a: str, lang_a: str, text_b: str, lang_b: str) -> dict:
    """Score how alike two phrases sound across languages, in [0, 1].

    Returns the headline score plus both channel sub-scores and the IPA, so
    callers can show their work (and so disagreement is diagnosable).
    A practical decision threshold from the benchmark is ~0.45.
    """
    ipa_a = g2p(text_a, lang_a)
    ipa_b = g2p(text_b, lang_b)
    ngram = _ngram_channel(ipa_a, ipa_b)
    feat = _feat_channel(ipa_a, ipa_b)
    return {
        "score": 0.5 * ngram + 0.5 * feat,
        "ngram_dice": ngram,
        "featural_nw": feat,
        "ipa_a": ipa_a,
        "ipa_b": ipa_b,
    }


if __name__ == "__main__":
    for a, la, b, lb in [
        ("shoe", "en", "chou", "fr"),
        ("mayday", "en", "m'aider", "fr"),
        ("dog", "en", "chien", "fr"),
        ("over the moon", "en", "aux anges", "fr"),
    ]:
        r = homophone_score(a, la, b, lb)
        print(f"{a!r:18s} ~ {b!r:14s}  {r['score']:.3f}  "
              f"(ngram {r['ngram_dice']:.2f} / feat {r['featural_nw']:.2f})  "
              f"[{r['ipa_a']} | {r['ipa_b']}]")
