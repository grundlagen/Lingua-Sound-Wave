"""Offline benchmark: which homophone-matching method actually discriminates?

Everything is deterministic and runs without API keys:
  - G2P + TTS: espeak-ng (deterministic, EN + FR voices)
  - articulatory features: panphon (24 features per IPA segment)
  - audio features: numpy MFCC port of artifacts/api-server/src/lib/dsp.ts

Methods compared (the families the user has tried, plus syntheses):
  ipa-lev          Levenshtein on raw IPA strings (baseline)
  xsampa-lev       same on X-SAMPA (control: encoding doesn't matter)
  ngram-dice       phoneme bigram Dice coefficient ("n-gram" family)
  feat-nw          Needleman-Wunsch, panphon feature substitution costs
  feat-nw-rules    + language-rule variants (schwa drop, diphthong smoothing,
                   nasal vowel splits, rhotic equivalence) — best-of alignment
  feat-dtw         DTW over per-segment feature vectors (continuous alignment,
                   no symbolic gap penalty)
  mfcc-dtw         acoustic: espeak audio -> MFCC -> DTW (cross-language voices)
  mfcc-dtw-xvoice  median over 4 cross-voice pairings (f/m x f/m)
  hybrid-geo       geometric mean of feat-nw-rules x mfcc-dtw-xvoice
  gate             feat-nw-rules ranks; acoustic channel can only demote
  allosaurus-nw    audio -> universal phoneme recognizer -> feat-nw (optional)

Metrics: ROC-AUC over all pairs (threshold-free), mean pos/neg separation,
held-out accuracy at the train-optimal threshold (split by stable hash).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unicodedata
from functools import lru_cache

import numpy as np
import panphon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import all_pairs  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

FT = panphon.FeatureTable()
N_FEATURES = 24

# ---------------------------------------------------------------- G2P / TTS

def espeak(args: list[str]) -> str:
    out = subprocess.run(["espeak-ng", "-q"] + args, capture_output=True, text=True, check=True)
    return out.stdout.strip()


@lru_cache(maxsize=4096)
def g2p_ipa(text: str, lang: str) -> str:
    voice = "en-us" if lang == "en" else "fr"
    raw = espeak(["--ipa", "-v", voice, text])
    return normalize_ipa(raw)


@lru_cache(maxsize=4096)
def g2p_xsampa(text: str, lang: str) -> str:
    voice = "en-us" if lang == "en" else "fr"
    raw = espeak(["-x", "-v", voice, text])
    return "".join(raw.split())


def normalize_ipa(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    # strip stress, syllable and tie marks; keep length + nasalization
    drop = {"ˈ", "ˌ", "‿", ".", "|", "‖", " ", "\n", "\t"}
    return "".join(ch for ch in s if ch not in drop)


def tts_wav(text: str, lang: str, variant: str) -> str:
    """Synthesize and return path to mono 16k f32-decodable wav. Cached."""
    voice = ("en-us" if lang == "en" else "fr") + (f"+{variant}" if variant else "")
    key = hashlib.md5(f"{voice}|{text}".encode()).hexdigest()
    path = os.path.join(CACHE_DIR, f"{key}.wav")
    if not os.path.exists(path):
        subprocess.run(["espeak-ng", "-v", voice, "-w", path, text], check=True, capture_output=True)
    return path


def load_mono16k(path: str) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
         "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


# ------------------------------------------------------ articulatory features

@lru_cache(maxsize=4096)
def segs_and_vecs(ipa: str):
    """IPA string -> (segments, feature matrix [n, 24] in {-1,0,1})."""
    segs = tuple(FT.ipa_segs(ipa))
    if not segs:
        return (), np.zeros((0, N_FEATURES))
    vecs = np.array(FT.word_to_vector_list("".join(segs), numeric=True), dtype=float)
    return segs, vecs


def feat_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized articulatory distance between two segments, [0, 1]."""
    return float(np.abs(a - b).sum()) / (2.0 * N_FEATURES)


# Raw panphon L1 distances are compressed (most features agree between any
# two segments: random cross-segment distance ≈ 0.3, not 1.0). Sharpen so a
# typical unrelated substitution saturates to ~1.0 while near-equivalent
# segments (rhotics, schwa/reduced vowels) stay cheap.
SHARPEN = 0.35


def sharpened(sub: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, sub / SHARPEN)


# ------------------------------------------------------------------ aligners

GAP = 0.42  # cost of inserting/deleting one segment (≈ a bad substitution)


def nw_distance(va: np.ndarray, vb: np.ndarray, sharpen: bool = False) -> float:
    """Needleman-Wunsch; returns average per-step cost along optimal path."""
    n, m = len(va), len(vb)
    if n == 0 or m == 0:
        return 1.0
    sub = np.abs(va[:, None, :] - vb[None, :, :]).sum(axis=2) / (2.0 * N_FEATURES)
    if sharpen:
        sub = sharpened(sub)
    cost = np.zeros((n + 1, m + 1))
    length = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, :] = np.arange(m + 1) * GAP
    cost[:, 0] = np.arange(n + 1) * GAP
    length[0, :] = np.arange(m + 1)
    length[:, 0] = np.arange(n + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                (cost[i - 1, j - 1] + sub[i - 1, j - 1], length[i - 1, j - 1]),
                (cost[i - 1, j] + GAP, length[i - 1, j]),
                (cost[i, j - 1] + GAP, length[i, j - 1]),
            )
            c, l = min(options, key=lambda t: t[0])
            cost[i, j] = c
            length[i, j] = l + 1
    return cost[n, m] / max(1, length[n, m])


def dtw_distance(va: np.ndarray, vb: np.ndarray, metric: str = "l1") -> float:
    """DTW; average per-step distance along optimal path (port of dsp.ts)."""
    n, m = len(va), len(vb)
    if n == 0 or m == 0:
        return 1.0
    if metric == "l1":
        d = np.abs(va[:, None, :] - vb[None, :, :]).sum(axis=2) / (2.0 * N_FEATURES)
    else:  # cosine
        na = np.linalg.norm(va, axis=1, keepdims=True)
        nb = np.linalg.norm(vb, axis=1, keepdims=True)
        eps = 1e-12
        sim = (va @ vb.T) / np.maximum(na @ nb.T, eps)
        d = 1.0 - sim
        both_zero = (na < eps) & (nb.T < eps)
        d[both_zero] = 0.0
    INF = np.inf
    prev_c = np.full(m + 1, INF); prev_c[0] = 0.0
    prev_l = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        curr_c = np.full(m + 1, INF)
        curr_l = np.zeros(m + 1, dtype=int)
        row = d[i - 1]
        for j in range(1, m + 1):
            c_dia, l_dia = prev_c[j - 1], prev_l[j - 1]
            c_del, l_del = prev_c[j], prev_l[j]
            c_ins, l_ins = curr_c[j - 1], curr_l[j - 1]
            best, blen = c_dia, l_dia
            if c_del < best:
                best, blen = c_del, l_del
            if c_ins < best:
                best, blen = c_ins, l_ins
            curr_c[j] = row[j - 1] + best
            curr_l[j] = blen + 1
        prev_c, prev_l = curr_c, curr_l
    total, ln = prev_c[m], prev_l[m]
    if not np.isfinite(total) or ln <= 0:
        return 1.0
    return float(total / ln)


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------- rule variants

NASAL_SPLIT = {"ɑ̃": "ɑn", "ɛ̃": "ɛn", "ɔ̃": "ɔn", "œ̃": "œn"}
DIPHTHONG_SMOOTH = {"eɪ": "e", "oʊ": "o", "əʊ": "o", "aɪ": "a", "aʊ": "a", "ɔɪ": "ɔ"}
RHOTIC_MAP = str.maketrans({"ʁ": "ɹ", "ʀ": "ɹ", "ɾ": "ɹ", "ɽ": "ɹ", "r": "ɹ"})


def canonical(ipa: str) -> str:
    s = ipa.replace("ː", "").translate(RHOTIC_MAP)
    return s


def variants(ipa: str, lang: str) -> list[str]:
    """Bounded set of pronunciation variants encoding elision/reduction rules."""
    base = canonical(ipa)
    out = {base}
    # nasal vowel <-> vowel+n (FR nasals vs EN VN sequences)
    s = base
    for k, v in NASAL_SPLIT.items():
        s = s.replace(unicodedata.normalize("NFD", k), v)
    out.add(s)
    # diphthong smoothing (EN -> FR-like monophthongs)
    s2 = base
    for k, v in DIPHTHONG_SMOOTH.items():
        s2 = s2.replace(k, v)
    out.add(s2)
    out.add(s2 if s == base else "".join(s2))
    # combined
    s3 = s2
    for k, v in NASAL_SPLIT.items():
        s3 = s3.replace(unicodedata.normalize("NFD", k), v)
    out.add(s3)
    # final-schwa drop (FR e muet, EN reduced final)
    more = set()
    for v_ in out:
        if v_.endswith("ə"):
            more.add(v_[:-1])
    out |= more
    return list(out)[:8]


# --------------------------------------------------------------------- MFCC

def mfcc(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """13 MFCC + delta + delta2 (39 dims/frame), CMN. Port of dsp.ts."""
    if len(samples) < 400:
        return np.zeros((0, 39))
    pre = np.empty_like(samples)
    pre[0] = samples[0]
    pre[1:] = samples[1:] - 0.97 * samples[:-1]
    frame_len, hop, nfft, nmels, nmfcc = 400, 160, 1024, 40, 13
    win = np.hamming(frame_len)
    nframes = 1 + (len(pre) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(nframes)[:, None]
    frames = pre[idx] * win
    power = np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2 / nfft
    # mel filterbank
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), nmels + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fbank = np.zeros((nmels, nfft // 2 + 1))
    for m in range(1, nmels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c > l:
            fbank[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fbank[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    mel_e = np.log(power @ fbank.T + 1e-10)
    # DCT-II
    n = np.arange(nmels)
    dct = np.cos(np.pi / nmels * (n[None, :] + 0.5) * np.arange(nmfcc)[:, None])
    cep = mel_e @ dct.T
    cep -= cep.mean(axis=0, keepdims=True)  # CMN
    def delta(x):
        pad = np.pad(x, ((2, 2), (0, 0)), mode="edge")
        return (pad[3:-1] - pad[1:-3] + 2 * (pad[4:] - pad[:-4])) / 10.0
    d1 = delta(cep)
    d2 = delta(d1)
    return np.concatenate([cep, d1, d2], axis=1)


def pool_frames(x: np.ndarray, max_frames: int = 120) -> np.ndarray:
    if len(x) <= max_frames:
        return x
    edges = np.linspace(0, len(x), max_frames + 1).astype(int)
    return np.stack([x[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:]) if b > a])


@lru_cache(maxsize=4096)
def audio_features(text: str, lang: str, variant: str) -> tuple:
    samples = load_mono16k(tts_wav(text, lang, variant))
    return (pool_frames(mfcc(samples)),)


def mfcc_dtw_sim(en: str, fr: str, ven: str, vfr: str) -> float:
    a = audio_features(en, "en", ven)[0]
    b = audio_features(fr, "fr", vfr)[0]
    d = dtw_distance(a, b, metric="cosine")
    return float(np.clip(np.exp(-max(0.0, d - 0.04) * 4.0), 0, 1))


# ------------------------------------------------------------------- methods

def m_ipa_lev(en, fr):
    a, b = g2p_ipa(en, "en"), g2p_ipa(fr, "fr")
    return 1.0 - levenshtein(a, b) / max(len(a), len(b), 1)


def m_xsampa_lev(en, fr):
    a, b = g2p_xsampa(en, "en"), g2p_xsampa(fr, "fr")
    return 1.0 - levenshtein(a, b) / max(len(a), len(b), 1)


def m_ngram_dice(en, fr):
    sa, _ = segs_and_vecs(canonical(g2p_ipa(en, "en")))
    sb, _ = segs_and_vecs(canonical(g2p_ipa(fr, "fr")))
    pad_a = ("#",) + sa + ("#",)
    pad_b = ("#",) + sb + ("#",)
    ga = {pad_a[i] + pad_a[i + 1] for i in range(len(pad_a) - 1)}
    gb = {pad_b[i] + pad_b[i + 1] for i in range(len(pad_b) - 1)}
    if not ga or not gb:
        return 0.0
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def _nw_sim(ipa_a: str, ipa_b: str, sharpen: bool = False) -> float:
    _, va = segs_and_vecs(ipa_a)
    _, vb = segs_and_vecs(ipa_b)
    return 1.0 - nw_distance(va, vb, sharpen=sharpen)


def m_feat_nw(en, fr):
    return _nw_sim(g2p_ipa(en, "en"), g2p_ipa(fr, "fr"))


def m_feat_nw_rules(en, fr):
    va = variants(g2p_ipa(en, "en"), "en")
    vb = variants(g2p_ipa(fr, "fr"), "fr")
    return max(_nw_sim(a, b) for a in va for b in vb)


def m_feat_nw_sharp(en, fr):
    """feat-nw-rules with sharpened substitution costs (calibrated range)."""
    va = variants(g2p_ipa(en, "en"), "en")
    vb = variants(g2p_ipa(fr, "fr"), "fr")
    return max(_nw_sim(a, b, sharpen=True) for a in va for b in vb)


# Coarse articulatory classes for order-aware class n-grams: robust to small
# segment differences across languages while keeping sequence structure.
def _seg_class(seg: str) -> str:
    vec = FT.word_to_vector_list(seg, numeric=True)
    if not vec:
        return "?"
    v = vec[0]
    names = FT.names
    f = dict(zip(names, v))
    if f.get("syl", 0) == 1:  # vowel: height x backness x rounding (coarse)
        hi = "H" if f.get("hi", 0) == 1 else ("L" if f.get("lo", 0) == 1 else "M")
        bk = "B" if f.get("back", 0) == 1 else "F"
        rd = "R" if f.get("round", 0) == 1 else "U"
        return f"V{hi}{bk}{rd}"
    son = f.get("son", 0)
    cont = f.get("cont", 0)
    voi = "v" if f.get("voi", 0) == 1 else "u"
    if f.get("nas", 0) == 1:
        manner = "N"
    elif son == 1:
        manner = "A"  # approximant/liquid
    elif cont == 1:
        manner = "F"
    else:
        manner = "P"
    if f.get("lab", 0) == 1:
        place = "lab"
    elif f.get("cor", 0) == 1:
        place = "cor"
    else:
        place = "dor"
    return f"C{manner}{place}{voi}"


@lru_cache(maxsize=4096)
def class_string(ipa: str) -> tuple:
    segs, _ = segs_and_vecs(ipa)
    return tuple(_seg_class(s) for s in segs)


def m_class_ngram(en, fr):
    """Bigram Dice over coarse articulatory classes (order-aware, fuzzy)."""
    ca = ("#",) + class_string(canonical(g2p_ipa(en, "en"))) + ("#",)
    cb = ("#",) + class_string(canonical(g2p_ipa(fr, "fr"))) + ("#",)
    ga = {(ca[i], ca[i + 1]) for i in range(len(ca) - 1)}
    gb = {(cb[i], cb[i + 1]) for i in range(len(cb) - 1)}
    if not ga or not gb:
        return 0.0
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def m_combo(en, fr):
    """Synthesis: exact-segment bigram overlap x sharpened featural alignment.

    ngram-dice supplies hard, order-aware exact-match evidence (high
    precision); sharpened feat-nw-rules supplies gradient alignment that
    tolerates near-equivalent segments (high recall on loose matches).
    Arithmetic mean: either channel can carry a pair it is confident about.
    """
    return 0.5 * m_ngram_dice(en, fr) + 0.5 * m_feat_nw_sharp(en, fr)


def m_combo_geo(en, fr):
    """Conservative variant: both channels must agree (geometric mean)."""
    return float(np.sqrt(max(0.0, m_ngram_dice(en, fr)) * max(0.0, m_feat_nw_sharp(en, fr))))


def m_combo2(en, fr):
    """class-ngram (fuzzy order-aware) + sharpened featural NW."""
    return 0.5 * m_class_ngram(en, fr) + 0.5 * m_feat_nw_sharp(en, fr)


def m_feat_dtw(en, fr):
    _, va = segs_and_vecs(canonical(g2p_ipa(en, "en")))
    _, vb = segs_and_vecs(canonical(g2p_ipa(fr, "fr")))
    return 1.0 - dtw_distance(va, vb, metric="l1")


def m_mfcc_dtw(en, fr):
    return mfcc_dtw_sim(en, fr, "", "")


def m_mfcc_dtw_xvoice(en, fr):
    sims = [mfcc_dtw_sim(en, fr, a, b)
            for a in ("f3", "m3") for b in ("f4", "m4")]
    return float(np.median(sims))


def m_hybrid_geo(en, fr):
    return float(np.sqrt(max(0, m_feat_nw_rules(en, fr)) * max(0, m_mfcc_dtw_xvoice(en, fr))))


VETO = 0.30  # acoustic floor below which the symbolic score gets demoted


def m_gate(en, fr):
    s = m_feat_nw_rules(en, fr)
    a = m_mfcc_dtw_xvoice(en, fr)
    if a < VETO:
        s *= a / VETO
    return s


METHODS = {
    "ipa-lev": m_ipa_lev,
    "xsampa-lev": m_xsampa_lev,
    "ngram-dice": m_ngram_dice,
    "feat-nw": m_feat_nw,
    "feat-nw-rules": m_feat_nw_rules,
    "feat-nw-sharp": m_feat_nw_sharp,
    "class-ngram": m_class_ngram,
    "combo": m_combo,
    "combo-geo": m_combo_geo,
    "combo2": m_combo2,
    "feat-dtw": m_feat_dtw,
    "mfcc-dtw": m_mfcc_dtw,
    "mfcc-dtw-xvoice": m_mfcc_dtw_xvoice,
    "hybrid-geo": m_hybrid_geo,
    "gate": m_gate,
}

# Optional: allosaurus (audio -> universal phoneme recognizer -> featural NW)
try:
    from allosaurus.app import read_recognizer
    _ALLO = read_recognizer()

    @lru_cache(maxsize=4096)
    def allo_ipa(text: str, lang: str) -> str:
        wav = tts_wav(text, lang, "")
        out = _ALLO.recognize(wav)
        return normalize_ipa("".join(out.split()))

    def m_allosaurus_nw(en, fr):
        return _nw_sim(allo_ipa(en, "en"), allo_ipa(fr, "fr"))

    METHODS["allosaurus-nw"] = m_allosaurus_nw
except Exception as e:  # noqa: BLE001
    print(f"[allosaurus unavailable: {type(e).__name__}: {e}]", file=sys.stderr)


# ------------------------------------------------------------------- metrics

def auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Exact ROC-AUC via Mann-Whitney U."""
    wins = ties = 0
    for p in scores_pos:
        for n in scores_neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    total = len(scores_pos) * len(scores_neg)
    return (wins + 0.5 * ties) / total if total else 0.0


def is_test(en: str) -> bool:
    return int(hashlib.md5(en.encode()).hexdigest(), 16) % 2 == 0


def main():
    only = sys.argv[1:] or None
    pairs = all_pairs()
    results = {}
    raw = {}
    for name, fn in METHODS.items():
        if only and name not in only:
            continue
        scores = []
        for en, fr, label, tier in pairs:
            try:
                s = float(fn(en, fr))
            except Exception as e:  # noqa: BLE001
                print(f"  ! {name} failed on ({en!r}, {fr!r}): {e}", file=sys.stderr)
                s = 0.0
            scores.append((en, fr, label, tier, s))
        raw[name] = scores
        pos = [s for *_x, l, _t, s in [(e, f, l, t, s) for e, f, l, t, s in scores] if l == 1]
        neg = [s for e, f, l, t, s in scores if l == 0]
        neg_tr = [s for e, f, l, t, s in scores if l == 0 and t == "translation"]
        # threshold fit on train, accuracy on test
        train = [(l, s) for e, f, l, t, s in scores if not is_test(e)]
        test = [(l, s) for e, f, l, t, s in scores if is_test(e)]
        best_thr, best_acc = 0.5, 0.0
        for thr in sorted({s for _, s in train}):
            acc = sum((s >= thr) == bool(l) for l, s in train) / len(train)
            if acc > best_acc:
                best_acc, best_thr = acc, thr
        test_acc = sum((s >= best_thr) == bool(l) for l, s in test) / len(test)
        pos_loose = [s for e, f, l, t, s in scores if l == 1 and t == "loose"]
        # AUC restricted to the hard sub-problem: loose positives vs translations
        auc_loose = auc(pos_loose, neg_tr) if pos_loose else 0.0
        results[name] = {
            "auc": auc(pos, neg),
            "auc_hard": auc(pos, neg_tr),
            "auc_loose": auc_loose,
            "mean_pos": float(np.mean(pos)),
            "mean_neg": float(np.mean(neg)),
            "sep": float(np.mean(pos) - np.mean(neg)),
            "thr": best_thr,
            "test_acc": test_acc,
        }
        r = results[name]
        print(f"{name:18s} AUC {r['auc']:.3f}  hard {r['auc_hard']:.3f}  "
              f"loose {r['auc_loose']:.3f}  pos {r['mean_pos']:.3f}  neg {r['mean_neg']:.3f}  "
              f"sep {r['sep']:+.3f}  test-acc {r['test_acc']:.3f}")
    with open(os.path.join(CACHE_DIR, "results.json"), "w") as f:
        json.dump({"summary": results,
                   "raw": {k: [list(r) for r in v] for k, v in raw.items()}}, f, indent=1)
    print(f"\nwrote {os.path.join(CACHE_DIR, 'results.json')}")


if __name__ == "__main__":
    main()

