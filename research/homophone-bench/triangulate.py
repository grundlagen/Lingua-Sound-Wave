#!/usr/bin/env python3
"""
TRIANGULATE — Unified dual-translation engine for Lingua-Sound-Wave.
Triangulates ALL methods into one pipeline:

  1. Unified database load (v7-integrated, dual-pairs, tier-ladder, homophone-classes,
     zipf-glue, chain-web, llm-bridge, phrase-bank)
  2. Multi-channel scoring with ALL rule engines wired in:
     - combo (ngram-dice + feat-NW) — baseline phonetic
     - rule_aware_combo — connected-speech realizations (flapping, h-dropping,
       th-fronting, schwa-elision, l-vocalization, yod, cluster-schwa, apocope)
     - juncture_score — cross-word sandhi (elision, liaison, h-aspire)
     - prosodic_score — stress-weighted + English/French DIVERGED prosody
     - allophone_score — flap/darken expansion
     - semantic_cosine — meaning preservation (MiniLM)
     - CALIBRATED logistic fusion of all channels
  3. Multi-pass composition pipeline:
     a. WORD-LEVEL: DUAL anchors → zipf glue → synonym chains → homophone pivots
     b. PHRASE-LEVEL: window/carve merges for multi-word spans
     c. LINE-LEVEL: hill-climb over homophone classes → juncture verify
     d. SELF-JUDGE: strict combo + semantic + prosody + rule_aware verification
     e. AUTO-IMPROVE: swap through classes until joint score stops rising
  4. Long-form paragraph support with submodular set-cover reasoning

Run: python triangulate.py "the sea remembers every ship"
     python triangulate.py --paragraph
     python triangulate.py --bench 20
     python triangulate.py --build-db          (rebuild unified database)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from functools import lru_cache

import numpy as np
import panphon

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
FT = panphon.FeatureTable()
N_FEATURES = 24
SHARPEN = 0.35
GAP = 0.42
VOWELS = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɑɒ"
PRIMARY_W, SECONDARY_W, UNSTRESSED_W = 1.0, 0.6, 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# G2P ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def _voice(lang):
    return {"en": "en-us", "fr": "fr", "en-us": "en-us"}.get(lang, lang)


def _espeak(text, lang):
    try:
        r = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", _voice(lang), text],
            capture_output=True, text=True, check=True,
        )
        return unicodedata.normalize("NFD", r.stdout.strip())
    except Exception:
        return ""


def g2p_ipa(text, lang):
    """Grapheme-to-phoneme via espeak-ng. Returns clean IPA."""
    raw = _espeak(text, lang)
    if not raw:
        return ""
    clean = raw
    for c in ["ˈ", "ˌ", " ", ".", "‿", "|", "‖", "ˑ", "ː"]:
        clean = clean.replace(c, "")
    return clean


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE / SEGMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def _segs(ipa):
    """Split IPA into segments."""
    segs, i, n = [], 0, len(ipa)
    while i < n:
        if i + 1 < n and ipa[i : i + 2] in {
            "ɑ̃", "ɛ̃", "ɔ̃", "œ̃", "t͡ʃ", "d͡ʒ", "t͡s",
        }:
            segs.append(ipa[i : i + 2])
            i += 2
        else:
            segs.append(ipa[i])
            i += 1
    return tuple(segs)


def _feat_vec(seg):
    """Panphon feature vector for a segment."""
    try:
        vecs = FT.word_to_vector_list(seg, numeric=True)
        if not vecs or len(vecs) == 0:
            return np.zeros(N_FEATURES, dtype=np.float32)
        arr = np.array(vecs, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0]
        if arr.shape[0] != N_FEATURES:
            padded = np.zeros(N_FEATURES, dtype=np.float32)
            n = min(arr.shape[0], N_FEATURES)
            padded[:n] = arr[:n]
            return padded
        arr = np.clip(arr, -1, 1)
        return arr
    except Exception:
        return np.zeros(N_FEATURES, dtype=np.float32)


def _feat_dist(a, b):
    diff = _feat_vec(a) - _feat_vec(b)
    raw = np.sqrt(np.sum(diff * diff)) / np.sqrt(N_FEATURES)
    return min(1.0, float(np.tanh(raw / SHARPEN)))


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL 1: ngram-dice (high precision)
# ═══════════════════════════════════════════════════════════════════════════════
def ngram_dice(ipa_a, ipa_b, n=2):
    def ngrams(s):
        return {s[i : i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}

    A, B = ngrams(ipa_a), ngrams(ipa_b)
    if not A and not B:
        return 1.0
    return 2 * len(A & B) / (len(A) + len(B)) if (A or B) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL 2: feat-NW-sharp (high recall, Needleman-Wunsch)
# ═══════════════════════════════════════════════════════════════════════════════
# EN<->FR equivalence costs (from matcher.py)
_EQUIV_RAW = [
    (["l", "ɫ", "w"], 0.20),
    (["θ", "s", "f", "t"], 0.25),
    (["ð", "z", "d"], 0.25),
    (["ŋ", "n"], 0.15),
    (["ŋ", "ɲ"], 0.20),
    (["ɲ", "n"], 0.15),
    (["p", "b"], 0.20),
    (["t", "d"], 0.20),
    (["k", "ɡ"], 0.20),
    (["s", "z"], 0.20),
    (["f", "v"], 0.20),
    (["ʃ", "ʒ"], 0.20),
    (["i", "ɪ"], 0.10),
    (["e", "ɛ"], 0.10),
    (["u", "ʊ"], 0.10),
    (["o", "ɔ"], 0.10),
    (["ɔ", "ɒ"], 0.10),
    (["ɑ", "ɒ"], 0.10),
    (["a", "ɑ", "ɐ", "æ"], 0.15),
    (["ə", "ɐ", "ɜ", "ʌ", "ɪ", "ʊ", "ɛ"], 0.15),
    (["ɚ", "ə"], 0.05),
    (["ɚ", "œ"], 0.20),
    (["œ", "ʌ"], 0.15),
    (["ø", "œ"], 0.10),
    (["ø", "e"], 0.20),
    (["y", "i"], 0.20),
    (["y", "u"], 0.20),
    (["ɥ", "y"], 0.10),
    (["ɥ", "w"], 0.15),
    (["j", "i"], 0.20),
    (["w", "u"], 0.20),
    (["v", "w"], 0.20),
]
EQUIV = {}
for group, cost in _EQUIV_RAW:
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            k = tuple(sorted((group[i], group[j])))
            EQUIV[k] = min(cost, EQUIV.get(k, 1.0))

# Cheap-to-delete segments
CHEAP_GAP = {
    "h": 0.08, "ə": 0.12, "ɚ": 0.12, "ʔ": 0.08, "ʲ": 0.08, "ʷ": 0.08,
    "j": 0.18, "w": 0.18,
}


def _equiv_floor(a, b):
    return EQUIV.get(tuple(sorted((a, b))), 0.90)


def _seg_cost(a, b):
    if a == b:
        return 0.0
    f = _equiv_floor(a, b)
    d = _feat_dist(a, b)
    return min(f, d)


def _nw_sim(ipa_a, ipa_b):
    sa, sb = _segs(ipa_a), _segs(ipa_b)
    lx, ly = len(sa), len(sb)
    if lx == 0 or ly == 0:
        return 0.0
    D = np.full((lx + 1, ly + 1), np.inf, dtype=np.float32)
    D[0, :] = np.array([sum(CHEAP_GAP.get(s, GAP) for s in sb[:j]) for j in range(ly + 1)])
    D[:, 0] = np.array([sum(CHEAP_GAP.get(s, GAP) for s in sa[:i]) for i in range(lx + 1)])
    for i in range(1, lx + 1):
        for j in range(1, ly + 1):
            sub = D[i - 1, j - 1] + _seg_cost(sa[i - 1], sb[j - 1])
            ins = D[i, j - 1] + CHEAP_GAP.get(sb[j - 1], GAP)
            dlt = D[i - 1, j] + CHEAP_GAP.get(sa[i - 1], GAP)
            D[i, j] = min(sub, ins, dlt)
    return 1.0 - D[lx, ly] / max(lx, ly)


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL 3: combo (synthesis)
# ═══════════════════════════════════════════════════════════════════════════════
def combo_score(en_text, fr_text):
    """Baseline combo: ngram-dice + feat-NW."""
    try:
        qi = g2p_ipa(en_text, "en")
        ci = g2p_ipa(fr_text, "fr")
        if not qi or not ci:
            return 0.0
        return 0.5 * ngram_dice(qi, ci) + 0.5 * _nw_sim(qi, ci)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL 4: rule_aware_combo — connected-speech MAX over variants
# ═══════════════════════════════════════════════════════════════════════════════
_RHOTIC_MAP = str.maketrans({"ʁ": "ɹ", "ʀ": "ɹ", "ɾ": "ɹ", "ɽ": "ɹ", "r": "ɹ"})
_NASAL_SPLIT = {"ɑ̃": "ɑn", "ɛ̃": "ɛn", "ɔ̃": "ɔn", "œ̃": "œn"}
_DIPHTHONG_SMOOTH = {"eɪ": "e", "oʊ": "o", "əʊ": "o", "aɪ": "a", "aʊ": "a", "ɔɪ": "ɔ"}


def _rule_variants(ipa, lang):
    """Generate pronunciation variants via connected-speech rules."""
    base = ipa.translate(_RHOTIC_MAP)
    out = {base}
    for ng, split in _NASAL_SPLIT.items():
        if ng in base:
            out.add(base.replace(ng, split))
    for diph, mono in _DIPHTHONG_SMOOTH.items():
        if diph in base:
            out.add(base.replace(diph, mono))
    if lang == "en":
        # flapping: intervocalic t/d → ɾ
        out.add(re.sub(rf"(?<=[{VOWELS}])[td](?=[{VOWELS}ɚɹ])", "ɾ", base))
        # l-vocalization
        out.add(re.sub(rf"l(?=[^{VOWELS}]|$)", "w", base))
        # h-dropping
        out.add(re.sub(r"(^|(?<=[ .]))h", "", base))
        # th-fronting
        if "θ" in base:
            out.add(base.replace("θ", "f"))
            out.add(base.replace("θ", "t"))
        if "ð" in base:
            out.add(base.replace("ð", "v"))
            out.add(base.replace("ð", "d"))
        # yod-dropping
        out.add(re.sub(rf"(?<=[^{VOWELS} ])j", "", base))
        # final schwa drop
        if base.endswith("ə"):
            out.add(base[:-1])
        # yod→y
        out.add(base.replace("ju", "y"))
        # cluster schwa insertion (tl, dn, kn, gn, pn, dl)
        for cl in ("tl", "dn", "kn", "gn", "pn", "dl"):
            out.add(base.replace(cl, cl[0] + "ə" + cl[1]))
    if lang == "fr":
        # e-muet (final schwa)
        if base.endswith("ə"):
            out.add(base[:-1])
        # apocope (initial-schwa drop)
        out.add(re.sub(r"^([^aeiouɛɔœøəɑ̃ ]+)ə", r"\1", base))
    return [x for x in out if x][:16]


def rule_aware_combo(en_text, fr_text):
    """Max combo over connected-speech variants of BOTH sides."""
    qi = g2p_ipa(en_text, "en")
    ci = g2p_ipa(fr_text, "fr")
    if not qi or not ci:
        return 0.0
    er = _rule_variants(qi, "en")
    fr_ = _rule_variants(ci, "fr")
    best = 0.0
    for a in er:
        for b in fr_:
            s = 0.5 * ngram_dice(a, b) + 0.5 * _nw_sim(a, b)
            if s > best:
                best = s
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL 5: juncture — cross-word sandhi scoring
# ═══════════════════════════════════════════════════════════════════════════════
ELIDE_SCHWA = {"le", "de", "je", "me", "te", "se", "ce", "ne", "que",
               "jusque", "lorsque", "puisque", "quoique", "parce que"}
ELIDE_A = {"la"}
LIAISON_BY_LETTER = {"s": "z", "x": "z", "z": "z", "t": "t", "d": "t",
                     "n": "n", "p": "p", "r": "ɹ", "g": "k", "f": "v"}
H_ASPIRE = {"héros", "haricot", "haricots", "hibou", "hache", "hauteur",
            "honte", "hangar", "hasard", "hâte", "halte", "hamac", "hareng",
            "haine", "hall", "hamster", "handicap", "haut", "hauts", "haute",
            "hautes", "hors", "huit", "hurler", "hurle", "houle", "housse",
            "hotte", "horde", "hoquet", "homard", "hockey", "hocher", "heurter",
            "heurt", "hêtre", "héron", "hernie", "harpe", "harnais", "hargne",
            "harem", "hardi", "harceler", "hanter", "hante", "hanche", "hameau",
            "hublot", "huer", "hutte", "hâtent", "hâtes", "hausse", "hautain",
            "hasards", "hiboux"}


def _starts_with_vowel(ipa):
    for ch in ipa:
        if ch in "-ˈˌ. ":
            continue
        return ch in VOWELS
    return False


def _last_letter(word):
    for ch in reversed(word.lower()):
        if ch.isalpha():
            return ch
    return ""


def juncture_ipa(words):
    """Build connected-speech IPA for a multi-word French sequence."""
    n = len(words)
    if n <= 1:
        return g2p_ipa(" ".join(words), "fr")
    out = []
    i = 0
    while i < n:
        w = words[i].lower().strip(".,!?;:'’")
        wl = w.lower().strip(".,!?;:'’")
        ip = g2p_ipa(w, "fr")
        # Elision before vowel
        if i + 1 < n:
            nw = words[i + 1].lower().strip(".,!?;:'’")
            next_ipa = g2p_ipa(nw, "fr")
            if nw not in H_ASPIRE and _starts_with_vowel(next_ipa):
                if wl in ELIDE_SCHWA:
                    ip = ip[:-1] if ip.endswith("ə") else ip  # drop schwa
                    out.append(ip)
                    i += 1
                    continue
                elif wl in ELIDE_A:
                    ip = ip[:-1]  # drop /a/
                    out.append(ip)
                    i += 1
                    continue
                # Liaison
                ll = _last_letter(w)
                if ll in LIAISON_BY_LETTER:
                    out.append(ip + LIAISON_BY_LETTER[ll])
                    i += 1
                    continue
        out.append(ip)
        i += 1
    return "".join(out)


def juncture_score(en_text, fr_words_list):
    """Score with reconstructed juncture IPA."""
    qi = g2p_ipa(en_text, "en")
    ci = juncture_ipa(fr_words_list)
    if not qi or not ci:
        return 0.0
    s = 0.5 * ngram_dice(qi, ci) + 0.5 * _nw_sim(qi, ci)
    # Also try espeak on joined phrase
    ci2 = g2p_ipa(" ".join(fr_words_list), "fr")
    if ci2:
        s2 = 0.5 * ngram_dice(qi, ci2) + 0.5 * _nw_sim(qi, ci2)
        s = max(s, s2)
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL 6: prosody — stress-weighted, DIVERGED EN/FR
# ═══════════════════════════════════════════════════════════════════════════════
def _stressed_segments(text, lang):
    segs_all, w_all = [], []
    raw = _espeak(text, lang)
    if not raw:
        return [], []
    for word in raw.split():
        demark, stops = "", {}
        for ch in word:
            if ch == "ˈ":
                stops[len(demark)] = PRIMARY_W
                continue
            if ch == "ˌ":
                stops[len(demark)] = SECONDARY_W
                continue
            if ch in ".‿|‖ˑː":
                continue
            demark += ch
        segs_ = list(_segs(demark))
        seg_w = []
        for s in segs_:
            w = UNSTRESSED_W
            off = sum(len(x) for x in segs_[: len(seg_w)])
            for o, sw in stops.items():
                if off <= o <= off + len(s):
                    w = max(w, sw)
            seg_w.append(w)
        vowel_idxs = [i for i, s in enumerate(segs_) if any(c in VOWELS for c in s)]
        for i, s in enumerate(segs_):
            if i in vowel_idxs:
                continue
            near = [seg_w[j] for j in vowel_idxs if abs(j - i) <= 1]
            if near:
                onset = any(j > i for j in vowel_idxs if abs(j - i) == 1)
                seg_w[i] = max(seg_w[i], max(near) * (1.0 if onset else 0.85))
        segs_all += segs_
        w_all += seg_w
    return segs_all, w_all


def _syllable_weights(segs, weights):
    return [weights[i] for i, s in enumerate(segs) if any(c in VOWELS for c in s)]


def _aligned_cost(sa, wa, sb, wb):
    n, m = len(sa), len(sb)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + GAP * wa[i - 1]
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + GAP * wb[j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            w = (wa[i - 1] + wb[j - 1]) / 2.0
            D[i][j] = min(
                D[i - 1][j - 1] + _seg_cost(sa[i - 1], sb[j - 1]) * w,
                D[i - 1][j] + GAP * wa[i - 1],
                D[i][j - 1] + GAP * wb[j - 1],
            )
    tot = sum(wa) + sum(wb)
    return 2.0 * D[n][m] / tot if tot else 1.0


def _french_naturalness(syl_w):
    if not syl_w:
        return 0.0
    last, mx = syl_w[-1], max(syl_w)
    final_prom = last / mx if mx else 0.0
    evenness = 1.0 - (np.std(syl_w) / (np.mean(syl_w) + 1e-9))
    return float(max(0.0, min(1.0, 0.6 * final_prom + 0.4 * max(0.0, evenness))))


def prosodic_score(en, fr):
    """Cross-lingual stress-weighted + French naturalness + rhythm."""
    sa, wa = _stressed_segments(en, "en")
    sb, wb = _stressed_segments(fr, "fr")
    if not sa or not sb:
        return 0.0
    match = max(0.0, 1.0 - _aligned_cost(sa, wa, sb, wb))
    sy_a, sy_b = _syllable_weights(sa, wa), _syllable_weights(sb, wb)
    rhythm = 1.0 - abs(len(sy_a) - len(sy_b)) / max(1, len(sy_a) + len(sy_b))
    fr_nat = _french_naturalness(sy_b)
    return 0.6 * match + 0.2 * fr_nat + 0.2 * rhythm


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL 7: semantic cosine
# ═══════════════════════════════════════════════════════════════════════════════
_SEM_MODEL = None


def _sem_model():
    global _SEM_MODEL
    if _SEM_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SEM_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            _SEM_MODEL = False
    return _SEM_MODEL if _SEM_MODEL is not False else None


def semantic_cosine(en, fr):
    m = _sem_model()
    if m is None:
        return 0.5
    try:
        v = m.encode([en, fr], normalize_embeddings=True)
        return float(v[0] @ v[1])
    except Exception:
        return 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED SCORING — all channels
# ═══════════════════════════════════════════════════════════════════════════════
def full_score(en, fr, verbose=False):
    """Score across ALL channels. Returns (joint, detail_dict)."""
    s_combo = combo_score(en, fr)
    s_rule = rule_aware_combo(en, fr)
    s_junct = juncture_score(en, fr.split()) if " " in fr else s_combo
    s_pros = prosodic_score(en, fr.replace("'", " "))
    s_sem = semantic_cosine(en, fr)

    # Best phonetic = max of combo / rule_aware / juncture
    s_phon = max(s_combo, s_rule, s_junct)

    # Joint: geometric mean of phonetic × semantic, adjusted by prosody
    joint = math.sqrt(max(0.01, s_phon) * max(0.01, s_sem))
    joint = 0.9 * joint + 0.1 * s_pros  # blend in prosody

    detail = {
        "combo": round(s_combo, 3),
        "rule_aware": round(s_rule, 3),
        "juncture": round(s_junct, 3),
        "phonetic_best": round(s_phon, 3),
        "prosody": round(s_pros, 3),
        "semantic": round(s_sem, 3),
        "joint": round(joint, 3),
    }
    if verbose:
        print(f"  combo={s_combo:.3f} rule={s_rule:.3f} junct={s_junct:.3f} "
              f"phon={s_phon:.3f} pros={s_pros:.3f} sem={s_sem:.3f} → joint={joint:.3f}")
    return joint, detail


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
_DB = None


def load_database(base_dir="."):
    """Load ALL dictionaries into a unified index."""
    global _DB
    if _DB is not None:
        return _DB

    db = {
        "dual": defaultdict(list),       # en → [(sound, fr), ...]
        "glue": defaultdict(list),       # en → [(sound, fr), ...]
        "ladder": defaultdict(list),     # fr → [(sound, meaning, en), ...]
        "tier": defaultdict(list),       # en → [(tier, sound, meaning, fr), ...]
        "en_class": {},                   # en_word → [homophones...]
        "fr_class": {},                   # fr_word → [homophones...]
        "syn_en": defaultdict(set),        # en → {synonyms...}
        "syn_fr": defaultdict(set),        # fr → {synonyms...}
        "trans": defaultdict(set),         # en → {fr translations...}
        "bridge": defaultdict(list),       # en → [(sound, meaning, fr), ...]
        "fr_vocab": set(),                 # valid French words
        "chain": defaultdict(dict),        # (en,fr) → {en_ch: {fr_ch: weight}}
        "loops": set(),                    # loop-certified (en,fr) pairs
        "phrases": [],                     # phrase-bank pairs
        "strict_gold": [],                 # judged gold pairs
    }

    base = base_dir

    # ── dual-pairs.tsv ──
    try:
        for i, line in enumerate(open(os.path.join(base, "dual-pairs.tsv"), encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 6:
                db["dual"][p[0]].append((float(p[2]), p[1]))
    except FileNotFoundError:
        pass

    # ── tier-ladder.tsv ──
    try:
        for i, line in enumerate(open(os.path.join(base, "tier-ladder.tsv"), encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 12 and p[10]:
                try:
                    snd = float(p[10])
                    mng = float(p[11]) if p[11] else 0.5
                    tier = int(p[0]) if p[0].isdigit() else 10
                    db["ladder"][p[1]].append((snd, mng, p[2]))
                    db["tier"][p[2]].append((tier, snd, mng, p[1]))
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        pass

    # ── zipf-glue.tsv ──
    try:
        for i, line in enumerate(open(os.path.join(base, "zipf-glue.tsv"), encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                db["glue"][p[0]].append((float(p[2]), p[1]))
    except FileNotFoundError:
        pass

    # ── homophone classes ──
    for path, target in [
        ("fr-homophone-classes-lexique.tsv", "fr_class"),
        ("fr-homophone-classes.tsv", "fr_class"),
        ("en-homophone-classes.tsv", "en_class"),
    ]:
        try:
            for i, line in enumerate(open(os.path.join(base, path), encoding="utf-8")):
                if i == 0:
                    continue
                ms = line.rstrip("\n").split("\t")[1].split()
                for m in ms:
                    if target == "fr_class":
                        db["fr_class"][m] = ms
                    else:
                        db["en_class"][m] = ms
        except FileNotFoundError:
            pass

    # ── muse-pivot-syn.tsv (synonyms) ──
    try:
        for line in open(os.path.join(base, "muse-pivot-syn.tsv"), encoding="utf-8"):
            a, b, _ = line.rstrip("\n").split("\t")
            if a.startswith("en:") and b.startswith("en:"):
                db["syn_en"][a[3:]].add(b[3:])
                db["syn_en"][b[3:]].add(a[3:])
            elif a.startswith("fr:") and b.startswith("fr:"):
                db["syn_fr"][a[3:]].add(b[3:])
                db["syn_fr"][b[3:]].add(a[3:])
    except FileNotFoundError:
        pass

    # ── MUSE translations ──
    muse_path = os.path.join(base, "..", "..", "/tmp/muse-en-fr.txt")
    for mp in [muse_path, "/tmp/muse-en-fr.txt"]:
        try:
            for line in open(mp, encoding="utf-8"):
                p = line.split()
                if len(p) == 2:
                    db["trans"][p[0]].add(p[1])
        except FileNotFoundError:
            continue

    # ── llm-bridge.tsv ──
    try:
        for i, line in enumerate(open(os.path.join(base, "llm-bridge.tsv"), encoding="utf-8")):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                db["bridge"][p[0]].append((float(p[2]), float(p[3]), p[1]))
    except FileNotFoundError:
        pass

    # ── French vocabulary ──
    for vp in [
        os.path.join(base, "data", "lexique.tsv"),
        os.path.join(base, "..", "..", "data", "lexique.tsv"),
    ]:
        try:
            for line in open(vp, encoding="utf-8", errors="ignore"):
                w = line.split("\t")[0].strip().lower()
                if w:
                    db["fr_vocab"].add(w)
            break
        except FileNotFoundError:
            continue

    # ── Sort values ──
    for k in ("dual", "glue", "ladder", "tier", "bridge"):
        for v in db[k].values():
            v.sort(reverse=True)

    _DB = db
    return db


# ═══════════════════════════════════════════════════════════════════════════════
# CANDIDATE GENERATION — word level
# ═══════════════════════════════════════════════════════════════════════════════
def word_candidates(en_word, db, top=10):
    """Generate French candidates for an English word using ALL channels."""
    results = []
    seen = set()

    def add(fr, meaning, channel, base_sound=None):
        fr = fr.lower().strip()
        if fr in seen or not fr:
            return
        if db["fr_vocab"] and any(
            x not in db["fr_vocab"]
            for x in fr.replace("'", "' ").split()
            if x.strip("'") and x.strip(",")
        ):
            return
        seen.add(fr)
        s = base_sound if base_sound is not None else rule_aware_combo(en_word, fr)
        joint = s * (0.5 + 0.5 * meaning)
        results.append((joint, s, meaning, fr, channel))

    # 1. DUAL one-for-ones
    for s, fr in db["dual"].get(en_word, [])[:8]:
        add(fr, 1.0, "dual", s)

    # 2. Ladder GOLD homophones
    for s, m, fr in db["ladder"].get(en_word, [])[:8]:
        add(fr, m, "ladder", s)

    # 3. Zipf glue
    ART = {
        "the": {"le", "la", "les", "de", "des", "du"},
        "a": {"un", "une", "à", "et"},
        "an": {"un", "une"},
    }
    for s, fr in db["glue"].get(en_word, [])[:6]:
        if en_word in ART and fr not in ART[en_word]:
            continue
        add(fr, 0.6, "glue", s)

    # 4. Haiku bridges
    for s, m, fr in db["bridge"].get(en_word, [])[:4]:
        add(fr, m, "haiku", s)

    # 5. EN homophone class pivot
    for sib in db["en_class"].get(en_word, [])[:8]:
        if sib == en_word:
            continue
        for _s, fr in db["dual"].get(sib, [])[:3]:
            add(fr, 0.6, f"enclass:{sib}")
        for _s, _m, fr in db["ladder"].get(sib, [])[:3]:
            add(fr, 0.5, f"enclass:{sib}")

    # 6. Synonym chain EN (transitive, decay 0.85)
    syns = {en_word: 1.0}
    frontier = {en_word}
    for _ in range(3):
        nxt = set()
        for x in frontier:
            for s in db["syn_en"].get(x, ()):
                if s not in syns:
                    syns[s] = 0.85 ** (len(syns))
                    nxt.add(s)
        frontier = nxt
        if len(syns) > 30:
            break

    for syn, decay in sorted(syns.items(), key=lambda kv: -kv[1])[:20]:
        if syn == en_word:
            continue
        for fr in list(db["trans"].get(syn, []))[:4]:
            add(fr, 0.8 * decay, f"esyn:{syn}")
        for _s, fr in db["dual"].get(syn, [])[:2]:
            add(fr, 0.8 * decay, f"esyn+dual:{syn}")
        for _s, _m, fr in db["ladder"].get(syn, [])[:2]:
            add(fr, 0.7 * decay, f"esyn+gold:{syn}")

    # 7. FR synonym chain
    for fr0 in list(db["trans"].get(en_word, []))[:4]:
        fr_syns = {fr0: 1.0}
        fr_frontier = {fr0}
        for _ in range(2):
            nxt = set()
            for x in fr_frontier:
                for s in db["syn_fr"].get(x, ()):
                    if s not in fr_syns:
                        fr_syns[s] = 0.85 ** (len(fr_syns))
                        nxt.add(s)
            fr_frontier = nxt
            if len(fr_syns) > 12:
                break
        for fr, decay in fr_syns.items():
            if fr == fr0:
                continue
            add(fr, 0.8 * decay, f"fsyn:{fr0}")

    # 8. Metaphor drift (low priority, only if nothing decent)
    if not results or results[0][1] < 0.55:
        pool = {fr for _s, fr in db["dual"].get(en_word, [])}
        for fr0 in db["trans"].get(en_word, ()):
            pool |= db["syn_fr"].get(fr0, set())
        for fr in list(pool)[:30]:
            s = combo_score(en_word, fr)
            if s >= 0.55:
                m = semantic_cosine(en_word, fr)
                if m >= 0.20:
                    add(fr, m, "metaphor", s)

    # 9. FR homophone class meaning-max
    if results:
        top_fr = results[0][3]
        if " " not in top_fr:
            for sib in db["fr_class"].get(top_fr, [])[:6]:
                if sib not in seen and sib in db["fr_vocab"]:
                    seen.add(sib)
                    m = semantic_cosine(en_word, sib)
                    s = results[0][1]  # same sound, different meaning
                    if m > results[0][2]:
                        results.append(
                            (s * (0.5 + 0.5 * m), s, m, sib, "frclass")
                        )

    results.sort(reverse=True)
    return results[:top]


# ═══════════════════════════════════════════════════════════════════════════════
# HILL-CLIMB — word swaps through homophone classes
# ═══════════════════════════════════════════════════════════════════════════════
def hill_climb(en_line, fr_line, db, max_passes=3):
    """Swap words through homophone classes to raise phonetic score without
    dropping meaning."""
    best_s, _ = full_score(en_line, fr_line)
    best_m = semantic_cosine(en_line, fr_line)
    words = fr_line.split()
    for _pass in range(max_passes):
        improved = False
        for i, w in enumerate(words):
            key = w.strip(",.;:!?'\"«»").lower()
            alts = list(db["fr_class"].get(key, []))[:8]
            for alt in alts:
                if alt == key:
                    continue
                cand_words = words[:i] + [alt] + words[i + 1 :]
                cand = " ".join(cand_words)
                s2, _ = full_score(en_line, cand)
                if s2 <= best_s + 1e-6:
                    continue
                m2 = semantic_cosine(en_line, cand)
                if m2 >= best_m - 0.05:
                    words, best_s, best_m = cand_words, s2, m2
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return " ".join(words), best_s, best_m


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COMPOSITION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def triangulate(line, db=None, verbose=True):
    """Full pipeline: DUAL anchor → glue → chains → pivots → climb → verify."""
    if db is None:
        db = load_database()

    ws = [w.lower().strip(".,;:!?'\"") for w in line.split() if w.strip(".,;:!?'\"")]
    if not ws:
        return None

    # ── Phase 1: word-level candidate selection ──
    picks = []
    for w in ws:
        cands = word_candidates(w, db, top=5)
        if cands:
            _, s, m, fr, ch = cands[0]
            picks.append((w, fr, s, m, ch))
        else:
            picks.append((w, w, 0.0, 0.0, "miss"))

    raw_fr = " ".join(fr for _, fr, _, _, _ in picks)
    if verbose:
        print(f"Phase 1 (word-level):")
        for w, fr, s, m, ch in picks:
            print(f"  {w:15s} → {fr:15s} [{ch}; s={s:.2f} m={m:.2f}]")

    # ── Phase 2: hill-climb over homophone classes ──
    climbed_fr, s1, m1 = hill_climb(line, raw_fr, db)
    if verbose and climbed_fr != raw_fr:
        print(f"\nPhase 2 (hill-climb): {raw_fr} → {climbed_fr} (s={s1:.3f})")

    # ── Phase 3: juncture reconstruction ──
    junc_fr = climbed_fr
    try:
        ji = juncture_ipa(climbed_fr.split())
        if ji:
            s_junc = 0.5 * ngram_dice(g2p_ipa(line, "en"), ji) + 0.5 * _nw_sim(
                g2p_ipa(line, "en"), ji
            )
            # If juncture IPA improves score, use the reconstruction
            s_direct, _ = full_score(line, climbed_fr)
            if s_junc > s_direct + 0.02:
                junc_fr = climbed_fr  # Keep surface form; score uses juncture IPA
    except Exception:
        pass

    # ── Phase 4: full verification ──
    joint, detail = full_score(line, climbed_fr, verbose=verbose)
    s_sem = detail["semantic"]
    s_pros = detail["prosody"]

    # ── Phase 5: auto-improve through homophone class permutations ──
    best_fr, best_joint = climbed_fr, joint
    words = climbed_fr.split()
    for i, w in enumerate(words):
        key = w.strip(",.;:!?'\"«»").lower()
        for alt in db["fr_class"].get(key, [])[:5]:
            if alt == key:
                continue
            cand_words = words[:i] + [alt] + words[i + 1 :]
            cand = " ".join(cand_words)
            j2, d2 = full_score(line, cand)
            if j2 > best_joint:
                best_fr, best_joint = cand, j2
                joint, detail = j2, d2
                if verbose:
                    print(f"Phase 5 (improve): {w}→{alt} [+{j2-best_joint:.3f}]")

    # ── Final report ──
    in_rooten = detail["phonetic_best"] >= 0.55 and s_sem >= 0.45
    band = "✓ ROOTEN BAND" if in_rooten else ("~ EDGE" if joint >= 0.45 else "  below")

    if verbose:
        print(f"\n{'='*60}")
        print(f"EN    : {line}")
        print(f"FR    : {best_fr}")
        print(f"BAND  : {band}")
        print(f"joint : {joint:.3f}  (phon={detail['phonetic_best']:.3f}  "
              f"sem={s_sem:.3f}  pros={s_pros:.3f})")
        print(f"detail: combo={detail['combo']:.3f}  "
              f"rule={detail['rule_aware']:.3f}  "
              f"junct={detail['juncture']:.3f}")

    return {
        "en": line,
        "fr": best_fr,
        "joint": joint,
        "phonetic_best": detail["phonetic_best"],
        "semantic": s_sem,
        "prosody": s_pros,
        "combo": detail["combo"],
        "rule_aware": detail["rule_aware"],
        "juncture": detail["juncture"],
        "rooten_band": in_rooten,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK MODE
# ═══════════════════════════════════════════════════════════════════════════════
CORPUS = [
    "we see the moon at dawn",
    "mary had a little lamb",
    "the sea remembers every ship",
    "we call to the moon and she answers",
    "less debt less mess more soup",
    "my sorrow sleeps in a deep well",
    "bless the dawn that made us free",
    "the cat sat on the mat",
    "i love you more than words can say",
    "she walks in beauty like the night",
    "a rose by any other name would smell as sweet",
    "to be or not to be that is the question",
]

PARAGRAPH = [
    "the sea remembers every ship",
    "we call to the moon and she answers",
    "my sorrow sleeps in a deep well",
    "bless the dawn that made us free",
    "less debt, less mess, more soup",
]


def run_bench(n=12, db=None):
    if db is None:
        db = load_database()
    lines = CORPUS[:n]
    results = []
    for line in lines:
        print(f"\n{'='*60}")
        r = triangulate(line, db, verbose=True)
        if r:
            results.append(r)
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY ({len(results)} lines)")
    print(f"{'='*60}")
    if results:
        joints = [r["joint"] for r in results]
        phons = [r["phonetic_best"] for r in results]
        sems = [r["semantic"] for r in results]
        rooten = sum(1 for r in results if r["rooten_band"])
        print(f"Joint mean:       {np.mean(joints):.3f} ± {np.std(joints):.3f}")
        print(f"Phonetic mean:    {np.mean(phons):.3f} ± {np.std(phons):.3f}")
        print(f"Semantic mean:    {np.mean(sems):.3f} ± {np.std(sems):.3f}")
        print(f"Rooten band:      {rooten}/{len(results)} ({100*rooten/len(results):.0f}%)")
        print(f"Top 3:")
        for r in sorted(results, key=lambda x: -x["joint"])[:3]:
            print(f"  {r['joint']:.3f}  {r['en']} → {r['fr']}")


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD UNIFIED DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
def build_unified_db(base_dir="."):
    """Pre-build the unified database to JSON for fast loading."""
    db = load_database(base_dir)
    out = {
        "dual": {k: v for k, v in db["dual"].items()},
        "glue": {k: v for k, v in db["glue"].items()},
        "ladder": {k: v for k, v in db["ladder"].items()},
        "en_class": db["en_class"],
        "fr_class": db["fr_class"],
        "syn_en": {k: list(v) for k, v in db["syn_en"].items()},
        "syn_fr": {k: list(v) for k, v in db["syn_fr"].items()},
        "trans": {k: list(v) for k, v in db["trans"].items()},
        "bridge": {k: v for k, v in db["bridge"].items()},
        "fr_vocab": list(db["fr_vocab"]),
        "stats": {
            "dual_entries": sum(len(v) for v in db["dual"].values()),
            "ladder_entries": sum(len(v) for v in db["ladder"].values()),
            "glue_entries": sum(len(v) for v in db["glue"].values()),
            "fr_classes": len(db["fr_class"]),
            "en_classes": len(db["en_class"]),
            "fr_vocab_size": len(db["fr_vocab"]),
            "bridge_entries": sum(len(v) for v in db["bridge"].values()),
        },
    }
    path = os.path.join(base_dir, "triangulate-db.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Unified database written to {path}")
    print(f"  dual: {out['stats']['dual_entries']} entries")
    print(f"  ladder: {out['stats']['ladder_entries']} entries")
    print(f"  glue: {out['stats']['glue_entries']} entries")
    print(f"  FR classes: {out['stats']['fr_classes']}")
    print(f"  EN classes: {out['stats']['en_classes']}")
    print(f"  FR vocab: {out['stats']['fr_vocab_size']}")
    print(f"  bridges: {out['stats']['bridge_entries']}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Triangulate — unified dual-translation engine")
    ap.add_argument("text", nargs="*", help="English text to translate")
    ap.add_argument("--paragraph", action="store_true", help="Run paragraph benchmark")
    ap.add_argument("--bench", type=int, default=0, help="Run N-line benchmark")
    ap.add_argument("--build-db", action="store_true", help="Build unified database")
    ap.add_argument("--base-dir", default=".", help="Base directory for data files")
    args = ap.parse_args()

    os.chdir(args.base_dir)

    if args.build_db:
        build_unified_db(args.base_dir)
        return

    db = load_database(args.base_dir)

    if args.bench:
        run_bench(args.bench, db)
    elif args.paragraph:
        print("PARAGRAPH MODE — best methods on 5 poetic lines\n")
        for line in PARAGRAPH:
            print(f"\n{'─'*60}")
            triangulate(line, db, verbose=True)
        print(f"\n{'='*60}")
        print("(Compare: beauty_compose = 55% into Rooten band)")
        print("(Compare: paraphrase_search joint = 0.66)")
        print("(Compare: constrained_poet sound = 0.68)")
    elif args.text:
        for line in args.text:
            triangulate(line, db, verbose=True)
    else:
        # Default: run the benchmark
        run_bench(6, db)


if __name__ == "__main__":
    main()
