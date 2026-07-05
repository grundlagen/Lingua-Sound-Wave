#!/usr/bin/env python3
"""
UNIVERSAL HOMOPHONE WRITER — v2 orbit-level rewrite.
Triangulates ALL methods, generates from rules, self-improves.

PIPELINE:
  0. DECOMPOSE: morphological decomposition (remembers→remember)
  1. GENERATE (11 channels, priority-ordered):
     1. COGNATE: identical EN/FR forms — FREE (sound=1.0, meaning=1.0)
     2. DUAL: 102k translation∧homophone pairs
     3. LADDER: GOLD homophone one-for-ones
     4. GLUE: function-word phoneme-rule matches (573 zipf)
     5. BABEL many→one: 2-3 EN words → 1 FR unit (84k units, 20k words)
     6. BABEL one→many: 1 EN word → 2 FR words (carve)
     7. EN HOMOPHONE CLASS pivot → best translation
     8. SYNONYM CHAIN EN → translations → best sound
     9. SYNONYM CHAIN FR → translation → synonyms → best sound
     10. METAPHOR drift (sound≥0.55, cos≥0.20)
     11. CARVE fallback: sound-first decomposition
  2. SCORE: 7-channel (combo + rule_aware + juncture + prosody + semantic)
  3. CLIMB: swap through 33,660 FR homophone classes
  4. IMPROVE: if joint < threshold, try alternative path permutations
  5. RECORD: write back new discoveries

Run: python homophone_writer.py "the sea remembers every ship"
     python homophone_writer.py --bench 12
     python homophone_writer.py --gen 100      (generate from rules, self-improve)
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
from collections import Counter, defaultdict

import numpy as np
import panphon

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════
FT = panphon.FeatureTable()
N_FEATURES = 24
SHARPEN = 0.35
GAP = 0.42
VOWELS = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɑɒ"
PRI, SEC, UNST = 1.0, 0.6, 0.3

# EN↔FR equivalence costs (from matcher.py, rule_aware.py)
_EQUIV_RAW = [
    (["l", "ɫ", "w"], 0.20), (["θ", "s", "f", "t"], 0.25),
    (["ð", "z", "d"], 0.25), (["ŋ", "n"], 0.15), (["ŋ", "ɲ"], 0.20),
    (["ɲ", "n"], 0.15), (["p", "b"], 0.20), (["t", "d"], 0.20),
    (["k", "ɡ"], 0.20), (["s", "z"], 0.20), (["f", "v"], 0.20),
    (["ʃ", "ʒ"], 0.20), (["i", "ɪ"], 0.10), (["e", "ɛ"], 0.10),
    (["u", "ʊ"], 0.10), (["o", "ɔ"], 0.10), (["ɔ", "ɒ"], 0.10),
    (["ɑ", "ɒ"], 0.10), (["a", "ɑ", "ɐ", "æ"], 0.15),
    (["ə", "ɐ", "ɜ", "ʌ", "ɪ", "ʊ", "ɛ"], 0.15),
    (["ɚ", "ə"], 0.05), (["ɚ", "œ"], 0.20), (["œ", "ʌ"], 0.15),
    (["ø", "œ"], 0.10), (["ø", "e"], 0.20), (["y", "i"], 0.20),
    (["y", "u"], 0.20), (["ɥ", "y"], 0.10), (["ɥ", "w"], 0.15),
    (["j", "i"], 0.20), (["w", "u"], 0.20), (["v", "w"], 0.20),
]
EQUIV = {}
for group, cost in _EQUIV_RAW:
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            k = tuple(sorted((group[i], group[j])))
            EQUIV[k] = min(cost, EQUIV.get(k, 1.0))

CHEAP_GAP = {"h": 0.08, "ə": 0.12, "ɚ": 0.12, "ʔ": 0.08, "ʲ": 0.08, "ʷ": 0.08,
             "j": 0.18, "w": 0.18}

_RHOTIC = str.maketrans({"ʁ": "ɹ", "ʀ": "ɹ", "ɾ": "ɹ", "ɽ": "ɹ", "r": "ɹ"})
_NASAL_SPLIT = {"ɑ̃": "ɑn", "ɛ̃": "ɛn", "ɔ̃": "ɔn", "œ̃": "œn"}
_DIPH_SMOOTH = {"eɪ": "e", "oʊ": "o", "əʊ": "o", "aɪ": "a", "aʊ": "a", "ɔɪ": "ɔ"}

ELIDE_SCHWA = {"le", "de", "je", "me", "te", "se", "ce", "ne", "que",
               "jusque", "lorsque", "puisque", "quoique", "parce que"}
ELIDE_A = {"la"}
LIAISON_LETTER = {"s": "z", "x": "z", "z": "z", "t": "t", "d": "t",
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

# Common EN suffixes to strip for morphological decomposition
_EN_SUFFIXES = [
    ("ing", ""), ("ingly", ""), ("ings", ""), ("ment", ""), ("ments", ""),
    ("ness", ""), ("tion", "t"), ("sions", "sion"), ("tions", "tion"),
    ("able", ""), ("ible", ""), ("ally", "al"), ("izes", "ize"),
    ("ised", "ise"), ("ising", "ise"), ("ized", "ize"), ("izing", "ize"),
    ("ers", "er"), ("est", ""), ("er", ""), ("s", ""), ("ed", ""),
    ("'s", ""), ("'d", ""), ("'ll", ""), ("'re", ""), ("'ve", ""),
    ("n't", ""),
]
_EN_MANUAL = {
    "remembers": "remember", "remembered": "remember",
    "ships": "ship", "answers": "answer", "answered": "answer",
    "sleeps": "sleep", "slept": "sleep", "sleeping": "sleep",
    "calls": "call", "calling": "call", "called": "call",
    "blesses": "bless", "blessed": "bless", "blessing": "bless",
    "made": "make", "makes": "make", "making": "make",
    "walks": "walk", "walking": "walk", "walked": "walk",
    "smells": "smell", "smelling": "smell", "smelled": "smell",
    "says": "say", "saying": "say", "said": "say",
    "goes": "go", "going": "go", "went": "go", "gone": "go",
    "has": "have", "had": "have", "having": "have",
    "does": "do", "doing": "do", "done": "do", "did": "do",
    "sees": "see", "seeing": "see", "seen": "see", "saw": "see",
}


# ═══════════════════════════════════════════════════════════════════
# G2P & SEGMENTS
# ═══════════════════════════════════════════════════════════════════
_ESPEAK_CACHE = {}
def _espeak(text, lang):
    key = (text, lang)
    if key in _ESPEAK_CACHE: return _ESPEAK_CACHE[key]
    try:
        r = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v",
             {"en": "en-us", "fr": "fr"}.get(lang, lang), text],
            capture_output=True, text=True, check=True)
        val = unicodedata.normalize("NFD", r.stdout.strip())
    except Exception:
        val = ""
    _ESPEAK_CACHE[key] = val
    return val


def g2p_clean(text, lang):
    raw = _espeak(text, lang)
    if not raw:
        return ""
    for c in "ˈˌ .‿|‖ˑː":
        raw = raw.replace(c, "")
    return raw


def _segs(ipa):
    segs, i, n = [], 0, len(ipa)
    while i < n:
        if i + 1 < n and ipa[i:i + 2] in {"ɑ̃", "ɛ̃", "ɔ̃", "œ̃", "t͡ʃ", "d͡ʒ", "t͡s"}:
            segs.append(ipa[i:i + 2]); i += 2
        else:
            segs.append(ipa[i]); i += 1
    return tuple(segs)


_FEAT_CACHE = {}
def _feat_vec(seg):
    if seg in _FEAT_CACHE: return _FEAT_CACHE[seg]
    try:
        vecs = FT.word_to_vector_list(seg, numeric=True)
        if not vecs:
            arr = np.zeros(N_FEATURES, dtype=np.float32)
        else:
            arr = np.clip(np.array(vecs, dtype=np.float32), -1, 1)
            if arr.ndim == 2: arr = arr[0]
            if arr.shape[0] != N_FEATURES:
                p = np.zeros(N_FEATURES, dtype=np.float32)
                p[:min(arr.shape[0], N_FEATURES)] = arr[:min(arr.shape[0], N_FEATURES)]
                arr = p
    except Exception:
        arr = np.zeros(N_FEATURES, dtype=np.float32)
    _FEAT_CACHE[seg] = arr
    return arr


def _equiv_cost(a, b):
    if a == b:
        return 0.0
    f = EQUIV.get(tuple(sorted((a, b))), 0.90)
    d = min(1.0, float(np.tanh(
        np.sqrt(np.sum((_feat_vec(a) - _feat_vec(b)) ** 2)) / np.sqrt(N_FEATURES) / SHARPEN
    )))
    return min(f, d)


# ═══════════════════════════════════════════════════════════════════
# PHONETIC CHANNELS
# ═══════════════════════════════════════════════════════════════════
def ngram_dice(ia, ib, n=2):
    def ng(s):
        return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}
    A, B = ng(ia), ng(ib)
    return 2 * len(A & B) / (len(A) + len(B)) if (A or B) else 1.0


def feat_nw(ia, ib):
    sa, sb = _segs(ia), _segs(ib)
    lx, ly = len(sa), len(sb)
    if lx == 0 or ly == 0:
        return 0.0
    D = np.full((lx + 1, ly + 1), np.inf, dtype=np.float32)
    D[0, :] = [sum(CHEAP_GAP.get(s, GAP) for s in sb[:j]) for j in range(ly + 1)]
    D[:, 0] = [sum(CHEAP_GAP.get(s, GAP) for s in sa[:i]) for i in range(lx + 1)]
    for i in range(1, lx + 1):
        for j in range(1, ly + 1):
            D[i, j] = min(
                D[i - 1, j - 1] + _equiv_cost(sa[i - 1], sb[j - 1]),
                D[i, j - 1] + CHEAP_GAP.get(sb[j - 1], GAP),
                D[i - 1, j] + CHEAP_GAP.get(sa[i - 1], GAP),
            )
    return max(0.0, 1.0 - D[lx, ly] / max(lx, ly))


def combo(en_text, fr_text):
    qi, ci = g2p_clean(en_text, "en"), g2p_clean(fr_text, "fr")
    if not qi or not ci:
        return 0.0
    return 0.5 * ngram_dice(qi, ci) + 0.5 * feat_nw(qi, ci)


# ── rule_aware: connected-speech variant expansion ──
def _rule_variants(ipa, lang):
    base = ipa.translate(_RHOTIC)
    out = {base}
    for ng, sp in _NASAL_SPLIT.items():
        if ng in base: out.add(base.replace(ng, sp))
    for dp, mn in _DIPH_SMOOTH.items():
        if dp in base: out.add(base.replace(dp, mn))
    if lang == "en":
        out.add(re.sub(rf"(?<=[{VOWELS}])[td](?=[{VOWELS}ɚɹ])", "ɾ", base))
        out.add(re.sub(rf"l(?=[^{VOWELS}]|$)", "w", base))
        out.add(re.sub(r"(^|(?<=[ .]))h", "", base))
        for th, sub in (("θ", "f"), ("θ", "t"), ("ð", "v"), ("ð", "d")):
            if th in base: out.add(base.replace(th, sub))
        out.add(re.sub(rf"(?<=[^{VOWELS} ])j", "", base))
        if base.endswith("ə"): out.add(base[:-1])
        out.add(base.replace("ju", "y"))
        for cl in ("tl", "dn", "kn", "gn", "pn", "dl"):
            out.add(base.replace(cl, cl[0] + "ə" + cl[1]))
    if lang == "fr":
        if base.endswith("ə"): out.add(base[:-1])
        out.add(re.sub(r"^([^aeiouɛɔœøəɑ̃ ]+)ə", r"\1", base))
    return [x for x in out if x][:16]


def rule_aware_combo(en_text, fr_text):
    qi, ci = g2p_clean(en_text, "en"), g2p_clean(fr_text, "fr")
    if not qi or not ci:
        return 0.0
    # Cap: single words only for full expansion (multi-word = too many combos)
    if " " in en_text or " " in fr_text:
        return 0.5 * ngram_dice(qi, ci) + 0.5 * feat_nw(qi, ci)
    er = _rule_variants(qi, "en")
    fr_ = _rule_variants(ci, "fr")
    best = 0.0
    for a in er:
        for b in fr_:
            s = 0.5 * ngram_dice(a, b) + 0.5 * feat_nw(a, b)
            if s > best: best = s
    return best


# ── juncture: cross-word sandhi ──
def _starts_vowel(ipa):
    for ch in ipa:
        if ch in "-ˈˌ. ": continue
        return ch in VOWELS
    return False


def juncture_ipa(words):
    n = len(words)
    if n <= 1:
        return g2p_clean(" ".join(words), "fr")
    out = []
    i = 0
    while i < n:
        w = words[i].lower().strip(".,!?;:'’")
        ip = g2p_clean(w, "fr")
        if i + 1 < n:
            nw = words[i + 1].lower().strip(".,!?;:'’")
            next_ip = g2p_clean(nw, "fr")
            if nw not in H_ASPIRE and _starts_vowel(next_ip):
                if w in ELIDE_SCHWA:
                    ip = ip[:-1] if ip.endswith("ə") else ip
                    out.append(ip); i += 1; continue
                elif w in ELIDE_A:
                    out.append(ip[:-1]); i += 1; continue
                ll = w[-1] if w else ""
                if ll in LIAISON_LETTER:
                    out.append(ip + LIAISON_LETTER[ll]); i += 1; continue
        out.append(ip); i += 1
    return "".join(out)


def juncture_score(en_text, fr_words):
    qi = g2p_clean(en_text, "en")
    ci = juncture_ipa(fr_words)
    if not qi or not ci: return 0.0
    s1 = 0.5 * ngram_dice(qi, ci) + 0.5 * feat_nw(qi, ci)
    ci2 = g2p_clean(" ".join(fr_words), "fr")
    s2 = 0.5 * ngram_dice(qi, ci2) + 0.5 * feat_nw(qi, ci2) if ci2 else 0.0
    return max(s1, s2)


# ── prosody: stress-weighted DIVERGED EN/FR ──
def _stressed_segs(text, lang):
    s_all, w_all = [], []
    raw = _espeak(text, lang)
    if not raw: return [], []
    for word in raw.split():
        demark, stops = "", {}
        for ch in word:
            if ch == "ˈ": stops[len(demark)] = PRI; continue
            if ch == "ˌ": stops[len(demark)] = SEC; continue
            if ch in ".‿|‖ˑː": continue
            demark += ch
        segs_ = list(_segs(demark))
        seg_w = [UNST] * len(segs_)
        off = 0
        for idx, s in enumerate(segs_):
            for o, sw in stops.items():
                if off <= o <= off + len(s):
                    seg_w[idx] = max(seg_w[idx], sw)
            off += len(s)
        vidx = [j for j, s in enumerate(segs_) if any(c in VOWELS for c in s)]
        for j, s in enumerate(segs_):
            if j in vidx: continue
            near = [seg_w[k] for k in vidx if abs(k - j) <= 1]
            if near:
                onset = any(k > j for k in vidx if abs(k - j) == 1)
                seg_w[j] = max(seg_w[j], max(near) * (1.0 if onset else 0.85))
        s_all += segs_; w_all += seg_w
    return s_all, w_all


def _syl_w(segs, weights):
    return [weights[i] for i, s in enumerate(segs) if any(c in VOWELS for c in s)]


def _aligned_cost(sa, wa, sb, wb):
    n, m = len(sa), len(sb)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): D[i][0] = D[i - 1][0] + GAP * wa[i - 1]
    for j in range(1, m + 1): D[0][j] = D[0][j - 1] + GAP * wb[j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            w = (wa[i - 1] + wb[j - 1]) / 2.0
            D[i][j] = min(
                D[i - 1][j - 1] + _equiv_cost(sa[i - 1], sb[j - 1]) * w,
                D[i - 1][j] + GAP * wa[i - 1],
                D[i][j - 1] + GAP * wb[j - 1])
    tot = sum(wa) + sum(wb)
    return 2.0 * D[n][m] / tot if tot else 1.0


def prosodic_score(en, fr):
    sa, wa = _stressed_segs(en, "en"); sb, wb = _stressed_segs(fr, "fr")
    if not sa or not sb: return 0.0
    match = max(0.0, 1.0 - _aligned_cost(sa, wa, sb, wb))
    sya, syb = _syl_w(sa, wa), _syl_w(sb, wb)
    rhythm = 1.0 - abs(len(sya) - len(syb)) / max(1, len(sya) + len(syb))
    if syb:
        fr_final = syb[-1] / max(syb) if max(syb) else 0
        fr_even = max(0.0, 1.0 - (np.std(syb) / (np.mean(syb) + 1e-9)))
        fr_nat = max(0.0, min(1.0, 0.6 * fr_final + 0.4 * fr_even))
    else:
        fr_nat = 0.5
    return 0.6 * match + 0.2 * fr_nat + 0.2 * rhythm


# ── semantic ──
_SEM = None


def semantic_cosine(en, fr):
    global _SEM
    if _SEM is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SEM = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            _SEM = False
    if _SEM is False: return 0.5
    try:
        v = _SEM.encode([en, fr], normalize_embeddings=True)
        return float(v[0] @ v[1])
    except Exception:
        return 0.5


# ── unified score ──
def full_score(en, fr, verbose=False):
    sc = combo(en, fr)
    sr = rule_aware_combo(en, fr)
    sj = juncture_score(en, fr.split()) if " " in fr else sc
    sp = prosodic_score(en, fr.replace("'", " "))
    ss = semantic_cosine(en, fr)
    s_phon = max(sc, sr, sj)
    joint = math.sqrt(max(0.01, s_phon) * max(0.01, ss))
    joint = 0.88 * joint + 0.12 * sp
    detail = {"combo": sc, "rule": sr, "junct": sj, "phon": s_phon,
              "pros": sp, "sem": ss, "joint": joint}
    if verbose:
        print(f"  c={sc:.2f} r={sr:.2f} j={sj:.2f} p_={s_phon:.2f} "
              f"pros={sp:.2f} sem={ss:.2f} → joint={joint:.3f}")
    return joint, detail


# ═══════════════════════════════════════════════════════════════════
# MORPHOLOGICAL DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════
def decompose(word):
    """returns (stem, stripped_suffix) or (word, '')"""
    wl = word.lower()
    if wl in _EN_MANUAL:
        return _EN_MANUAL[wl], word[len(_EN_MANUAL[wl]):]
    for suffix, replacement in _EN_SUFFIXES:
        if wl.endswith(suffix) and len(wl) - len(suffix) >= 3:
            stem = wl[:len(wl) - len(suffix)] + replacement
            return stem, suffix
    return wl, ""


# ═══════════════════════════════════════════════════════════════════
# UNIFIED DATABASE LOADER
# ═══════════════════════════════════════════════════════════════════
_DB = None


def load_database(base_dir="."):
    global _DB
    if _DB is not None: return _DB

    db = {
        "dual": defaultdict(list), "glue": defaultdict(list),
        "ladder": defaultdict(list), "tier": defaultdict(list),
        "en_class": {}, "fr_class": {},
        "syn_en": defaultdict(set), "syn_fr": defaultdict(set),
        "trans": defaultdict(set), "bridge": defaultdict(list),
        "fr_vocab": set(), "cognates": {},     # en→fr identical forms
        "fr_idx": {}, "fr_units": [],           # babel windows data
        "fr_bylen": None, "unit_bylen": None,
    }

    b = base_dir

    # ── dual-pairs ──
    try:
        for i, line in enumerate(open(f"{b}/dual-pairs.tsv", encoding="utf-8")):
            if i == 0: continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 6: db["dual"][p[0]].append((float(p[2]), p[1]))
    except FileNotFoundError: pass

    # ── tier-ladder ──
    try:
        for i, line in enumerate(open(f"{b}/tier-ladder.tsv", encoding="utf-8")):
            if i == 0: continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 12 and p[10]:
                try:
                    db["ladder"][p[1]].append((float(p[10]), float(p[11]) if p[11] else 0.5, p[2]))
                    db["tier"][p[2]].append((int(p[0]) if p[0].isdigit() else 10, float(p[10]), float(p[11]) if p[11] else 0.5, p[1]))
                except (ValueError, IndexError): continue
    except FileNotFoundError: pass

    # ── zipf-glue ──
    try:
        for i, line in enumerate(open(f"{b}/zipf-glue.tsv", encoding="utf-8")):
            if i == 0: continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3: db["glue"][p[0]].append((float(p[2]), p[1]))
    except FileNotFoundError: pass

    # ── homophone classes ──
    for path, target in [("fr-homophone-classes-lexique.tsv", "fr_class"),
                         ("fr-homophone-classes.tsv", "fr_class"),
                         ("en-homophone-classes.tsv", "en_class")]:
        try:
            for i, line in enumerate(open(f"{b}/{path}", encoding="utf-8")):
                if i == 0: continue
                ms = line.rstrip("\n").split("\t")[1].split()
                for m in ms: db[target][m] = ms
        except FileNotFoundError: pass

    # ── synonyms ──
    try:
        for line in open(f"{b}/muse-pivot-syn.tsv", encoding="utf-8"):
            a, b, _ = line.rstrip("\n").split("\t")
            if a.startswith("en:") and b.startswith("en:"):
                db["syn_en"][a[3:]].add(b[3:]); db["syn_en"][b[3:]].add(a[3:])
            elif a.startswith("fr:") and b.startswith("fr:"):
                db["syn_fr"][a[3:]].add(b[3:]); db["syn_fr"][b[3:]].add(a[3:])
    except FileNotFoundError: pass

    # ── translations (MUSE) ──
    for mp in [f"{b}/muse-en-fr.txt", "/tmp/muse-en-fr.txt"]:
        try:
            for line in open(mp, encoding="utf-8"):
                p = line.split()
                if len(p) == 2: db["trans"][p[0]].add(p[1])
            break
        except FileNotFoundError: continue

    # ── Haiku bridges ──
    try:
        for i, line in enumerate(open(f"{b}/llm-bridge.tsv", encoding="utf-8")):
            if i == 0: continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4: db["bridge"][p[0]].append((float(p[2]), float(p[3]), p[1]))
    except FileNotFoundError: pass

    # ── babel windows data (load BEFORE vocab - vocab needs fr_idx) ──
    for fp in [os.path.join(b, "fr-word-ipa.tsv"), "fr-word-ipa.tsv"]:
        try:
            for i, line in enumerate(open(fp, encoding="utf-8")):
                if i == 0: continue
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[1]: db["fr_idx"][p[0]] = p[1]
            break
        except (FileNotFoundError, OSError): continue
    for fp in [os.path.join(b, "fr-units.tsv"), "fr-units.tsv"]:
        try:
            for i, line in enumerate(open(fp, encoding="utf-8")):
                if i == 0: continue
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3: db["fr_units"].append((p[0], p[1], p[2]))
            break
        except (FileNotFoundError, OSError): continue

    # ── French vocab (fast: fr_idx keys + homophone classes) ──
    for w in db["fr_idx"]:
        db["fr_vocab"].add(w)
    for cls in db["fr_class"].values():
        for w in cls:
            if w: db["fr_vocab"].add(w)

    # ── COGNATES: identical EN/FR forms from dictionary ──
    # Only include if FR word is in Lexique vocabulary (real French)
    try:
        for i, line in enumerate(open(f"{b}/dictionary-v7.tsv", encoding="utf-8")):
            if i == 0: continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 8 and p[5] == "1":   # cognate column
                en_w, fr_w = p[6], p[7]
                if not db["fr_vocab"] or fr_w in db["fr_vocab"]:
                    db["cognates"][en_w] = fr_w
    except FileNotFoundError: pass
    # Also add from dual-pairs where en==fr AND fr is real French
    for en_w in list(db["dual"].keys()):
        for _s, fr_w in db["dual"][en_w]:
            if en_w.lower() == fr_w.lower():
                if not db["fr_vocab"] or fr_w.lower() in db["fr_vocab"]:
                    db["cognates"][en_w.lower()] = fr_w.lower()
                break
    # Common Romance cognates — Lexique-gated
    COMMON_COGNATES = {
        "nation": "nation", "nature": "nature", "table": "table",
        "art": "art", "centre": "centre", "center": "centre",
        "culture": "culture", "future": "future", "moment": "moment",
        "silence": "silence", "patience": "patience", "distance": "distance",
        "situation": "situation", "tradition": "tradition", "vision": "vision",
        "music": "musique", "possible": "possible", "impossible": "impossible",
        "terrible": "terrible", "horrible": "horrible", "normal": "normal",
        "animal": "animal", "capital": "capital", "total": "total",
        "important": "important", "different": "différent",
        "dance": "danse", "chance": "chance", "balance": "balance",
        "secret": "secret", "concert": "concert", "desert": "désert",
        "restaurant": "restaurant", "continent": "continent",
        "color": "couleur", "honor": "honneur",
    }
    for en, fr in COMMON_COGNATES.items():
        if not db["fr_vocab"] or fr in db["fr_vocab"]:
            db["cognates"][en] = fr

    # Pre-build length indices
    fr_bl = defaultdict(list)
    for w, p in db["fr_idx"].items(): fr_bl[len(_segs(p))].append((w, p))
    db["fr_bylen"] = dict(fr_bl)
    u_bl = defaultdict(list)
    for u, p, k in db["fr_units"]:
        u_bl[len(_segs(p))].append((f"{u}〔{k}〕", p))
    db["unit_bylen"] = dict(u_bl)

    # Sort
    for k in ("dual", "glue", "ladder", "tier", "bridge"):
        for v in db[k].values(): v.sort(reverse=True)

    _DB = db
    return db


# ═══════════════════════════════════════════════════════════════════
# BABEL WINDOWS — many↔one, one↔many
# ═══════════════════════════════════════════════════════════════════
def window_match(ipa, bylen, top=5, tol=2, max_cands=2000):
    """Match IPA span against length-indexed dictionary."""
    n = len(_segs(ipa))
    cands = []
    for L in range(max(1, n - tol), n + tol + 1):
        bucket = bylen.get(L, [])
        if len(cands) + len(bucket) <= max_cands:
            cands.extend(bucket)
        else:
            cands.extend(bucket[:max_cands - len(cands)])
            break
    scored = []
    for w, p in cands:
        s = 0.5 * ngram_dice(ipa, p) + 0.5 * feat_nw(ipa, p)
        if s >= 0.55: scored.append((s, w))
    scored.sort(reverse=True)
    return scored[:top]


def babel_many_to_one(gram, db, top=4):
    """2-3 EN words → 1 FR word/unit."""
    ipa = g2p_clean(gram, "en")
    if not ipa: return []
    hits = window_match(ipa, db["fr_bylen"], top=top)
    uhits = window_match(ipa, db["unit_bylen"], top=2)
    return sorted(hits + uhits, reverse=True)[:top]


def babel_one_to_many(en_word, db, top=3):
    """1 long EN word → 2 FR words via IPA split."""
    ipa = g2p_clean(en_word, "en")
    if not ipa: return []
    n = len(_segs(ipa))
    if n < 4: return []
    small = {w: p for w, p in db["fr_idx"].items() if 1 <= len(_segs(p)) <= 4}
    s_bl = defaultdict(list)
    for w, p in small.items(): s_bl[len(_segs(p))].append((w, p))
    best = []
    for cut in range(2, n - 1):
        head_ipa = "".join(_segs(ipa)[:cut])
        tail_ipa = "".join(_segs(ipa)[cut:])
        h1 = window_match(head_ipa, s_bl, top=1, tol=1)
        h2 = window_match(tail_ipa, s_bl, top=1, tol=1)
        if h1 and h2:
            s = (h1[0][0] + h2[0][0]) / 2
            best.append((s, f"{h1[0][1]} {h2[0][1]}"))
    best.sort(reverse=True)
    return [(s, fr) for s, fr in best[:top] if s >= 0.55]


# ═══════════════════════════════════════════════════════════════════
# CANDIDATE GENERATION — universal, all 11 channels
# ═══════════════════════════════════════════════════════════════════
def generate_candidates(en_word, db, top=12):
    """Generate French candidates from ALL channels, priority ordered."""
    results = []
    seen = set()

    def add(fr, meaning, channel, base_sound=None):
        fr = fr.lower().strip(".,;:!?'\"«»")
        if fr in seen or not fr: return
        # Lexique gate for single words
        if " " not in fr and db["fr_vocab"] and fr not in db["fr_vocab"]:
            return
        seen.add(fr)
        s = base_sound if base_sound is not None else rule_aware_combo(en_word, fr)
        rank = s * (0.5 + 0.5 * meaning)
        results.append((rank, s, meaning, fr, channel))

    articles = {"the": "le", "a": "un", "an": "un"}

    # 1. COGNATE — FREE, priority #1
    if en_word.lower() in db["cognates"]:
        fr = db["cognates"][en_word.lower()]
        add(fr, 1.0, "cognate", 1.0)

    # 2. DUAL one-for-ones
    for s, fr in db["dual"].get(en_word, [])[:8]:
        add(fr, 1.0, "dual", s)

    # 3. LADDER GOLD homophones (including cognates in ladder)
    for s, m, fr in db["ladder"].get(en_word, [])[:8]:
        add(fr, m, "ladder", s)

    # 4. GLUE — function-word phoneme-rule matches
    ART = {"the": {"le", "la", "les", "de", "des", "du"},
           "a": {"un", "une", "à", "et"}, "an": {"un", "une"}}
    for s, fr in db["glue"].get(en_word, [])[:6]:
        if en_word in ART and fr not in ART[en_word]: continue
        add(fr, 0.6, "glue", s)

    # 5. Haiku BRIDGES
    for s, m, fr in db["bridge"].get(en_word, [])[:4]:
        add(fr, m, "haiku", s)

    # 6. BABEL many→one (checked at phrase level — placeholder here)
    # 7. BABEL one→many
    for s, fr_pair in babel_one_to_many(en_word, db, top=3):
        add(fr_pair, 0.5, "babel_1n", s)

    # 8. EN HOMOPHONE CLASS pivot
    for sib in db["en_class"].get(en_word, [])[:6]:
        if sib == en_word: continue
        for _s, fr in db["dual"].get(sib, [])[:2]:
            add(fr, 0.6, f"enclass:{sib}")
        for _s, _m, fr in db["ladder"].get(sib, [])[:2]:
            add(fr, 0.5, f"enclass:{sib}")

    # 9. SYNONYM CHAIN EN (transitive, decay 0.85)
    syns = {en_word: 1.0}; frontier = {en_word}
    for _ in range(3):
        nxt = set()
        for x in frontier:
            for s in db["syn_en"].get(x, ()):
                if s not in syns: syns[s] = 0.85 ** len(syns); nxt.add(s)
        frontier = nxt
        if len(syns) > 30: break
    for syn, decay in sorted(syns.items(), key=lambda kv: -kv[1])[:20]:
        if syn == en_word: continue
        for fr in list(db["trans"].get(syn, []))[:3]:
            add(fr, 0.8 * decay, f"esyn:{syn}")
        for _s, fr in db["dual"].get(syn, [])[:2]:
            add(fr, 0.8 * decay, f"esyn+dual:{syn}")
        for _s, _m, fr in db["ladder"].get(syn, [])[:2]:
            add(fr, 0.7 * decay, f"esyn+gold:{syn}")

    # 10. SYNONYM CHAIN FR
    for fr0 in list(db["trans"].get(en_word, []))[:4]:
        fr_syns = {fr0: 1.0}; f_frontier = {fr0}
        for _ in range(2):
            nxt = set()
            for x in f_frontier:
                for s in db["syn_fr"].get(x, ()):
                    if s not in fr_syns: fr_syns[s] = 0.85 ** len(fr_syns); nxt.add(s)
            f_frontier = nxt
            if len(fr_syns) > 12: break
        for fr, decay in fr_syns.items():
            if fr == fr0: continue
            add(fr, 0.8 * decay, f"fsyn:{fr0}")

    # 11. METAPHOR drift (last resort)
    if not results or results[0][1] < 0.55:
        pool = {fr for _s, fr in db["dual"].get(en_word, [])}
        for fr0 in db["trans"].get(en_word, ()): pool |= db["syn_fr"].get(fr0, set())
        for fr in list(pool)[:30]:
            s = combo(en_word, fr)
            if s >= 0.55:
                m = semantic_cosine(en_word, fr)
                if m >= 0.20: add(fr, m, "metaphor", s)

    # FR HOMOPHONE CLASS meaning-max on top pick
    if results:
        top_fr = results[0][3]
        if " " not in top_fr:
            for sib in db["fr_class"].get(top_fr, [])[:6]:
                if sib not in seen and (not db["fr_vocab"] or sib in db["fr_vocab"]):
                    seen.add(sib)
                    m = semantic_cosine(en_word, sib)
                    if m > results[0][2]:
                        results.append((results[0][1] * (0.5 + 0.5 * m),
                                        results[0][1], m, sib, "frclass"))

    results.sort(reverse=True)
    return results[:top]


# ═══════════════════════════════════════════════════════════════════
# HILL-CLIMB — swap through homophone classes
# ═══════════════════════════════════════════════════════════════════
def hill_climb(en_line, fr_line, db, max_passes=4):
    best_j, _ = full_score(en_line, fr_line)
    best_m = semantic_cosine(en_line, fr_line)
    words = fr_line.split()
    for _pass in range(max_passes):
        improved = False
        for i, w in enumerate(words):
            key = w.strip(",.;:!?'\"«»").lower()
            alts = list(db["fr_class"].get(key, []))[:8]
            for alt in alts:
                if alt == key: continue
                cand = " ".join(words[:i] + [alt] + words[i + 1:])
                j2, _ = full_score(en_line, cand)
                if j2 <= best_j + 1e-6: continue
                m2 = semantic_cosine(en_line, cand)
                if m2 >= best_m - 0.05:
                    words[i] = alt; best_j, best_m = j2, m2
                    improved = True; break
            if improved: break
        if not improved: break
    return " ".join(words), best_j


# ═══════════════════════════════════════════════════════════════════
# AUTO-IMPROVE — try alternative candidate permutations
# ═══════════════════════════════════════════════════════════════════
def auto_improve(en_line, picks, db, threshold=0.45, max_tries=20):
    """If joint < threshold, try alternative candidate combinations."""
    fr_words = [fr for _, fr, _, _, _ in picks]
    best_j, _ = full_score(en_line, " ".join(fr_words))
    best_picks = list(picks)
    tries = 0
    for pos in range(len(picks)):
        w, _, _, _, _ = picks[pos]
        alt_cands = generate_candidates(w, db, top=6)
        for j, (_, s, m, fr, ch) in enumerate(alt_cands[1:], 1):
            if tries >= max_tries: break
            new_words = [p[1] for p in best_picks]
            new_words[pos] = fr
            new_fr = " ".join(new_words)
            j2, _ = full_score(en_line, new_fr)
            if j2 > best_j:
                best_j = j2
                best_picks[pos] = (w, fr, s, m, ch)
                tries += 1
                if best_j >= threshold: break
        if best_j >= threshold: break
    return best_picks, best_j


# ═══════════════════════════════════════════════════════════════════
# RECORD — write back discoveries
# ═══════════════════════════════════════════════════════════════════
_DISCOVERIES = []


def record_discovery(en, fr, joint, phonetic, semantic, channel):
    _DISCOVERIES.append({
        "en": en, "fr": fr, "joint": joint,
        "phonetic": phonetic, "semantic": semantic,
        "channel": channel, "timestamp": "",
    })


def flush_discoveries(path="new-discoveries.tsv"):
    if not _DISCOVERIES: return
    existed = os.path.exists(path)
    with open(path, "a" if existed else "w", encoding="utf-8") as f:
        if not existed:
            f.write("en\tfr\tjoint\tphonetic\tsemantic\tchannel\n")
        for d in _DISCOVERIES:
            f.write(f"{d['en']}\t{d['fr']}\t{d['joint']:.3f}\t"
                    f"{d['phonetic']:.3f}\t{d['semantic']:.3f}\t{d['channel']}\n")
    print(f"\n  [recorded {len(_DISCOVERIES)} discoveries → {path}]")
    _DISCOVERIES.clear()


# ═══════════════════════════════════════════════════════════════════
# WRITE — main homophone generation pipeline
# ═══════════════════════════════════════════════════════════════════
def write(line, db=None, verbose=True, record=True, improve_threshold=0.45):
    """Universal homophone writer: decompose → generate → score → climb → improve."""
    if db is None: db = load_database()

    ws_orig = [w.lower().strip(".,;:!?'\"") for w in line.split() if w.strip(".,;:!?'\"")]
    if not ws_orig: return None

    # ── 0. DECOMPOSE ──
    stems = []
    for w in ws_orig:
        stem, suffix = decompose(w)
        stems.append((w, stem, suffix))

    # ── 1. GENERATE candidates per word ──
    picks = []
    for orig, stem, suffix in stems:
        cands = generate_candidates(stem, db, top=10)
        # Try decomposed stem first, then original
        if not cands and stem != orig:
            cands = generate_candidates(orig, db, top=10)
        if cands:
            _, s, m, fr, ch = cands[0]
            # Morphological suffix handling: add French plural/article
            if suffix in ("s",) and not fr.endswith("s"):
                # French plurals are mostly silent - don't change
                pass
            picks.append((orig, fr, s, m, ch))
        else:
            picks.append((orig, orig, 0.0, 0.0, "miss"))

    raw_fr = " ".join(fr for _, fr, _, _, _ in picks)
    if verbose:
        print(f"Phase 1 (generate {len(picks)} words × 11 channels):")
        for w, fr, s, m, ch in picks:
            mark = " ★" if ch == "cognate" else ""
            print(f"  {w:15s} → {fr:18s} [{ch:12s}; s={s:.2f} m={m:.2f}]{mark}")

    # ── 2. BABEL WINDOW — many→one merges ──
    ws = [p[0] for p in picks]
    frs = [p[1] for p in picks]
    merged_picks = list(picks)
    i = 0
    while i < len(ws) - 1:
        for span in (3, 2):
            if i + span > len(ws): continue
            gram = " ".join(ws[i:i + span])
            bhits = babel_many_to_one(gram, db, top=3)
            if bhits and bhits[0][0] >= 0.72:
                # Replace those picks with one merged FR unit
                fr_unit = bhits[0][1].split("〔")[0]
                m = semantic_cosine(gram, fr_unit)
                merged_picks[i] = (gram, fr_unit, bhits[0][0], m, "babel_m1")
                for j in range(i + 1, i + span):
                    merged_picks[j] = None
                i += span
                break
        else:
            i += 1
    picks = [p for p in merged_picks if p is not None]
    merged_fr = " ".join(fr for _, fr, _, _, _ in picks)

    if verbose and merged_fr != raw_fr:
        print(f"\nPhase 2 (babel window):")
        for w, fr, s, m, ch in picks:
            print(f"  {w:15s} → {fr:18s} [{ch:12s}; s={s:.2f} m={m:.2f}]")

    # ── 3. HILL-CLIMB ──
    climbed_fr, joint_3 = hill_climb(line, merged_fr, db)
    if verbose and climbed_fr != merged_fr:
        print(f"\nPhase 3 (hill-climb): {merged_fr} → {climbed_fr}")

    # ── 4. SCORE ──
    joint, detail = full_score(line, climbed_fr, verbose=verbose)
    s_phon = detail["phon"]
    s_sem = detail["sem"]
    best_fr, best_joint = climbed_fr, joint

    # ── 5. AUTO-IMPROVE ──
    if best_joint < improve_threshold:
        improved_picks, imp_joint = auto_improve(line, picks, db, improve_threshold)
        if imp_joint > best_joint:
            best_fr = " ".join(fr for _, fr, _, _, _ in improved_picks)
            best_joint = imp_joint
            if verbose:
                print(f"\nPhase 5 (auto-improve): joint {joint:.3f} → {best_joint:.3f}")

    # ── 6. Juncture polish ──
    try:
        ji = juncture_ipa(best_fr.split())
        if ji:
            s_j = 0.5 * ngram_dice(g2p_clean(line, "en"), ji) + 0.5 * feat_nw(g2p_clean(line, "en"), ji)
            if s_j > s_phon + 0.02:
                s_phon = s_j
    except Exception: pass

    # ── 7. RECORD ──
    in_rooten = s_phon >= 0.55 and s_sem >= 0.45
    if record and best_joint >= 0.40:
        record_discovery(line, best_fr, best_joint, s_phon, s_sem, "triangulate")

    # ── Final report ──
    band = "✓ ROOTEN" if in_rooten else ("~ EDGE" if best_joint >= 0.45 else "  below")
    if verbose:
        print(f"\n{'='*60}")
        print(f"EN    : {line}")
        print(f"FR    : {best_fr}")
        print(f"BAND  : {band}")
        print(f"joint : {best_joint:.3f}  (phon={s_phon:.3f} sem={s_sem:.3f} pros={detail['pros']:.3f})")
        print(f"detail: c={detail['combo']:.3f} r={detail['rule']:.3f} "
              f"j={detail['junct']:.3f}")

    return {
        "en": line, "fr": best_fr, "joint": best_joint,
        "phonetic": s_phon, "semantic": s_sem,
        "prosody": detail["pros"], "combo": detail["combo"],
        "rule_aware": detail["rule"], "juncture": detail["junct"],
        "rooten_band": in_rooten,
    }


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════
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


def run_bench(n=12, db=None, record=True):
    if db is None: db = load_database()
    lines = CORPUS[:n]
    results = []
    for line in lines:
        print(f"\n{'─'*60}")
        r = write(line, db, verbose=True, record=record)
        if r: results.append(r)
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY ({len(results)} lines)")
    print(f"{'='*60}")
    if results:
        joints = [r["joint"] for r in results]
        phons = [r["phonetic"] for r in results]
        sems = [r["semantic"] for r in results]
        rooten = sum(1 for r in results if r["rooten_band"])
        print(f"Joint mean:       {np.mean(joints):.3f} ± {np.std(joints):.3f}")
        print(f"Phonetic mean:    {np.mean(phons):.3f} ± {np.std(phons):.3f}")
        print(f"Semantic mean:    {np.mean(sems):.3f} ± {np.std(sems):.3f}")
        print(f"Rooten band:      {rooten}/{len(results)} ({100*rooten/len(results):.0f}%)")
        print(f"Top 3:")
        for r in sorted(results, key=lambda x: -x["joint"])[:3]:
            print(f"  {r['joint']:.3f}  ph={r['phonetic']:.2f} se={r['semantic']:.2f}  "
                  f"{r['en']} → {r['fr']}")
    if record: flush_discoveries()


# ═══════════════════════════════════════════════════════════════════
# GENERATE FROM RULES — auto-improve loop
# ═══════════════════════════════════════════════════════════════════
def generate_loop(db=None, rounds=50):
    """Self-improving generation: try lines, record what works, improve."""
    if db is None: db = load_database()
    lines = CORPUS + PARAGRAPH
    for rnd in range(rounds):
        line = lines[rnd % len(lines)]
        r = write(line, db, verbose=False, record=True)
        if r:
            status = "✓" if r["rooten_band"] else ("~" if r["joint"] >= 0.45 else "✗")
            print(f"[{rnd+1:3d}] {status} j={r['joint']:.3f} p={r['phonetic']:.2f} "
                  f"s={r['semantic']:.2f}  {r['en']} → {r['fr']}")
    flush_discoveries()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Universal Homophone Writer v2 — orbit-level")
    ap.add_argument("text", nargs="*", help="English text to homophone")
    ap.add_argument("--bench", type=int, default=0, help="Run N-line benchmark")
    ap.add_argument("--gen", type=int, default=0, help="Self-improve for N rounds")
    ap.add_argument("--db-only", action="store_true", help="Just build unified DB and exit")
    ap.add_argument("--base-dir", default=".", help="Base directory")
    args = ap.parse_args()

    os.chdir(args.base_dir)

    print("Loading unified database...")
    db = load_database(args.base_dir)
    print(f"  dual: {sum(len(v) for v in db['dual'].values())} entries")
    print(f"  ladder: {sum(len(v) for v in db['ladder'].values())} entries")
    print(f"  cognates: {len(db['cognates'])} identified")
    print(f"  FR classes: {len(db['fr_class'])} (Lexique383)")
    print(f"  EN classes: {len(db['en_class'])}")
    print(f"  FR vocab: {len(db['fr_vocab'])}")
    print(f"  babel idx: {len(db['fr_idx'])} words, {len(db['fr_units'])} units")
    print(f"  glue: {sum(len(v) for v in db['glue'].values())} mappings")
    print(f"  bridges: {sum(len(v) for v in db['bridge'].values())}")
    print()

    if args.db_only: return
    if args.gen: generate_loop(db, args.gen); return
    if args.bench: run_bench(args.bench, db); return
    if args.text:
        for line in args.text: write(line, db, verbose=True)
    else:
        run_bench(6, db)


if __name__ == "__main__":
    main()
