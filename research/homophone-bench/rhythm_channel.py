"""N2 experiment: a PROSODY/RHYTHM channel for the matcher, measured honestly.

Hypothesis (from DEPS_RABBIT_AND_NOVEL.md N2): every existing channel scores
*segments*; none scores *rhythm*. Phrase homophony ("a name"/"an aim", the Van
Rooten effect) is metrical. So a channel that aligns two phrases as METRICAL
GRIDS (syllable nuclei + stress prominence) should help where segmental methods
struggle -- on multi-syllable PHRASES -- even if it is neutral on monosyllables.

We extract the grid from espeak stress marks (ˈ primary, ˌ secondary), score
rhythm match, and report AUC overall AND per tier, plus combo+rhythm blends, on
the same 105-pair set bench.py uses. The honest question a good linguist asks
first: this dataset is monosyllable-heavy, so does rhythm help the PHRASES it is
meant for, even if the headline barely moves?

Run: python rhythm_channel.py
"""
from __future__ import annotations

import subprocess
import unicodedata
from functools import lru_cache

import matcher
from dataset import all_pairs

STRESS_PRIMARY = "ˈ"
STRESS_SECONDARY = "ˌ"
# IPA vowel nuclei (broad; espeak en-us/fr inventory). Nasalization combines.
VOWELS = set("iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒ")


@lru_cache(maxsize=4096)
def stressed_ipa(text: str, lang: str) -> str:
    voice = "en-us" if lang == "en" else "fr"
    out = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, text],
                         capture_output=True, text=True, check=True)
    return unicodedata.normalize("NFD", out.stdout.strip())


def metrical_grid(text: str, lang: str) -> list[int]:
    """Return a per-syllable prominence sequence: 2=primary, 1=secondary, 0=none.

    A stress mark attaches to the NEXT vowel nucleus. French has no lexical
    stress (espeak marks word-final); we additionally lift the phrase-final
    syllable to >=1 to encode French phrase-final prominence.
    """
    ipa = stressed_ipa(text, lang)
    grid: list[int] = []
    pending = 0
    prev_vowel = False
    for ch in ipa:
        base = unicodedata.normalize("NFD", ch)[0]
        if ch == STRESS_PRIMARY:
            pending = 2
        elif ch == STRESS_SECONDARY:
            pending = max(pending, 1)
        elif base in VOWELS:
            if prev_vowel:
                # adjacent vowels = one diphthong nucleus, not two syllables;
                # keep the higher prominence on the merged nucleus
                grid[-1] = max(grid[-1], pending)
            else:
                grid.append(pending)
            pending = 0
            prev_vowel = True
            continue
        else:
            prev_vowel = False
    if not grid:
        return grid
    if lang == "fr":
        grid[-1] = max(grid[-1], 1)   # phrase-final prominence
    return grid


def _align_cost(a: list[int], b: list[int]) -> float:
    """Normalized edit distance over prominence sequences (Levenshtein with
    graded substitution: |level_a - level_b| / 2)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1.0
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = abs(a[i - 1] - b[j - 1]) / 2.0
            dp[i][j] = min(dp[i - 1][j - 1] + sub,
                           dp[i - 1][j] + 1.0,
                           dp[i][j - 1] + 1.0)
    return dp[n][m] / max(n, m)


def rhythm_score(en: str, fr: str) -> float:
    ge, gf = metrical_grid(en, "en"), metrical_grid(fr, "fr")
    if not ge or not gf:
        return 0.0
    # syllable-count agreement x prominence-pattern agreement
    count_sim = 1.0 - abs(len(ge) - len(gf)) / max(len(ge), len(gf))
    patt_sim = 1.0 - _align_cost(ge, gf)
    return 0.5 * count_sim + 0.5 * patt_sim


# ----------------------------------------------------------------- evaluation

def auc(pos: list[float], neg: list[float]) -> float:
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    t = len(pos) * len(neg)
    return (wins + 0.5 * ties) / t if t else 0.0


def combo(en: str, fr: str) -> float:
    return matcher.homophone_score(en, "en", fr, "fr")["score"]


def evaluate():
    pairs = all_pairs()
    rows = []
    for en, fr, label, tier in pairs:
        c = combo(en, fr)
        r = rhythm_score(en, fr)
        rows.append((en, fr, label, tier, c, r))

    def col(idx, **filt):
        out = []
        for en, fr, label, tier, c, r in rows:
            if "label" in filt and label != filt["label"]:
                continue
            if "tier" in filt and tier != filt["tier"]:
                continue
            out.append((c, r)[idx] if idx < 2 else idx(c, r))
        return out

    scorers = {
        "combo (baseline)": lambda c, r: c,
        "rhythm only": lambda c, r: r,
        "combo+rhythm avg": lambda c, r: 0.5 * c + 0.5 * r,
        "combo*2+rhythm /3": lambda c, r: (2 * c + r) / 3,
        "combo, rhythm gate": lambda c, r: c * (0.7 + 0.3 * r),
    }

    def scores(fn, **filt):
        out = []
        for en, fr, label, tier, c, r in rows:
            if filt.get("tier") and tier != filt["tier"]:
                continue
            if filt.get("label") is not None and label != filt["label"]:
                continue
            out.append((label, tier, fn(c, r)))
        return out

    print("Per-syllable metrical grids (sample):")
    for en, fr in [("Humpty Dumpty sat on a wall",
                    "Un petit d'un petit s'étonne aux Halles"),
                   ("may day", "m'aider"), ("shoe", "chou"),
                   ("bell mare", "belle mère")]:
        print(f"  {en!r:34s} {metrical_grid(en,'en')}")
        print(f"  {fr!r:34s} {metrical_grid(fr,'fr')}  rhythm={rhythm_score(en,fr):.2f}")

    print(f"\n{'scorer':22s} {'AUC':>6s} {'AUC_hard':>9s} {'AUC_phrase':>11s}")
    print("-" * 52)
    allrows = [(l, t, c, r) for *_e, l, t, c, r in
               [(en, fr, label, tier, c, r) for en, fr, label, tier, c, r in rows]]
    pos = [(l, t) for l, t, c, r in allrows if l == 1]
    for name, fn in scorers.items():
        s = [(l, t, fn(c, r)) for l, t, c, r in allrows]
        P = [v for l, t, v in s if l == 1]
        N = [v for l, t, v in s if l == 0]
        Ntr = [v for l, t, v in s if l == 0 and t == "translation"]
        # phrase sub-problem: multi-word positives vs translation negatives
        Pph = [v for (l, t, v), (en, fr, *_z) in zip(s, rows)
               if l == 1 and " " in en]
        print(f"{name:22s} {auc(P,N):6.3f} {auc(P,Ntr):9.3f} {auc(Pph,Ntr):11.3f}")

    # how many positives are actually multi-syllable / multi-word?
    nph = sum(1 for en, fr, l, t, c, r in rows if l == 1 and " " in en)
    npos = sum(1 for *_x, l, t, c, r in
               [(en, fr, label, tier, c, r) for en, fr, label, tier, c, r in rows]
               if l == 1)
    print(f"\ndataset: {npos} positives, {nph} multi-word "
          f"({100*nph//max(1,npos)}% — the rhythm channel's actual target)")


if __name__ == "__main__":
    evaluate()
