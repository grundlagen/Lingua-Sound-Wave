"""N1 experiment: cycle-consistency as a LABEL-FREE quality signal.

Idea (DEPS_RABBIT_AND_NOVEL.md N1): a true homophone round-trips. Decode an
English word's phoneme stream into French, then decode THAT French phrase's
phoneme stream back into English. If the path is genuinely sound-preserving,
the recovered English IPA should land back near where it started. The
round-trip distortion is computable with NO labels and NO trusted scorer in
the loop -- it is pure self-supervision (the back-translation / cycle-GAN trick
applied to sound).

The honest test a good engineer demands: is the label-free cycle signal
actually informative? We check whether low cycle-distortion AGREES with the
trusted AUC-0.993 matcher on the (en, fr) pair it produced. If a label-free
quantity tracks the hand-validated scorer, N1's premise holds: you can rank /
filter / eventually train on cycle-consistency without ever consulting labels.

      en --G2P--> ipa_en --decode(FR)--> fr (ipa_fr) --decode(EN)--> en' (ipa_en')
      cycle_fidelity = nw_sim(ipa_en, ipa_en')          # label-free
      forward_combo  = matcher.combo(en, fr)            # trusted, for validation only

Run: python cycle_consistency.py [n_words=120]
"""
from __future__ import annotations

import subprocess
import sys
import unicodedata
from functools import lru_cache

from wordfreq import top_n_list

import matcher
import phonetic_decoder as pd
from lexicon_g2p import load_en


def auc(pos, neg):
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    t = len(pos) * len(neg)
    return (wins + 0.5 * ties) / t if t else 0.0


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


@lru_cache(maxsize=8192)
def g2p(text, lang):
    voice = "en-us" if lang == "en" else "fr"
    out = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, text],
                         capture_output=True, text=True, check=True)
    return matcher._canonical(
        "".join(c for c in unicodedata.normalize("NFD", out.stdout.strip())
                if c not in {"ˈ", "ˌ", "‿", ".", " ", "\n"}))


def best(ipa, root, lang):
    cands = pd.decode(ipa, root, top_n=4)
    for c in cands:
        if c["expensive_deletions"] == 0:
            return c
    return cands[0] if cands else None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    pd.BEAM = 200
    lex_en = load_en()
    en_root = pd.build_trie(min_zipf=2.5, lang="en")
    fr_root = pd.build_trie(min_zipf=2.5, lang="fr")

    words = [w for w in top_n_list("en", 4000)
             if w.isalpha() and len(w) >= 3 and w in lex_en][:n]

    rows = []
    for w in words:
        ipa_en = (lex_en.get(w) or [None])[0]
        if not ipa_en:
            continue
        ipa_en = matcher._canonical(ipa_en)
        fr_c = best(ipa_en, fr_root, "fr")
        if not fr_c:
            continue
        fr_phrase = fr_c["fr"]
        ipa_fr = g2p(fr_phrase, "fr")
        en_c = best(ipa_fr, en_root, "en")
        if not en_c:
            continue
        ipa_en2 = g2p(en_c["fr"], "en")          # decode() labels output "fr"
        cycle = matcher.nw_sim_ipa(ipa_en, ipa_en2)        # LABEL-FREE
        fwd = matcher.homophone_score(w, "en", fr_phrase, "fr")["score"]  # trusted
        rows.append((w, fr_phrase, en_c["fr"], cycle, fwd))

    rows.sort(key=lambda r: -r[3])
    print(f"\n{len(rows)} round-trips.  en -> fr -> en', cycle-fidelity vs trusted combo\n")
    print(f"{'en':12s} {'fr (homophone)':22s} {'en-prime':14s} {'cycle':>6s} {'combo':>6s}")
    print("-" * 64)
    for w, fr, en2, cyc, fwd in rows[:18]:
        print(f"{w:12s} {fr:22s} {en2:14s} {cyc:6.2f} {fwd:6.2f}")
    print("  ...")
    for w, fr, en2, cyc, fwd in rows[-6:]:
        print(f"{w:12s} {fr:22s} {en2:14s} {cyc:6.2f} {fwd:6.2f}")

    cyc = [r[3] for r in rows]
    fwd = [r[4] for r in rows]
    r = pearson(cyc, fwd)

    # Does the label-free cycle signal recover the trusted scorer's verdict?
    # Treat combo>=0.55 as "trusted good"; can cycle-fidelity rank them up?
    good = [c for c, f in zip(cyc, fwd) if f >= 0.55]
    bad = [c for c, f in zip(cyc, fwd) if f < 0.55]
    a = auc(good, bad)

    print(f"\nPearson(cycle, combo) = {r:+.3f}")
    print(f"AUC of label-free cycle-fidelity vs trusted-combo verdict "
          f"(>=0.55): {a:.3f}")
    print(f"  ({len(good)} trusted-good, {len(bad)} trusted-weak)")
    print("\nReading: if AUC >> 0.5, the cycle signal -- computed with NO labels")
    print("and NO trusted scorer -- recovers the hand-validated matcher's verdict,")
    print("which is exactly what a self-supervised training target must do (N1).")


if __name__ == "__main__":
    main()
