"""Try to GENERATE homophonic Mother Goose with the reranked v5 table, and
diagnose why it is hard. Public-domain source lines only (The Real Mother Goose,
1916). No copyrighted rendering is reproduced; we GENERATE our own and inspect
the failure modes.

Two generation modes, to separate the causes:
  A. word-for-word: replace each English word by its reranked-v5 French
     homophone (boundaries PRESERVED). Tests the table directly.
  B. (diagnostic) compare against the one attested human line already in the
     repo to see what word-for-word structurally cannot do.

Run: python test_mothergoose_gen.py
"""
from __future__ import annotations

import subprocess

import matcher

try:
    import bigram_lm
    FRLM = bigram_lm.load("fr")
except Exception:
    FRLM = None


def load_reranked(path="dictionary-v5-reranked.tsv"):
    best = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                best[p[0]] = (p[1], float(p[2]))
    return best


def ipa(text, voice):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, text],
                       capture_output=True, text=True, check=True)
    import unicodedata
    return matcher._canonical("".join(
        c for c in unicodedata.normalize("NFD", r.stdout.strip())
        if c not in {"ˈ", "ˌ", "‿", ".", " ", "\n"}))


def coherence(fr):
    toks = [t.lower() for t in fr.replace("'", " ").split() if t]
    if FRLM and toks:
        return FRLM.fluency(toks)
    return 0.0


# Public-domain Mother Goose lines (1916, Gutenberg #10607). Short, famous.
LINES = [
    "Jack and Jill went up the hill",
    "Hickory dickory dock",
    "Pat a cake pat a cake",
    "Hey diddle diddle",
    "Little Jack Horner sat in a corner",
    "Humpty Dumpty sat on a wall",
]


def main():
    best = load_reranked()
    print(f"reranked table: {len(best)} headwords\n")

    for line in LINES:
        words = [w.lower().strip(".,!?;:") for w in line.split()]
        subs, gaps = [], 0
        for w in words:
            if w in best:
                subs.append(best[w][0])
            else:
                subs.append(f"[{w}]")
                gaps += 1
        fr_w4w = " ".join(subs)
        cov = (len(words) - gaps) / len(words)
        # sound fidelity of the concatenated FR vs the English LINE
        fr_clean = " ".join(s for s in subs if not s.startswith("["))
        snd = (0.5 * matcher._ngram_channel(ipa(line, "en-us"), ipa(fr_clean, "fr"))
               + 0.5 * matcher._feat_channel(ipa(line, "en-us"), ipa(fr_clean, "fr"))) \
            if fr_clean else 0.0
        coh = coherence(fr_clean)
        print(f"EN: {line}")
        print(f"  w4w FR : {fr_w4w}")
        print(f"  coverage {cov:.0%}  line-sound {snd:.2f}  L2coh {coh:.2f}")
        print()

    # the structural point, against the one attested human line in the repo
    print("=" * 70)
    print("Why word-for-word cannot reach the art (boundary count):")
    en = "Humpty Dumpty sat on a wall"          # 6 English words
    human = "Un petit d'un petit s'étonne aux Halles"  # attested, in repo
    print(f"  EN ({len(en.split())} words):    {en}")
    print(f"  human FR ({len(human.replace(chr(39),' ').split())} words): {human}")
    print(f"  EN IPA:    {ipa(en,'en-us')}")
    print(f"  human IPA: {ipa(human,'fr')}")
    print("  -> the French word boundaries fall in DIFFERENT places than the")
    print("     English: 'Humpty' (1 word) maps across 'un petit' (2 words).")
    print("     Word-for-word substitution preserves English boundaries, so it")
    print("     can NEVER produce this. The stream must be RE-CUT (resegmented).")


if __name__ == "__main__":
    main()
