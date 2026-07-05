"""Sentence-level homophonic renderer: English text -> French word sequences
that sound like it (the Van Rooten generator). Same decoder, whole-utterance
phoneme stream, more words allowed.

Usage: python soramimi.py "Humpty Dumpty sat on a wall"
"""
import subprocess
import sys

import phonetic_decoder as pd
from lexicon_g2p import clean_ipa


def main():
    root = pd.build_trie(min_zipf=2.2)  # common words only: natural phrases
    for text in sys.argv[1:]:
        r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", text],
                           capture_output=True, text=True, check=True)
        ipa = clean_ipa(r.stdout.strip())
        print(f"\n{text!r}  [{ipa}]")
        for c in pd.decode(ipa, root, top_n=8, max_words=10):
            print(f"  {c['similarity']:.3f} cov{c['coverage']:.2f} "
                  f"sub{c['max_substitution']:.2f}  {c['fr']}")


if __name__ == "__main__":
    main()
