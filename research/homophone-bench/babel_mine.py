"""BABEL MINE -- institute the mineable IDEAS_BABEL routes, BOTH directions.

Every trick mirrored: if French gets it, English gets it too.

  C27  homophone classes    FR: vert=verre=vers=ver=vair; EN: their=there,
       (both languages)     sea=see -- group top-N words by identical espeak
                            IPA -> {fr,en}-homophone-classes.tsv. Free meaning
                            pivots: any member matches the sound, CHOOSE by sense.
  A9   elision units (FR)   l'/d'/j'/qu'/n'/m'/t'/s'/c' + vowel-initial word ->
                            one sound unit (d'or, l'âme) -> fr-units.tsv
  A10  liaison units (FR)   les|mes|des|... + vowel word with the liaison
                            consonant REALIZED (les amis = lezami) -> fr-units.tsv
  MIRROR (EN)               poetic contractions ('tis,'twas,o'er,e'er,ne'er) and
                            weak forms (and=n', of=o') -> en-units.tsv
  D29/32/33 lexicons        archaic FR, French place-names, interjections --
                            appended to fr-units.tsv with register tags
  E36  rhyme index          group both lexicons by final 2 IPA segments ->
                            rhyme-index.tsv (compose to rhyme for free)
  B17  window indexes       word->IPA tables double as the many-to-one lookup:
                            composer slides EN n-grams against FR index and FR
                            n-grams against EN index (babel_windows.py)

Run: python babel_mine.py [--n_fr 20000] [--n_en 20000]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from wordfreq import top_n_list

import matcher


def ipa_of(words, lang):
    out = {}
    for i, w in enumerate(words):
        try:
            out[w] = matcher._canonical(matcher.g2p(w, lang))
        except Exception:
            continue
        if i % 4000 == 0:
            print(f"  {lang} g2p {i}/{len(words)}", file=sys.stderr)
    return out


def homophone_classes(ipa, path):
    by = defaultdict(list)
    for w, p in ipa.items():
        if p:
            by[p].append(w)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("ipa\tmembers\n")
        for p, ws in sorted(by.items(), key=lambda kv: -len(kv[1])):
            if len(ws) >= 2:
                f.write(f"{p}\t{' '.join(sorted(ws))}\n")
                n += 1
    print(f"{path}: {n} homophone classes")
    return by


ELIDERS = ["l'", "d'", "j'", "qu'", "n'", "m'", "t'", "s'", "c'"]
LIAISON = {"les": "z", "des": "z", "mes": "z", "ses": "z", "tes": "z", "nos": "z",
           "vos": "z", "aux": "z", "deux": "z", "trois": "z", "six": "z",
           "dix": "z", "chez": "z", "très": "z", "nous": "z", "vous": "z",
           "ils": "z", "elles": "z", "un": "n", "on": "n", "en": "n",
           "mon": "n", "ton": "n", "son": "n", "bien": "n", "rien": "n",
           "tout": "t", "petit": "t", "grand": "t", "quand": "t", "est": "t"}
VOWEL0 = "aeiouâàéèêëîïôöûüœh"

ARCHAIC = ["ores", "oncques", "moult", "céans", "icelle", "icelui", "jadis",
           "las", "point", "fort", "guère", "maintes", "sitôt", "tantôt"]
PLACES = ["lille", "caen", "tours", "nice", "metz", "sète", "pau", "gap",
          "albi", "auch", "foix", "lyon", "arles", "brest", "sens", "laon"]
INTERJ = ["oh", "ah", "hé", "hein", "bah", "ouf", "aïe", "eh", "ho", "hop",
          "chut", "zut", "hélas", "ma foi", "dame", "tiens"]
EN_POETIC = ["'tis", "'twas", "o'er", "e'er", "ne'er", "'neath", "oft", "ere",
             "yon", "hark", "lo", "alas", "nay", "aye", "thee", "thou", "thy"]


def fr_units(fr_ipa, path):
    rows = []
    vowel_words = [w for w in fr_ipa if w and w[0] in VOWEL0][:4000]
    for el in ELIDERS:                                  # A9 elision units
        for w in vowel_words:
            unit = el + w
            try:
                p = matcher._canonical(matcher.g2p(unit, "fr"))
                if p:
                    rows.append((unit, p, "elision"))
            except Exception:
                pass
    for lead, cons in LIAISON.items():                  # A10 liaison units
        pl = fr_ipa.get(lead)
        if not pl:
            continue
        for w in vowel_words[:1500]:
            pw = fr_ipa.get(w)
            if pw:
                rows.append((f"{lead} {w}", pl + cons + pw, "liaison"))
    for lex, tag in ((ARCHAIC, "archaic"), (PLACES, "place"), (INTERJ, "interj")):
        for w in lex:
            try:
                p = matcher._canonical(matcher.g2p(w, "fr"))
                if p:
                    rows.append((w, p, tag))
            except Exception:
                pass
    with open(path, "w", encoding="utf-8") as f:
        f.write("unit\tipa\tkind\n")
        for u, p, k in rows:
            f.write(f"{u}\t{p}\t{k}\n")
    print(f"{path}: {len(rows)} French units (elision/liaison/archaic/place/interj)")


def en_units(path):
    rows = []
    for w in EN_POETIC:
        try:
            p = matcher._canonical(matcher.g2p(w.replace("'", ""), "en"))
            if p:
                rows.append((w, p, "poetic"))
        except Exception:
            pass
    with open(path, "w", encoding="utf-8") as f:
        f.write("unit\tipa\tkind\n")
        for u, p, k in rows:
            f.write(f"{u}\t{p}\t{k}\n")
    print(f"{path}: {len(rows)} English poetic units (the mirror)")


def rhyme_index(fr_ipa, en_ipa, path):
    by = defaultdict(lambda: defaultdict(list))
    for lang, ipa in (("fr", fr_ipa), ("en", en_ipa)):
        for w, p in ipa.items():
            segs = matcher._segs(p)
            if len(segs) >= 2:
                by["".join(segs[-2:])][lang].append(w)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("rime\tfr\ten\n")
        for r, d in sorted(by.items(), key=lambda kv: -(len(kv[1]["fr"]) + len(kv[1]["en"]))):
            if d["fr"] and d["en"]:
                f.write(f"{r}\t{' '.join(sorted(d['fr'])[:12])}\t{' '.join(sorted(d['en'])[:12])}\n")
                n += 1
    print(f"{path}: {n} cross-language rhyme families")


def word_ipa_tsv(ipa, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("word\tipa\n")
        for w, p in ipa.items():
            if p:
                f.write(f"{w}\t{p}\n")
    print(f"{path}: {len(ipa)} word->IPA rows (window-index fuel)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_fr", type=int, default=20000)
    ap.add_argument("--n_en", type=int, default=20000)
    args = ap.parse_args()

    fr_words = [w for w in top_n_list("fr", args.n_fr) if w.isalpha()]
    en_words = [w for w in top_n_list("en", args.n_en) if w.isalpha()]
    print(f"g2p: {len(fr_words)} FR + {len(en_words)} EN words", file=sys.stderr)
    fr_ipa = ipa_of(fr_words, "fr")
    en_ipa = ipa_of(en_words, "en")

    homophone_classes(fr_ipa, "fr-homophone-classes.tsv")   # C27
    homophone_classes(en_ipa, "en-homophone-classes.tsv")   # C27 mirror
    fr_units(fr_ipa, "fr-units.tsv")                        # A9+A10+D29/32/33
    en_units("en-units.tsv")                                # mirror
    rhyme_index(fr_ipa, en_ipa, "rhyme-index.tsv")          # E36
    word_ipa_tsv(fr_ipa, "fr-word-ipa.tsv")                 # B17 fuel
    word_ipa_tsv(en_ipa, "en-word-ipa.tsv")                 # B17 mirror fuel
    print("\nBabel mine complete.")


if __name__ == "__main__":
    main()
