"""GRAMMAR-AWARE SENTENCE FORMER -- assemble matches into real French sentence
structure, modelling the effect of each choice on its NEIGHBOURS.

The decoder produces sound-true word sequences ("con sait neuf") with no idea of
French syntax. This layer adds the missing linguistics, in four passes:

  1. TAG      every word: POS + gender + number from Lexique 3.83 (pylexique),
              with a closed-class function-word table taking precedence
              (Lexique tags "la" as NOM(la musique)/PRO/ART; we force DET etc.).
  2. SYNTAX   score the tag sequence with an explicit French POS-transition
              grammar (hand-encoded, no corpus needed): DET wants a NOM/ADJ
              next, clitic pronouns want a VERB, prepositions must not end the
              sentence, adjectives sit post-nominal by default...
  3. REPAIR   the neighbourhood effects -- what French does to ADJACENT words:
                a. det-noun agreement    le mer -> la mer, un série -> une série
                b. mandatory elision     le ami -> l'ami, de un -> d'un
                c. prep-article fusion   de le -> du, à les -> aux
                d. plural spreading      les âne -> les ânes (det number wins)
              Repairs CHANGE the surface, hence the phoneme stream.
  4. RESCORE  sound of the repaired sentence via the juncture layer (whole-
              phrase espeak + symbolic liaison/elision), because the repairs
              live exactly at the boundaries the citation-form judge forgets.

  joint = sound^SND_POW * syntax^SYN_POW * fluency^FLU_POW

Run:  python sentence_former.py "the sea is cold" "according to"
      python sentence_former.py --demo
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from functools import lru_cache

from wordfreq import zipf_frequency

import juncture
import phonetic_decoder as pd
import phrase_weave as pw

# ---- knobs ----
SND_POW, SYN_POW, FLU_POW = 1.0, 0.55, 0.35
TOP_DECODE = 60          # raw carves pulled from the decoder before forming
MIN_SOUND = 0.80

# ------------------------------------------------------------------ lexicon

_LEXIQUE = None


def _lexique():
    global _LEXIQUE
    if _LEXIQUE is None:
        from pylexique import Lexique383
        _LEXIQUE = Lexique383()
    return _LEXIQUE


def _norm_lig(w: str) -> list[str]:
    """Try both ligature spellings (œufs / oeufs)."""
    a = w
    b = w.replace("œ", "oe").replace("æ", "ae")
    return [a] if a == b else [a, b]


# Closed-class scaffold of French: these tags OVERRIDE Lexique, because the
# function words are exactly where Lexique is ambiguous (la: ART/PRO/NOM).
DETS = {
    # surface: (gender, number, family) -- family used for agreement swaps
    "le": ("m", "s", "def"), "la": ("f", "s", "def"), "les": (None, "p", "def"),
    "l'": (None, "s", "def"),
    "un": ("m", "s", "ind"), "une": ("f", "s", "ind"), "des": (None, "p", "ind"),
    "du": ("m", "s", "part"), "au": ("m", "s", "fus"), "aux": (None, "p", "fus"),
    "ce": ("m", "s", "dem"), "cet": ("m", "s", "dem"), "cette": ("f", "s", "dem"),
    "ces": (None, "p", "dem"),
    "mon": ("m", "s", "pos"), "ma": ("f", "s", "pos"), "mes": (None, "p", "pos"),
    "son": ("m", "s", "pos"), "sa": ("f", "s", "pos"), "ses": (None, "p", "pos"),
    "notre": (None, "s", "pos"), "votre": (None, "s", "pos"),
    "nos": (None, "p", "pos"), "vos": (None, "p", "pos"), "leur": (None, "s", "pos"),
    "leurs": (None, "p", "pos"),
}
DET_FAMILY = {  # family -> {(gender, number): surface}
    "def": {("m", "s"): "le", ("f", "s"): "la", ("m", "p"): "les", ("f", "p"): "les"},
    "ind": {("m", "s"): "un", ("f", "s"): "une", ("m", "p"): "des", ("f", "p"): "des"},
    "dem": {("m", "s"): "ce", ("f", "s"): "cette", ("m", "p"): "ces", ("f", "p"): "ces"},
    "pos": {("m", "s"): "son", ("f", "s"): "sa", ("m", "p"): "ses", ("f", "p"): "ses"},
}
CLITIC_SUBJ = {"je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles"}
PREPS = {"de", "à", "en", "dans", "sur", "sous", "avec", "sans", "pour", "par",
         "chez", "vers", "dès", "entre", "contre"}
CONJ = {"et", "ou", "mais", "donc", "or", "ni", "car"}
REL = {"que", "qui", "dont", "où"}
NEG = {"ne", "pas", "plus", "jamais", "rien"}
# frequent pre-nominal adjectives (French default is post-nominal)
PRENOMINAL = {"petit", "petite", "petits", "grand", "grande", "grands", "bon",
              "bonne", "bons", "beau", "belle", "beaux", "jeune", "vieux",
              "vieille", "gros", "grosse", "haut", "haute", "long", "longue"}


@lru_cache(maxsize=50000)
def tag(word: str) -> tuple[str, str | None, str | None]:
    """(POS, gender, number). POS in {DET,NOM,ADJ,VER,PRO,PRE,ADV,CON,NEG,X}."""
    w = word.lower()
    if w in DETS:
        g, n, _ = DETS[w]
        return "DET", g, n
    if w in CLITIC_SUBJ:
        return "PRO", None, None
    if w in PREPS:
        return "PRE", None, None
    if w in CONJ or w in REL:
        return "CON", None, None
    if w in NEG:
        return "NEG", None, None
    lx = _lexique()
    best = None
    for form in _norm_lig(w):
        e = lx.lexique.get(form)
        if e is None:
            continue
        entries = e if isinstance(e, list) else [e]
        for x in entries:
            f = float(getattr(x, "freqlivres", 0) or 0)
            if best is None or f > best[0]:
                best = (f, x)
    if best is None:
        # unknown word: guess noun (open class default); gender by suffix
        g = "f" if w.endswith(("tion", "sion", "té", "ée", "ie", "ure", "ance",
                               "ence", "ette")) else "m"
        return "NOM", g, "p" if w.endswith(("s", "x")) else "s"
    x = best[1]
    cg = str(getattr(x, "cgram", "") or "")
    pos = ("DET" if cg.startswith("ART") else
           "NOM" if cg.startswith("NOM") else
           "ADJ" if cg.startswith("ADJ") else
           "VER" if cg.startswith(("VER", "AUX")) else
           "ADV" if cg.startswith("ADV") else
           "PRO" if cg.startswith("PRO") else
           "CON" if cg.startswith("CON") else "X")
    g = getattr(x, "genre", None) or None
    n = getattr(x, "nombre", None) or None
    return pos, g, n


# ------------------------------------------------- French POS-transition grammar

# log-ish weights in (0,1]; 1.0 = perfectly natural adjacency.
_T = {
    ("DET", "NOM"): 1.0, ("DET", "ADJ"): 0.75, ("DET", "X"): 0.3,
    ("ADJ", "NOM"): 0.85, ("NOM", "ADJ"): 0.85,
    ("NOM", "VER"): 0.8, ("NOM", "PRE"): 0.85, ("NOM", "CON"): 0.7,
    ("NOM", "NOM"): 0.25, ("NOM", "DET"): 0.15, ("NOM", "PRO"): 0.4,
    ("PRO", "VER"): 1.0, ("PRO", "NEG"): 0.8, ("PRO", "PRO"): 0.5,
    ("VER", "DET"): 0.85, ("VER", "PRE"): 0.8, ("VER", "ADV"): 0.75,
    ("VER", "NOM"): 0.45, ("VER", "VER"): 0.4, ("VER", "PRO"): 0.5,
    ("VER", "ADJ"): 0.55, ("VER", "CON"): 0.6,
    ("PRE", "DET"): 0.95, ("PRE", "NOM"): 0.7, ("PRE", "VER"): 0.5,
    ("PRE", "PRO"): 0.35, ("PRE", "ADJ"): 0.3,
    ("ADV", "VER"): 0.6, ("ADV", "ADJ"): 0.7, ("ADV", "DET"): 0.5,
    ("CON", "DET"): 0.8, ("CON", "PRO"): 0.8, ("CON", "NOM"): 0.6,
    ("CON", "VER"): 0.6,
    ("NEG", "VER"): 0.9, ("VER", "NEG"): 0.8,
}
_START = {"DET": 1.0, "PRO": 1.0, "NOM": 0.6, "PRE": 0.5, "ADV": 0.5,
          "CON": 0.25, "VER": 0.4, "ADJ": 0.3, "NEG": 0.4, "X": 0.2}
_END = {"NOM": 1.0, "ADJ": 0.9, "VER": 0.8, "ADV": 0.8, "PRO": 0.35,
        "NEG": 0.6, "X": 0.4, "DET": 0.05, "PRE": 0.05, "CON": 0.05}
_T_DEFAULT = 0.2


def syntax_score(words: list[str]) -> float:
    """Geometric-mean naturalness of the POS chain under the French grammar."""
    if not words:
        return 0.0
    poss = [tag(w)[0] for w in words]
    parts = [_START.get(poss[0], 0.2)]
    for a, b in zip(poss, poss[1:]):
        w = _T.get((a, b), _T_DEFAULT)
        # pre-nominal adjectives are the exception, not the rule
        if (a, b) == ("ADJ", "NOM") and words[poss.index(a)] not in PRENOMINAL:
            w = min(w, 0.6)
        parts.append(w)
    parts.append(_END.get(poss[-1], 0.3))
    p = 1.0
    for x in parts:
        p *= x
    return p ** (1.0 / len(parts))


# ------------------------------------------------------------------ repairs

def _vowel_initial(word: str) -> bool:
    ipa = juncture._word_ipa(word, "fr")
    return juncture._starts_with_vowel(ipa) and \
        word.lower() not in juncture.H_ASPIRE


def repair(words: list[str]) -> tuple[list[str], list[str]]:
    """Apply French neighbourhood grammar; return (new_words, notes)."""
    out = list(words)
    notes: list[str] = []
    # a. det-noun agreement + d. plural spreading (det number wins on the noun
    #    only when the noun form exists; otherwise noun's number wins on det)
    for i, w in enumerate(out[:-1]):
        wl = w.lower()
        if wl in DETS:
            g_d, n_d, fam = DETS[wl]
            # find the head: next NOM within 2 words (allow one adjective)
            j = i + 1
            if j < len(out) and tag(out[j])[0] == "ADJ" and j + 1 < len(out):
                j += 1
            pos_h, g_h, n_h = tag(out[j])
            if pos_h != "NOM" or fam not in DET_FAMILY:
                continue
            g = g_h or g_d or "m"
            n = n_d or n_h or "s"   # determiner number is the speaker's choice
            new = DET_FAMILY[fam].get((g, n))
            if new and new != wl:
                notes.append(f"agree: {w} {out[j]} -> {new} {out[j]}")
                out[i] = new
            # plural spreading onto the noun (les âne -> les ânes), only when
            # the pluralised form is a real Lexique word
            if n == "p" and (n_h or "s") == "s":
                for suf in ("s", "x"):
                    cand = out[j] + suf
                    if any(_lexique().lexique.get(f) for f in _norm_lig(cand)):
                        notes.append(f"number: {out[j]} -> {cand}")
                        out[j] = cand
                        break
    # c. prep-article fusion (must run before elision eats the article)
    i = 0
    fused: list[str] = []
    while i < len(out):
        w, nxt = out[i].lower(), (out[i + 1].lower() if i + 1 < len(out) else "")
        pair = {("de", "le"): "du", ("de", "les"): "des",
                ("à", "le"): "au", ("à", "les"): "aux"}.get((w, nxt))
        if pair:
            notes.append(f"fuse: {w} {nxt} -> {pair}")
            fused.append(pair)
            i += 2
        else:
            fused.append(out[i])
            i += 1
    out = fused
    # b. mandatory elision before vowel-initial words
    ELIDE = {"le": "l'", "la": "l'", "de": "d'", "ne": "n'", "que": "qu'",
             "je": "j'", "me": "m'", "te": "t'", "se": "s'", "ce": "c'"}
    for i in range(len(out) - 1):
        wl = out[i].lower()
        if wl in ELIDE and _vowel_initial(out[i + 1]):
            notes.append(f"elide: {out[i]} {out[i+1]} -> {ELIDE[wl]}{out[i+1]}")
            out[i] = ELIDE[wl] + out[i + 1]
            out[i + 1] = ""
    out = [w for w in out if w]
    return out, notes


# ------------------------------------------------------------------ forming

def _fluency(words: list[str]) -> float:
    if not words:
        return 0.0
    zs = [min(zipf_frequency(w.strip("'").split("'")[-1], "fr"), 6.0) / 6.0
          for w in words]
    return sum(zs) / len(zs)


def _junc_combo(en_ipa: str, fr_words: list[str]) -> float:
    """Best juncture-aware combo of a French surface vs the EN stream."""
    best = 0.0
    for r in juncture.juncture_realizations(fr_words, "fr"):
        best = max(best, juncture._combo_ipa(juncture.bench.canonical(en_ipa), r))
    return best


MIN_COVERAGE = 0.80      # a carve must account for (nearly) the whole EN stream


def form(text: str, root, top_n: int = 6) -> list[dict]:
    """English sentence -> grammar-formed French carves, best first."""
    en_ipa = pw.phrase_ipa(text, "en")
    cands = pd.decode(en_ipa, root, top_n=TOP_DECODE, max_words=pw.MAX_WORDS)
    seen: set[str] = set()
    rows: list[dict] = []
    for c in cands:
        if c["similarity"] < MIN_SOUND or c["expensive_deletions"] > 0:
            continue
        if c.get("coverage", 1.0) < MIN_COVERAGE:
            continue   # partial stubs must not outrank full carves on syntax
        raw = c["fr"].split()
        formed, notes = repair(raw)
        key = " ".join(formed)
        if key in seen:
            continue
        seen.add(key)
        # sound = calibrated decoder similarity, lifted (never lowered) by the
        # boundary-phonology effect of the repairs, measured with one scorer.
        lift = 0.0
        if formed != raw:
            lift = max(0.0, _junc_combo(en_ipa, formed) - _junc_combo(en_ipa, raw))
        snd = min(1.0, c["similarity"] + lift)
        syn = syntax_score(formed)
        flu = _fluency(formed)
        joint = ((snd ** SND_POW) * (syn ** SYN_POW) * (flu ** FLU_POW)
                 * c.get("coverage", 1.0))
        rows.append({
            "src": text, "raw": " ".join(raw), "formed": key,
            "sound": round(snd, 3), "syntax": round(syn, 3),
            "fluency": round(flu, 3), "coverage": round(c.get("coverage", 1.0), 3),
            "joint": round(joint, 3),
            "repairs": notes,
        })
    rows.sort(key=lambda r: -r["joint"])
    return rows[:top_n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences", nargs="*")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    sentences = args.sentences or []
    if args.demo or not sentences:
        sentences = ["the sea is cold", "according to", "the sun is gone",
                     "come to me now", "two men under the moon"]
    print("building fr trie...", file=sys.stderr)
    root = pd.build_trie(min_zipf=2.0, lang="fr")
    for s in sentences:
        rows = form(s, root)
        print(f"\nEN: {s}")
        if not rows:
            print("    (nothing formed)")
            continue
        for r in rows:
            mark = " *" if r["repairs"] else ""
            print(f"  FR: {r['formed']:30s} snd {r['sound']:.2f} "
                  f"syn {r['syntax']:.2f} flu {r['fluency']:.2f} "
                  f"joint {r['joint']:.2f}{mark}")
            for n in r["repairs"]:
                print(f"        [{n}]")
            if r["formed"] != r["raw"]:
                print(f"        (raw: {r['raw']})")


if __name__ == "__main__":
    main()
