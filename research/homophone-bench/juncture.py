"""CROSS-WORD JUNCTURE scoring -- the connected-speech rules that bite at WORD
BOUNDARIES, which neither the citation-form judge nor the within-word rule_aware
layer can reach.

Why this is a *different* gap from rule_aware.py
------------------------------------------------
`rule_aware.py` expands each word's own pronunciation (flapping, l-vocalization,
h-dropping, th-fronting, schwa-drop). Those are all WITHIN a single token.
`ELISION_PROPOSAL.md` ("Closing the loop") flagged the rules it could NOT encode:

    elision-before-vowel, liaison, enchainement, h-aspire block

are *context-sensitive across the word boundary* -- whether they fire depends on
the FOLLOWING word's first sound. A matcher that scores word units in isolation
(which is exactly what the decoder/`phrase_weave` does when it composes a French
carve from a word-keyed trie) never sees them.

Concretely, espeak applied to the WHOLE phrase already does French liaison:

    "les amis"  whole-phrase -> le-zami     (liaison /z/ surfaces)
    "les"+"amis" word-by-word -> le- + ami   (the /z/ is LOST)

So a carve like "les amis" is *under-rated* by the per-unit judge: spoken, it
liaises to /lezami/ and matches an English source far better than /leami/ does.

What this module does
---------------------
Reconstruct the connected-speech realization of a French word SEQUENCE and score
the best realization against the source -- the same honest "upper envelope" idea
as rule_aware: every realization here is a legal pronunciation (liaison is a real
feature of careful French; elision of le/de/... before a vowel is obligatory), so
it can only RAISE a true homophone, never invent one.

Two reconstruction routes, both used (max over them):
  1. espeak on the joined surface phrase -- espeak implements liaison/elision well
     where the orthography licenses it (e.g. "l'ami").
  2. a symbolic juncture pass over the per-unit citation IPA -- needed when the
     decoder emits a unit whose surface does NOT pre-spell the sandhi (it emits
     "le" + "ami", never "l'ami"), so espeak alone keeps /le.ami/. The symbolic
     pass applies elision and liaison from the spelling + the next unit's onset.

Run: python juncture.py
"""
from __future__ import annotations

import subprocess

import numpy as np

import bench
from rule_aware import _ngram_ipa, _feat_ipa

VOWELS = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɑɒ"

# Function words that ELIDE their final vowel before a vowel-initial word.
# (le/de/... drop /ə/ -> l'/d'; la drops /a/; que -> qu'.)  Obligatory in French.
ELIDE_SCHWA = {"le", "de", "je", "me", "te", "se", "ce", "ne", "que",
               "jusque", "lorsque", "puisque", "quoique", "parce que"}
ELIDE_A = {"la"}

# Word-final orthographic letter -> the consonant it surfaces as in LIAISON,
# but only when that consonant is otherwise LATENT (silent in citation form).
#   s/x/z -> [z]   t/d -> [t]   n -> [n]   p -> [p]   r -> [ʁ→ɹ]   g -> [k]   f -> [v]
LIAISON_BY_LETTER = {"s": "z", "x": "z", "z": "z", "t": "t", "d": "t",
                     "n": "n", "p": "p", "r": "ɹ", "g": "k", "f": "v"}

# h-aspire words block liaison AND elision (le|héros, les|haricots). Small high-
# frequency set; espeak knows most, but the symbolic pass needs them explicitly.
H_ASPIRE = {"héros", "haricot", "haricots", "hibou", "hache", "hauteur",
            "honte", "hangar", "hasard", "hâte", "halte", "hamac", "hareng",
            "haine", "hall", "hamster", "handicap",
            # A15 extension (IDEAS_BABEL): the rest of the frequent aspirates
            "haut", "hauts", "haute", "hautes", "hors", "huit", "hurler",
            "hurle", "houle", "housse", "hotte", "horde", "hoquet", "homard",
            "hockey", "hocher", "heurter", "heurt", "hêtre", "héron", "hernie",
            "harpe", "harnais", "hargne", "harem", "hardi", "harceler",
            "hanter", "hante", "hanche", "hameau", "hublot", "huer", "hutte",
            "hâtent", "hâtes", "hausse", "hautain", "hasards", "hiboux"}


def _voice(lang: str) -> str:
    return {"en": "en-us", "fr": "fr"}.get(lang, lang)


def _espeak_phrase(text: str, lang: str) -> str:
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", _voice(lang), text],
                       capture_output=True, text=True, check=True)
    return bench.canonical(bench.normalize_ipa(r.stdout.strip()))


def _word_ipa(word: str, lang: str) -> str:
    return bench.canonical(bench.g2p_ipa(word, lang))


def _starts_with_vowel(ipa: str) -> bool:
    for ch in ipa:
        if ch in "-ˈˌ. ":
            continue
        return ch in VOWELS
    return False


def _last_letter(word: str) -> str:
    for ch in reversed(word.lower()):
        if ch.isalpha():
            return ch
    return ""


def symbolic_juncture(words: list[str], lang: str = "fr") -> str:
    """Concatenate per-unit citation IPA, applying elision + liaison across each
    word boundary from the spelling and the next unit's onset."""
    units = [(w, _word_ipa(w, lang)) for w in words]
    out: list[str] = []
    n = len(units)
    for i, (w, ip) in enumerate(units):
        wl = w.lower().strip(".,!?;:'’")
        nxt_vowel = False
        if i + 1 < n:
            nw = units[i + 1][0].lower().strip(".,!?;:'’")
            nxt_vowel = _starts_with_vowel(units[i + 1][1]) and nw not in H_ASPIRE
        seg = ip
        # --- elision: drop the elidable final vowel before a vowel-initial word
        if nxt_vowel and wl not in H_ASPIRE:
            if wl in ELIDE_SCHWA and seg.rstrip("-").endswith("ə"):
                seg = seg.rstrip("-")[:-1]
            elif wl in ELIDE_A and seg.rstrip("-").endswith("a"):
                seg = seg.rstrip("-")[:-1]
        out.append(seg)
        # --- liaison: surface a latent final consonant before a vowel-initial word
        if nxt_vowel and wl not in H_ASPIRE:
            lc = LIAISON_BY_LETTER.get(_last_letter(wl))
            # only liaise if the consonant is genuinely latent (not already the
            # last sound of the citation form, e.g. "sept" /sɛt/ already ends /t/)
            if lc and not seg.rstrip("-").endswith(lc):
                out.append(lc)
    return bench.canonical("".join(out))


def citation_concat(words: list[str], lang: str = "fr") -> str:
    """Per-unit citation IPA glued together -- what the isolated-word judge sees."""
    return bench.canonical("".join(_word_ipa(w, lang) for w in words))


def juncture_realizations(words: list[str], lang: str = "fr") -> list[str]:
    """Connected-speech realizations of a word sequence: citation concat (the
    no-sandhi floor), espeak whole-phrase, and the symbolic juncture pass."""
    reals = {
        citation_concat(words, lang),
        _espeak_phrase(" ".join(words), lang),
        symbolic_juncture(words, lang),
    }
    return [r for r in reals if r]


def _combo_ipa(a: str, b: str) -> float:
    return 0.5 * _ngram_ipa(a, b) + 0.5 * _feat_ipa(a, b)


def juncture_score(src_text: str, tgt_words: list[str],
                   src_lang: str = "en", tgt_lang: str = "fr") -> tuple[float, float]:
    """(citation combo, best juncture-aware combo) for a source phrase vs a
    target word-sequence carve."""
    src_ipa = bench.canonical(bench.g2p_ipa(src_text, src_lang))
    cite = _combo_ipa(src_ipa, citation_concat(tgt_words, tgt_lang))
    best = cite
    for r in juncture_realizations(tgt_words, tgt_lang):
        best = max(best, _combo_ipa(src_ipa, r))
    return cite, best


def juncture_rescore(src_ipa: str, tgt_words: list[str], tgt_lang: str,
                     base_sound: float) -> float:
    """Re-rank hook for phrase_weave: lift a candidate's sound score to its best
    connected-speech realization. Never lowers it (upper envelope)."""
    best = base_sound
    for r in juncture_realizations(tgt_words, tgt_lang):
        best = max(best, _combo_ipa(bench.canonical(src_ipa), r))
    return best


def main() -> None:
    # French carves whose homophony with the English source exists only ACROSS the
    # word boundary (liaison z/t/n, elision of le/de/la). Citation-form scoring of
    # the isolated units misses it; juncture scoring recovers it.
    tests = [
        ("lazy me", ["les", "amis"]),       # z-liaison  le-z-ami
        ("they zon", ["deux", "ans"]),      # z-liaison  dø-z-ɑ̃
        ("voozavay", ["vous", "avez"]),     # z-liaison  vu-z-ave
        ("noozom", ["nous", "hommes"]),     # z-liaison  nu-z-ɔm
        ("petit tammy", ["petit", "ami"]),  # t-liaison  pəti-t-ami
        ("grand tom", ["grand", "homme"]),  # t-liaison (d->t)  grɑ̃-t-ɔm
        ("bonn ami", ["bon", "ami"]),       # n-liaison  bɔ̃-n-ami
        ("lammy", ["le", "ami"]),           # elision    l'ami  /lami/
        ("dunn ami", ["de", "un", "ami"]),  # elision    d'un ami
        ("la mee", ["la", "amie"]),         # elision    l'amie /lami/
    ]
    print("citation-form combo  vs  JUNCTURE combo (cross-word liaison/elision)\n")
    print(f"{'EN source':13s}{'FR carve':17s}{'cite':>6s}{'+junc':>7s}{'gain':>7s}")
    print("-" * 50)
    gains = []
    for en, fr in tests:
        try:
            c, j = juncture_score(en, fr)
        except Exception as ex:  # noqa: BLE001
            print(f"{en:13s}{' '.join(fr):17s}  ERROR {ex}")
            continue
        g = j - c
        gains.append(g)
        flag = "  <- liaison/elision surfaces the match" if g > 0.02 else ""
        print(f"{en:13s}{' '.join(fr):17s}{c:6.2f}{j:7.2f}{g:+7.2f}{flag}")
    if gains:
        print(f"\nmean lift from cross-word juncture rules: {np.mean(gains):+.3f}  "
              f"({sum(g > 0.02 for g in gains)}/{len(gains)} carves raised)")
    print("\nReading: a carve assembled from word-keyed units is scored in citation "
          "form, so its liaison /z,t,n/ and the elision of le/de/la before a vowel "
          "are invisible to the judge. Re-scoring the carve's connected-speech "
          "realization (espeak whole-phrase + a symbolic liaison/elision pass) is "
          "the honest upper envelope -- it only RAISES carves that are true "
          "homophones once spoken naturally, the cross-word complement to "
          "rule_aware's within-word rules.")


if __name__ == "__main__":
    main()
