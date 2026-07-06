"""French sandhi rules — the "babel rules" engine for sentence-level carving.

Deterministic word-boundary phonology: elision, liaison (obligatory /
forbidden / optional), h-aspiré blocking, enchaînement annotation, and
optional schwa drop. One phonology, three consumers:

  1. training-data build — sandhi-process FR targets so the model learns
     what a sentence SOUNDS like, not how it is written
  2. the DPO reward — score combo against sandhi(fr), not written fr
  3. Agent C — judge the spoken stream

Pure Python, no deps. Works at the orthographic level and emits the spoken
word stream plus realized liaison consonants; feed the result to FR g2p for
the phoneme stream. Self-test: python sandhi_fr.py
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ── vowel-initial detection ─────────────────────────────────────────────────

# h-aspiré words: written h blocks elision AND liaison ("le héros", not "l'héros").
# Lexical, not rule-derived — this list covers the frequent cases; extend freely.
H_ASPIRE = {
    "hache", "haine", "haïr", "hall", "halle", "halte", "hamac", "hameau",
    "hanche", "handicap", "hangar", "hanter", "harceler", "hardi", "harem",
    "hareng", "haricot", "harnais", "harpe", "hasard", "hâte", "hausse",
    "haut", "haute", "hauteur", "havre", "hennir", "hérisson", "héron",
    "héros", "hêtre", "hibou", "hideux", "hiérarchie", "hockey", "hollande",
    "homard", "honte", "hoquet", "hors", "houle", "housse", "huit",
    "huitième", "hurler", "hutte",
}

# 'onze', 'oui', 'yaourt', 'yoga' etc. behave like consonant-initial
DISJUNCTIVE_VOWEL = {"onze", "onzième", "oui", "ouistiti", "yaourt", "yacht",
                     "yoga", "yo-yo", "uhlan", "ululer"}


def _strip_accents(w: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", w)
                   if unicodedata.category(c) != "Mn")


def starts_vowel_sound(word: str) -> bool:
    """True if the word begins with a vowel sound for elision/liaison purposes."""
    w = word.lower().strip("'’")
    if not w:
        return False
    if w in DISJUNCTIVE_VOWEL or w in H_ASPIRE:
        return False
    base = _strip_accents(w)
    if base[0] in "aeiouy":
        return True
    if base[0] == "h":
        return w not in H_ASPIRE  # h muet: 'homme', 'heure', 'hiver' → vowel
    return False


# ── elision ─────────────────────────────────────────────────────────────────

# word → elided form before a vowel sound (obligatory in standard French)
ELISION = {
    "le": "l'", "la": "l'", "je": "j'", "me": "m'", "te": "t'", "se": "s'",
    "ne": "n'", "de": "d'", "que": "qu'", "ce": "c'", "jusque": "jusqu'",
    "lorsque": "lorsqu'", "puisque": "puisqu'", "quoique": "quoiqu'",
    # 'si' elides only before il/ils
}


# ── liaison ─────────────────────────────────────────────────────────────────

# latent final consonant realized in liaison, by final letter(s)
LATENT = [
    (re.compile(r"(s|x|z)$"), "z"),
    (re.compile(r"(t|d)$"), "t"),
    (re.compile(r"n$"), "n"),
    (re.compile(r"(r)$"), "r"),   # only in fixed contexts; kept optional
    (re.compile(r"(p)$"), "p"),   # trop, beaucoup
    (re.compile(r"g$"), "g"),     # long (archaic /k/, modern /g/)
]

# word classes that liaise obligatorily into a following vowel-initial word
OBLIG_LIAISON_WORDS = {
    # determiners
    "les", "des", "ces", "mes", "tes", "ses", "nos", "vos", "leurs", "aux",
    "un", "deux", "trois", "six", "dix", "aucun", "tout", "quels", "quelles",
    "quelques", "plusieurs", "certains", "certaines",
    # clitic pronouns
    "nous", "vous", "ils", "elles", "on", "en",
    # common monosyllabic preps/adverbs
    "en", "dans", "chez", "sans", "sous", "très", "plus", "bien", "moins",
    "rien", "tout",
    # forms of être commonly liaised
    "est", "sont", "était", "étaient", "suis", "es",
    # petit/grand/gros/premier/dernier etc. — prenominal adjectives
    "petit", "petits", "grand", "grands", "gros", "premier", "dernier",
    "bon", "mauvais", "deux",
}

# liaison NEVER happens after these (or across these boundaries)
FORBIDDEN_AFTER = {"et"}


@dataclass
class SpokenWord:
    written: str
    spoken: str                 # form actually pronounced (elided if applicable)
    liaison: str | None = None  # consonant carried INTO the next word, if any
    enchainement: bool = False  # final pronounced C resyllabifies into next word
    notes: list[str] = field(default_factory=list)


def apply_sandhi(text: str, schwa_drop: bool = False) -> list[SpokenWord]:
    """Apply elision + liaison + enchaînement annotation to a French line."""
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    out: list[SpokenWord] = []
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        nxt_vowel = starts_vowel_sound(nxt) if nxt else False
        lw = w.lower()
        sw = SpokenWord(written=w, spoken=w)

        # elision (obligatory)
        if nxt_vowel and lw in ELISION:
            sw.spoken = ELISION[lw]
            sw.notes.append(f"elision:{lw}→{ELISION[lw]}")
        elif lw == "si" and nxt and nxt.lower() in ("il", "ils"):
            sw.spoken = "s'"
            sw.notes.append("elision:si→s'")

        # liaison
        elif nxt_vowel and lw not in FORBIDDEN_AFTER:
            if lw in OBLIG_LIAISON_WORDS:
                for pat, cons in LATENT:
                    if pat.search(_strip_accents(lw)):
                        sw.liaison = cons
                        sw.notes.append(f"liaison:+{cons}‿")
                        break
            # enchaînement: an already-pronounced final consonant re-syllabifies
            elif re.search(r"[bcdfgklmnpqrstvz]e?$", lw) and not lw.endswith(("es", "s", "x", "t", "d")):
                sw.enchainement = True
                sw.notes.append("enchaînement")

        # optional schwa drop in flow (style knob, off by default)
        if schwa_drop and len(lw) > 3 and lw.endswith("e") and lw not in ELISION:
            sw.notes.append("schwa-drop?")

        out.append(sw)
    return out


def spoken_stream(text: str, schwa_drop: bool = False) -> str:
    """The line as spoken: elided forms joined, liaison consonants attached
    to the FOLLOWING word (where the ear hears them). Feed this to FR g2p."""
    ws = apply_sandhi(text, schwa_drop=schwa_drop)
    parts: list[str] = []
    carry = ""
    for sw in ws:
        tok = carry + sw.spoken
        carry = ""
        if sw.liaison:
            carry = sw.liaison + "‿"
        if sw.spoken.endswith("'"):
            # elided clitic fuses with the next word
            carry = tok + carry
            continue
        parts.append(tok)
    if carry:
        parts.append(carry.rstrip("‿"))
    return " ".join(parts)


# ── self-test ───────────────────────────────────────────────────────────────

def _selftest() -> int:
    cases = [
        # (input, expected spoken stream)
        ("le ami", "l'ami"),
        ("le héros", "le héros"),                    # h-aspiré blocks elision
        ("la heure", "l'heure"),                     # h muet elides
        ("les amis", "les z‿amis"),                  # obligatory liaison /z/
        ("les héros", "les héros"),                  # h-aspiré blocks liaison
        ("et alors", "et alors"),                    # 'et' never liaises
        ("petit ami", "petit t‿ami"),                # prenominal adj liaison /t/
        ("nous avons", "nous z‿avons"),
        ("un grand homme", "un grand t‿homme"),      # h muet + liaison
        ("je aime", "j'aime"),
        ("que il", "qu'il"),
        ("si il", "s'il"),
        ("deux enfants", "deux z‿enfants"),
        ("chez elle", "chez z‿elle"),
        ("le onze", "le onze"),                      # disjunctive 'onze'
    ]
    failed = 0
    for src, want in cases:
        got = spoken_stream(src)
        ok = got == want
        failed += not ok
        print(f"{'ok ' if ok else 'FAIL'}  {src!r:24} → {got!r}"
              + ("" if ok else f"   (want {want!r})"))
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _selftest() else 0)
