"""Dictionary-based G2P from the user's recovered lexicon data
(checkcehck/homophone/homophone-agent-audio):

  - data/lexique.tsv     Lexique 3-derived French word -> IPA (245k rows)
  - data/lexique_en.tsv  WikiPron-derived English word -> IPA (65k rows, UK)

Curated dictionary pronunciations beat espeak G2P where they exist: no
language-switch bugs (espeak says English 'doss' for French 'dos'), no
liaison contamination, real lexicographer transcriptions. espeak remains the
fallback for words outside the dictionaries and for whole phrases.
"""
from __future__ import annotations

import os
import unicodedata

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# strip stress/length/ties/zero-width characters; keep nasal combining tilde
_DROP = {"ˈ", "ˌ", "‿", ".", "|", "‖", " ", "\t", "-", "_",
         "‍", "‌", "͡", "͜", "(", ")", "ˑ", "ʼ", ",", "ʲ"}


def clean_ipa(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    # Lexique writes /ɡ/ as ASCII "g", which panphon silently drops —
    # normalize to the real IPA character before anything else sees it.
    s = s.replace("g", "ɡ")
    return "".join(ch for ch in s if ch not in _DROP)


def _parse_prons(raw: str, max_prons: int = 2) -> list[str]:
    """Handle multi-pronunciation cells like 'tu/, /tut'."""
    parts = [p.strip(" ,/") for p in raw.split("/")]
    out = []
    for p in parts:
        c = clean_ipa(p)
        if c and c not in out:
            out.append(c)
    return out[:max_prons]


def load_lexicon(path: str) -> dict[str, list[str]]:
    lex: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            w = parts[0].strip().lower()
            if not w or not w.isalpha():
                continue
            prons = _parse_prons(parts[1])
            if not prons:
                continue
            existing = lex.setdefault(w, [])
            for p in prons:
                if p not in existing and len(existing) < 2:
                    existing.append(p)
    return lex


def load_fr() -> dict[str, list[str]]:
    return load_lexicon(os.path.join(DATA_DIR, "lexique.tsv"))


_ARPA = {
    "AA": "ɑ", "AE": "æ", "AH0": "ə", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ",
    "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ",
    "ER": "ɚ", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ",
    "IY": "i", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s",
    "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}


def _arpa_to_ipa(phones: list[str]) -> str:
    out = []
    for p in phones:
        if p in _ARPA:                       # stressless consonants
            out.append(_ARPA[p]); continue
        base, digit = p[:-1], p[-1]
        if digit.isdigit():
            if p[:2] == "AH" and digit == "0":
                out.append(_ARPA["AH0"]); continue
            out.append(_ARPA.get(base, ""))
        else:
            out.append(_ARPA.get(p, ""))
    return "".join(out)


def load_en(include_cmu: bool = True) -> dict[str, list[str]]:
    """WikiPron (UK) merged with CMUdict (US) — two real accent variants
    per word, and CMUdict roughly doubles coverage (126k entries)."""
    lex = load_lexicon(os.path.join(DATA_DIR, "lexique_en.tsv"))
    if include_cmu:
        try:
            import cmudict
            for w, prons in cmudict.dict().items():
                if not w.isalpha() or len(w) < 2:
                    continue
                existing = lex.setdefault(w, [])
                for ph in prons[:1]:
                    ipa = clean_ipa(_arpa_to_ipa(ph))
                    if ipa and ipa not in existing and len(existing) < 2:
                        existing.append(ipa)
        except ImportError:
            pass
    return lex


if __name__ == "__main__":
    fr = load_fr()
    en = load_en()
    print(f"FR lexicon: {len(fr)} words, EN lexicon: {len(en)} words")
    for w in ["dos", "chou", "tout", "guette", "mère"]:
        print(f"  fr {w}: {fr.get(w)}")
    for w in ["dough", "shoe", "west", "said", "holly"]:
        print(f"  en {w}: {en.get(w)}")
