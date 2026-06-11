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


def load_en() -> dict[str, list[str]]:
    return load_lexicon(os.path.join(DATA_DIR, "lexique_en.tsv"))


if __name__ == "__main__":
    fr = load_fr()
    en = load_en()
    print(f"FR lexicon: {len(fr)} words, EN lexicon: {len(en)} words")
    for w in ["dos", "chou", "tout", "guette", "mère"]:
        print(f"  fr {w}: {fr.get(w)}")
    for w in ["dough", "shoe", "west", "said", "holly"]:
        print(f"  en {w}: {en.get(w)}")
