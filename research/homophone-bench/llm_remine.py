"""LLM-in-the-loop remining: let the model READ our fragment data + current
elision/liaison/allophonic rules and propose concrete improvements to how we
carve English sound into French (the "learn to remine for elision" loop).

It is in-context learning, not weight training: one iteration feeds the model
(1) real EN->FR phoneme-fragment mappings mined from the corpus and (2) the rules
we already encode, and asks for missing French phonological phenomena (mute-e
elision, liaison/enchaînement, h-aspiré, schwa, gemination) as codeable rules.
Output -> ELISION_PROPOSAL.md, which a human/next pass turns into matcher rules.

Run: python llm_remine.py        [FR_MODEL=... to pick the analyst model]
"""
from __future__ import annotations

import json
import os
import urllib.request

import _load_env
_load_env.load_keys()

MODEL = os.environ.get("FR_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

CURRENT_RULES = """\
Current rules we already encode (do NOT re-propose these):
- RHOTIC_MAP: French r (ʁ ʀ ɾ r) all map to English ɹ.
- NASAL_SPLIT: ɑ̃->ɑn, ɛ̃->ɛn, ɔ̃->ɔn, œ̃->œn (nasal vowel = vowel+n).
- LIAISON (word-final consonant that can bridge): s/x/z->z, d/t->t, n->n.
- EQUIV schwa class (cheap to swap): ə ɐ ɜ ʌ ɪ ʊ ɛ.
- CHEAP_GAP (cheap to delete): ʊ ɪ j w h offglides/schwa.
- FILLER words admitted free in the carve: un une le la les de des du d et est ...
"""


def sample_fragments(path="fragments.tsv", n=45):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            count, en, fr, ex = parts[:4]
            if en != fr:                      # the non-trivial allophonic mappings
                rows.append((int(count), en, fr, ex))
    rows.sort(reverse=True)
    return rows[:n]


def main():
    frags = sample_fragments()
    frag_txt = "\n".join(f"  EN /{en}/ -> FR /{fr}/  (x{c})  e.g. {ex[:60]}"
                         for c, en, fr, ex in frags)
    prompt = f"""You are a French phonologist helping a homophonic-translation engine
(English text rewritten as French that sounds the same, Van Rooten style). The
engine carves an English phoneme stream into French words. Below are real EN->FR
phoneme-FRAGMENT correspondences it mined (IPA), where the English chunk differs
from the French chunk, and the rules it already applies.

{CURRENT_RULES}

Mined fragment correspondences (English sound -> French sound it was matched to):
{frag_txt}

Task: propose ADDITIONAL French phonological rules that would let the carve match
more English sound to natural French — focus on ELISION and connected speech:
mute-e (schwa) elision and when French keeps vs drops it, liaison/enchaînement
contexts, h-aspiré vs h-muet, vowel hiatus, gemination, final-consonant silence.
For EACH rule give: (1) a one-line name, (2) the phonological condition, (3) a
concrete IPA example of an English fragment it would newly let us carve, (4) how
to encode it (an EQUIV pair, a CHEAP_GAP, a LIAISON entry, or a rewrite). Be
concrete and codeable. Number them. Keep under 600 words."""

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("No OPENROUTER_API_KEY; cannot run the advisor."); return
    body = json.dumps({"model": MODEL, "temperature": 0.3,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1800, "reasoning": {"enabled": False}}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 data=body, headers={"Authorization": f"Bearer {key}",
                                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    txt = out["choices"][0]["message"]["content"]
    with open("ELISION_PROPOSAL.md", "w", encoding="utf-8") as f:
        f.write(f"# LLM elision/remining proposal ({MODEL})\n\n")
        f.write("_In-context advisory from the LLM-in-the-loop; review before "
                "encoding into matcher.py._\n\n")
        f.write(txt)
    print(f"analyst: {MODEL}\n" + "=" * 60)
    print(txt)


if __name__ == "__main__":
    main()
