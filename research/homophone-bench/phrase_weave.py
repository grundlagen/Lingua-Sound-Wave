"""Phrase-to-phrase homophonic weave with a fluency prior on BOTH sides.

Motivation (the gap this fills)
-------------------------------
`dictionary-v5` has 1,299 single-EN -> multi-FR pairs, 104 multi-EN pairs, and
*zero* multiword<->multiword pairs.  The decoder already turns one phoneme
stream into a multi-word sequence on the target side, but:

  1. it ranks by SOUND only (FREQ_BONUS is 0.05*zipf, almost noise), so it
     surfaces sound-perfect non-words -- "any -> haie nient", "very -> vais
     rient".  "nient"/"rient"/"ès" are not clean French.
  2. there is no prior making the OTHER side a fluent, common-word phrase, so
     composed English comes out small and nonsensical.

This module keeps the proven engine (espeak G2P -> matcher equivalence-floored
costs -> Lexique trie beam in phonetic_decoder) and adds the one missing piece:
a real **language-model prior on both languages** so a candidate only wins when
it is *both* a close sound match *and* made of common, naturally-adjacent words.

    joint(EN, FR) = sound(EN_ipa, FR_ipa)  *  fluency(EN)^a  *  fluency(FR)^b

Running it in both directions (decode an English phrase into French words; decode
a French phrase into English words) and keeping only pairs where BOTH sides clear
a fluency gate is what finally yields phrase<->phrase homophones where neither
side is gibberish.

Recursive bootstrap (`--certify`): pairs that clear sound AND dual-fluency are
appended to `certified-phrase-pairs.tsv`; those are the new fragments a future
re-mining pass folds back into the dictionary (densify -> re-search), exactly the
explode_web loop-certification idea, but generating *new fluent phrase pairs*
instead of only recycling existing edges.

NOTE: needs espeak-ng on PATH and `panphon numpy wordfreq cmudict` installed
(same stack as phonetic_decoder.py).  Run from research/homophone-bench/.

Usage:
  python phrase_weave.py "the sea is cold"                # EN -> FR phrases
  python phrase_weave.py --src fr "tout doux ma belle"    # FR -> EN phrases
  python phrase_weave.py --certify "the sea is cold" ...  # also append gold pairs
"""
from __future__ import annotations

import subprocess
import sys
from functools import lru_cache

from wordfreq import zipf_frequency

import phonetic_decoder as pd
from lexicon_g2p import clean_ipa

# ---- knobs ----
MAX_WORDS = 8          # both sides may be multiword (the point of the module)
SOUND_MIN = 0.80       # candidate must still sound alike
FLUENCY_GATE = 2.5     # every output word must clear this zipf (kills "nient")
SOUND_POW = 1.0
FLUENCY_POW = 0.8
TOP_N = 40


def _voice(lang: str) -> str:
    return {"en": "en-us", "fr": "fr"}.get(lang, lang)


def phrase_ipa(text: str, lang: str) -> str:
    """Whole-phrase IPA (liaison/elision included -- we WANT connected speech
    here, unlike the isolated-word dictionary builders)."""
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", _voice(lang), text],
                       capture_output=True, text=True, check=True)
    return clean_ipa(r.stdout.strip())


@lru_cache(maxsize=100000)
def _zipf(word: str, lang: str) -> float:
    return zipf_frequency(word, lang)


def fluency(words: list[str], lang: str) -> float:
    """Mean clamped zipf over the words, in [0,1].  A phrase of common words
    scores ~1; a phrase containing a rare token or non-word is dragged down.
    Cheap, deterministic, no model download."""
    if not words:
        return 0.0
    zs = [min(_zipf(w, lang), 6.0) / 6.0 for w in words]
    return sum(zs) / len(zs)


def min_zipf_word(words: list[str], lang: str) -> float:
    return min((_zipf(w, lang) for w in words), default=0.0)


def transfer(text: str, src_lang: str, tgt_lang: str, root,
             *, top_n: int = TOP_N, clean_only: bool = True) -> list[dict]:
    """Decode a source phrase into TARGET word sequences, re-ranked by a joint
    sound*fluency objective so the target side is fluent, not just sound-true."""
    ipa = phrase_ipa(text, src_lang)
    cands = pd.decode(ipa, root, top_n=top_n, max_words=MAX_WORDS)
    src_words = [w.lower().strip(".,!?;:") for w in text.split()]
    src_flu = fluency(src_words, src_lang)

    out = []
    for c in cands:
        if c["similarity"] < SOUND_MIN or c["expensive_deletions"] > 0:
            continue
        tgt_words = c["fr"].split()
        if clean_only and min_zipf_word(tgt_words, tgt_lang) < FLUENCY_GATE:
            continue  # the line that removes "nient"/"ès"/obscure forms
        tgt_flu = fluency(tgt_words, tgt_lang)
        joint = (c["similarity"] ** SOUND_POW) * (tgt_flu ** FLUENCY_POW)
        out.append({
            "src": text, "tgt": c["fr"], "sound": round(c["similarity"], 3),
            "src_fluency": round(src_flu, 3), "tgt_fluency": round(tgt_flu, 3),
            "joint": round(joint, 3), "words": c["words"],
            "coverage": c["coverage"], "ipa": ipa,
        })
    out.sort(key=lambda r: -r["joint"])
    # de-dup target strings, keep best
    seen, dedup = set(), []
    for r in out:
        if r["tgt"] in seen:
            continue
        seen.add(r["tgt"])
        dedup.append(r)
    return dedup[:top_n]


def both_intelligible(text: str, src_lang: str, tgt_lang: str, root,
                      *, min_tgt_fluency: float = 0.55) -> list[dict]:
    """Keep only transfers where the SOURCE was a fluent phrase and the TARGET
    is fluent too -- i.e. neither side is gibberish.  This is the property the
    user asked for: both languages intelligible at once."""
    rows = transfer(text, src_lang, tgt_lang, root)
    src_words = [w.lower().strip(".,!?;:") for w in text.split()]
    if fluency(src_words, src_lang) < min_tgt_fluency:
        return []  # source itself wasn't a natural phrase
    return [r for r in rows if r["tgt_fluency"] >= min_tgt_fluency]


def main():
    args = [a for a in sys.argv[1:]]
    src_lang = "en"
    if "--src" in args:
        i = args.index("--src")
        src_lang = args[i + 1]
        del args[i:i + 2]
    certify = "--certify" in args
    args = [a for a in args if not a.startswith("--")]
    tgt_lang = "fr" if src_lang == "en" else "en"

    print(f"building {tgt_lang} trie (min_zipf=2.0)...", file=sys.stderr)
    root = pd.build_trie(min_zipf=2.0, lang=tgt_lang)

    sentences = args or ["the sea is cold", "two men under the moon",
                         "she said tell me more"]
    gold = []
    out_lines = []
    for sent in sentences:
        rows = both_intelligible(sent, src_lang, tgt_lang, root)
        out_lines.append(f"\n{src_lang.upper()}: {sent}")
        if not rows:
            out_lines.append("    (no candidate with both sides fluent)")
            continue
        for r in rows[:6]:
            out_lines.append(
                f"  {tgt_lang.upper()}: {r['tgt']:32s}  "
                f"snd {r['sound']:.2f}  flu {r['tgt_fluency']:.2f}  "
                f"joint {r['joint']:.2f}  [{r['words']}w]")
        # certify the top pair if it is genuinely strong on both axes
        best = rows[0]
        if certify and best["sound"] >= 0.88 and best["tgt_fluency"] >= 0.6:
            gold.append(best)

    text = "\n".join(out_lines)
    print(text)

    if certify and gold:
        with open("certified-phrase-pairs.tsv", "a", encoding="utf-8") as f:
            for r in gold:
                f.write(f"{r['src']}\t{r['tgt']}\t{src_lang}\t{tgt_lang}"
                        f"\t{r['sound']}\t{r['tgt_fluency']}\t{r['joint']}\n")
        print(f"\ncertified {len(gold)} phrase pair(s) -> "
              f"certified-phrase-pairs.tsv (re-mining seed)", file=sys.stderr)


if __name__ == "__main__":
    main()
