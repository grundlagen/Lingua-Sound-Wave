"""Van Rooten / poetry mode for the resegmentation engine.

What "messing with the matching" on Humpty revealed:
  - the filler words van Rooten leans on -- un /œ̃/, et /e/, on /ɔ̃/, a /a/,
    aux /o/, y /i/, eau /o/ -- are ONE segment, and the decoder's MIN_WORD_SEGS=2
    excludes them. Those are literally "the extra words we don't have".
  - the decoder's WORD_PENALTY rewards FEWER, LONGER words ("épidémie"), the
    opposite of the homophonic-poetry style (many short words + fillers).
  - pure sound even prefers "épidémie" over a "un petit"-style carve, so SENSE
    must override raw sound -- the coherence model has to carry it.

Principled fix (this file), without destabilising the default decoder:
  - a FILLER whitelist of common 1-segment French function words is admitted at
    length 1; all CONTENT words still require >=2 segments (no confetti);
  - WORD_PENALTY relaxed so the short-word carve can win;
  - selection stays sound x coherence so the fillers serve a fluent line.

Run: python poetry_mode.py
"""
from __future__ import annotations

import subprocess

import matcher
from matcher import _canonical
import phonetic_decoder as pd
from lexicon_g2p import load_fr, clean_ipa

try:
    import bigram_lm
    LM = bigram_lm.load("fr")
except Exception:
    LM = None

# Van Rooten's toolkit: monosyllabic French function/filler words that fill meter
# and bridge sound. Admitted even at 1 segment (content words still need >=2).
FILLER = {"un", "une", "le", "la", "les", "de", "des", "du", "d", "et", "est",
          "à", "a", "au", "aux", "on", "ont", "en", "y", "ô", "eau", "où", "ou",
          "il", "ils", "elle", "se", "ce", "ne", "que", "qui", "ses", "tes"}


def build_poetry_trie(min_zipf=2.0):
    """Trie that admits FILLER words at length 1, content words at length >=2."""
    fr = load_fr()
    from wordfreq import zipf_frequency
    root = pd.Node()
    n = 0
    for w, prons in fr.items():
        if len(w) < 1:
            continue
        z = zipf_frequency(w, "fr")
        if z < min_zipf:
            continue
        min_segs = 1 if w in FILLER else 2
        for p in prons:
            segs = matcher._segs(_canonical(p))
            if len(segs) < min_segs:
                continue
            node = root
            for s in segs:
                node = node.children.setdefault(s, pd.Node())
            node.words.append((w, z))
            n += 1
    return root


def en_ipa(text):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", text],
                       capture_output=True, text=True, check=True)
    return clean_ipa(r.stdout.strip())


def coherence(fr):
    toks = [t.lower() for t in fr.replace("'", " ").split() if t]
    return LM.fluency(toks) if (LM and toks) else 0.0


def decode_poetry(text, root, max_words=7, top_n=8):
    ipa = en_ipa(text)
    cands = pd.decode(ipa, root, top_n=top_n, max_words=max_words,
                      lm=LM, lm_weight=pd.LM_BEAM_WEIGHT if LM else 0.0)
    out = []
    for c in cands:
        fillers = sum(1 for w in c["fr"].split() if w in FILLER)
        out.append((c["similarity"] * coherence(c["fr"]), c["similarity"],
                    coherence(c["fr"]), c["fr"], c["words"], fillers))
    out.sort(reverse=True)
    return out


def main():
    print("Poetry mode: filler words ON (1-seg whitelist), splits allowed.\n")
    pd.BEAM = 350
    pd.MIN_WORD_SEGS = 1          # the decode loop must allow 1-seg emission...
    pd.WORD_PENALTY = 0.04        # ...but only just; content words still need 2 segs via the trie
    root = build_poetry_trie(min_zipf=2.0)

    for text in ["Humpty Dumpty", "Humpty Dumpty sat on a wall",
                 "Jack and Jill", "Hickory dickory dock"]:
        print(f"EN: {text!r}  [{en_ipa(text)}]")
        for dual, snd, coh, fr, nw, nf in decode_poetry(text, root)[:5]:
            tag = f"  <-- {nf} filler(s), {nw} words" if nf else ""
            print(f"   dual {dual:.2f} (snd {snd:.2f} coh {coh:.2f})  {fr}{tag}")
        print()

    print("""Reading: admitting the 1-segment fillers (un, et, on, a, aux, y...) lets
the stream carve into the short-word van Rooten style instead of one long noun.
But raw sound still rates a single tight word (épidémie) near the multi-word
carve, so SENSE has to break the tie -- which is why the coherence model (not the
dictionary) is the gating component. The fillers are necessary scaffolding; a
real L2 model choosing among filler-rich carves is what turns scaffolding into
verse.""")


if __name__ == "__main__":
    main()
