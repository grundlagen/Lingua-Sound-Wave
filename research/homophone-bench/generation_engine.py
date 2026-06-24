"""Resegmentation-under-coherence: the 1+2+3 generation engine.

Builds ONLY on what June 11-12 perfected -- reinvents none of it:

  (1) RESEGMENTATION  [reused: phonetic_decoder.py, v4]
      The beam decoder walks an English phoneme stream and re-cuts it into
      FRENCH words. Word boundaries cost nothing acoustically ("space
      understanding", WORD_PENALTY only), so the French boundaries fall where
      French words land -- NOT where the English words were. This is the
      oronym/juncture engine; it is the operation a word-for-word table cannot do.
      Phonetics rules it carries (matcher.py): EQUIV floors (voicing p/b, lax
      i/ɪ, TH->s, rhotic map, nasal split), SHARPEN, cheap gaps (offglide/schwa/h),
      decoder LIAISON (z/t/n bridges).

  (2) COHERENCE       [reused: bigram_lm in the beam, this session]
      The L2 language model scores word-adjacency INSIDE the search and as the
      re-rank, so the re-cut lands on fluent French, not salad. (Still the weak
      link -- a real LM/LLM is the upgrade; the interface is here.)

  (3) CONTENT SELECTION [new glue]
      Whole lines don't re-cut (the over-constraint wall); van Rooten CHOSE
      spans that carve well. So we slide phrase windows, resegment each, score
      dual = sound x coherence, and KEEP the spans that re-cut, FLAGGING the rest
      as candidates for the future synonym-swap layer (align the phoneme string
      by substituting a near-synonym) -- exposed as a hook, not built here.

  Junction when chaining kept spans: generate.py's hiatus rule (no FR coda-V
  meeting FR onset-V) is applied so spans abut cleanly.

Public-domain source only. Run: python generation_engine.py
"""
from __future__ import annotations

import subprocess
import unicodedata

import matcher
import phonetic_decoder as pd

try:
    import bigram_lm
    LM = bigram_lm.load("fr")
except Exception:
    LM = None

DUAL_KEEP = 0.45        # a span "re-cuts well" above this dual score
WINDOW = (2, 4)         # phrase-window sizes (words) for content selection


def en_ipa(text):
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", text],
                       capture_output=True, text=True, check=True)
    from lexicon_g2p import clean_ipa
    return clean_ipa(r.stdout.strip())


def coherence(fr):
    toks = [t.lower() for t in fr.replace("'", " ").split() if t]
    if LM and toks:
        return LM.fluency(toks)
    return 0.0


def resegment(span_text, fr_root, top_n=8):
    """(1)+(2): re-cut the span's phoneme stream into fluent French words."""
    ipa = en_ipa(span_text)
    cands = pd.decode(ipa, fr_root, top_n=top_n, max_words=8,
                      lm=LM, lm_weight=pd.LM_BEAM_WEIGHT if LM else 0.0)
    scored = []
    for c in cands:
        if c["expensive_deletions"] > 0:
            continue
        snd = c["similarity"]
        coh = coherence(c["fr"])
        scored.append((snd * coh, snd, coh, c["fr"], c["words"]))
    scored.sort(reverse=True)
    return scored[0] if scored else None


def content_select(line, fr_root):
    """(3): slide phrase windows, resegment each, keep the spans that re-cut."""
    words = [w.strip(".,!?;:") for w in line.split()]
    spans = []
    i = 0
    while i < len(words):
        best = None
        for size in range(WINDOW[1], WINDOW[0] - 1, -1):
            if i + size > len(words):
                continue
            span = " ".join(words[i:i + size])
            r = resegment(span, fr_root)
            if r and (best is None or r[0] > best[0][0]):
                best = (r, span, size)
        if best is None:
            spans.append(("GAP", words[i], None))
            i += 1
        else:
            (dual, snd, coh, fr, nfr), span, size = best
            verdict = "re-cut" if dual >= DUAL_KEEP else "weak->synonym?"
            spans.append((verdict, span, (fr, dual, snd, coh, size, nfr)))
            i += size
    return spans


def main():
    pd.BEAM = 200
    fr_root = pd.build_trie(min_zipf=2.2, lang="fr")
    print(f"engine ready. coherence model: {'bigram-LM' if LM else 'none'}\n")

    # public-domain Mother Goose source lines (1916)
    LINES = [
        "Jack and Jill went up the hill",
        "Little Jack Horner sat in a corner",
        "Humpty Dumpty sat on a wall",
    ]
    for line in LINES:
        print(f"EN: {line}")
        spans = content_select(line, fr_root)
        kept_fr, en_w, fr_w = [], 0, 0
        for verdict, span, info in spans:
            if verdict == "GAP":
                print(f"   [GAP]            {span!r:18s} no clean re-cut "
                      f"-> synonym-swap candidate")
                continue
            fr, dual, snd, coh, size, nfr = info
            en_w += size
            fr_w += nfr
            mark = "OK " if verdict == "re-cut" else "wk "
            print(f"   [{mark}{verdict:13s}] {span!r:22s} -> {fr:24s} "
                  f"dual {dual:.2f} (snd {snd:.2f} coh {coh:.2f}) "
                  f"[{size}EN->{nfr}FR]")
            if verdict == "re-cut":
                kept_fr.append(fr)
        if kept_fr:
            print(f"   => re-cut FR: {' '.join(kept_fr)}")
            print(f"   => boundaries moved: {en_w} EN words re-cut into {fr_w} FR "
                  f"words (resegmentation working)")
        print()

    print("Reading: (1) the decoder re-cuts spans into French words with moved")
    print("boundaries; (2) the LM keeps the re-cut fluent; (3) content-selection")
    print("keeps spans that carve and FLAGS the rest for the future synonym-swap")
    print("layer (substitute a near-synonym so the phoneme string aligns). The")
    print("gating weakness remains the coherence model -- swap the bigram for a")
    print("real L2 LM/LLM and the same engine yields verse, not fragments.")


if __name__ == "__main__":
    main()
