"""Why v6 finds fewer/worse word-for-word matches than v5 even with looser gates.

Hypothesis: v5 (build_dictionary.py) is RETRIEVAL -- block the whole French
lexicon by bigram Dice, rank the shortlist with combo, pick the best French WORD.
v6 (build_v6.py) is the DECODER -- beam-search the phoneme stream into a word
SEQUENCE. Retrieval scores the query against thousands of real candidates and
takes the global argmax; the decoder beam prunes and optimises its own path cost,
so for WORD-FOR-WORD it is the wrong tool -- it misses matches retrieval finds.

This runs both on the same English words and compares. If retrieval wins, the v6
build should use retrieval for word entries and reserve the decoder for line
carving.

Run: python retrieve_vs_decode.py
"""
from __future__ import annotations

from wordfreq import top_n_list

import matcher
from matcher import g2p, _segs, _canonical, _ngram_channel, _feat_channel
import phonetic_decoder as pd
import poetry_mode as pm

try:
    import bigram_lm
    LM = bigram_lm.load("fr")
except Exception:
    LM = None


def bigrams(ipa):
    s = ("#",) + _segs(_canonical(ipa)) + ("#",)
    return {s[i] + s[i + 1] for i in range(len(s) - 1)}


def combo_ipa(qi, ci):
    return 0.5 * _ngram_channel(qi, ci) + 0.5 * _feat_channel(qi, ci)


def build_fr_index(n_fr=4000):
    fr = [w for w in top_n_list("fr", n_fr) if w.isalpha() and len(w) > 1]
    out = []
    for w in fr:
        ipa = g2p(w, "fr")
        out.append((w, ipa, bigrams(ipa)))
    return out


def retrieve(w, fr_index, shortlist=25):
    qi = g2p(w, "en")
    qb = bigrams(qi)
    if not qb:
        return 0.0, None
    short = sorted(fr_index, key=lambda c: (len(qb & c[2]) / (len(qb) + len(c[2])) if c[2] else 0),
                   reverse=True)[:shortlist]
    best_s, best_w = 0.0, None
    for cw, ci, _ in short:
        if cw == w:
            continue
        s = combo_ipa(qi, ci)
        if s > best_s:
            best_s, best_w = s, cw
    return best_s, best_w


def decode_best(w, root):
    qi = g2p(w, "en")
    cands = pd.decode(qi, root, top_n=10, max_words=3,
                      lm=LM, lm_weight=pd.LM_BEAM_WEIGHT if LM else 0.0)
    best_s, best_fr = 0.0, None
    for c in cands:
        if c["coverage"] < 0.6:
            continue
        s = matcher.homophone_score(w, "en", c["fr"], "fr")["score"]
        if s > best_s:
            best_s, best_fr = s, c["fr"]
    return best_s, best_fr


def main():
    print("Building FR retrieval index...", flush=True)
    fr_index = build_fr_index(4000)
    pd.MIN_WORD_SEGS = 1
    pd.WORD_PENALTY = 0.04
    pd.BEAM = 300
    root = pm.build_poetry_trie(min_zipf=2.0)

    words = [w for w in top_n_list("en", 300) if w.isalpha() and 3 <= len(w) <= 8][:50]
    rwin = dwin = tie = 0
    rsum = dsum = 0.0
    print(f"\n{'word':10s} {'retrieve (v5)':22s} {'decode (v6)':22s}")
    print("-" * 56)
    shown = 0
    for w in words:
        rs, rw = retrieve(w, fr_index)
        ds, dw = decode_best(w, root)
        rsum += rs; dsum += ds
        if rs > ds + 0.02:
            rwin += 1
        elif ds > rs + 0.02:
            dwin += 1
        else:
            tie += 1
        if shown < 22:
            print(f"{w:10s} {(rw or '-')+f' ({rs:.2f})':22s} {(dw or '-')+f' ({ds:.2f})':22s}")
            shown += 1
    print(f"\nover {len(words)} words:  retrieve mean combo {rsum/len(words):.3f}  "
          f"decode mean combo {dsum/len(words):.3f}")
    print(f"wins: retrieve {rwin}, decode {dwin}, tie {tie}")
    print("""
Reading: if retrieve wins, v6's lower yield is explained -- it used the DECODER
(a line-carving segmenter) to build a WORD dictionary, where v5's RETRIEVAL
(block + rank over the whole lexicon, global argmax) is the right tool. The fix:
v6 should retrieve word entries the v5 way and reserve the decoder for line
carving. Looser gates can't help when the candidate generator itself prunes away
the best word.""")


if __name__ == "__main__":
    main()
