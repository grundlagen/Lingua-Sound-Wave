"""Phonetic decoder: segment an English phoneme stream into French word
sequences — the Knight & Graehl (1997) transliteration idea applied to
homophone generation.

This generalizes the 2-word concatenation hack to true multi-to-multi:
  - a pronunciation TRIE over the full Lexique (241k French words) replaces
    pairwise blocking, so rare conjugations are finally reachable;
  - BEAM SEARCH walks the query phoneme stream left to right, consuming
    segments through the trie with the matcher's equivalence-floored
    substitution costs and cheap deletions (offglides/schwa/h);
  - "space understanding": word boundaries cost nothing acoustically but
    each word pays a small penalty (discourages confetti of tiny words) and
    earns a frequency bonus (common words make natural phrases) — the
    language-model prior from the transliteration literature;
  - "language dynamics": optional liaison consonant (z/t/n) between words
    when the previous word's spelling licenses it.

Usage:
  python phonetic_decoder.py "remember"            # decode one word/phrase
  python phonetic_decoder.py --augment             # upgrade dictionary v3 -> v4
"""
from __future__ import annotations

import heapq
import json
import math
import sys
from functools import lru_cache

from wordfreq import zipf_frequency

import matcher
from matcher import _canonical, _equiv_floor, _gap_cost, GAP
from lexicon_g2p import load_fr

matcher._vecs = lru_cache(maxsize=None)(matcher._vecs.__wrapped__)
matcher._segs = lru_cache(maxsize=None)(matcher._segs.__wrapped__)

# ---- scoring knobs (language dynamics) ----
WORD_PENALTY = 0.18      # per word break: prefer fewer, longer words
FREQ_BONUS = 0.05        # x zipf: prefer common words
MIN_WORD_SEGS = 2        # no single-phoneme word confetti
BEAM = 350
MAX_WORDS = 3
LIAISON = {"s": "z", "x": "z", "z": "z", "d": "t", "t": "t", "n": "n"}

# ---------------------------------------------------------------- trie

class Node:
    __slots__ = ("children", "words")

    def __init__(self):
        self.children: dict[str, Node] = {}
        self.words: list[tuple[str, float]] = []  # (word, zipf)


def build_trie(min_zipf: float = 0.0, lang: str = "fr") -> Node:
    root = Node()
    if lang == "fr":
        fr = load_fr()
    else:
        from lexicon_g2p import load_en
        fr = load_en()
    n = 0
    for w, prons in fr.items():
        if len(w) < 2:
            continue
        z = zipf_frequency(w, lang)
        if z < min_zipf:
            continue
        for p in prons:
            segs = matcher._segs(_canonical(p))
            if len(segs) < MIN_WORD_SEGS:
                continue
            node = root
            for s in segs:
                node = node.children.setdefault(s, Node())
            node.words.append((w, z))
            n += 1
    print(f"trie: {n} pronunciations indexed", file=sys.stderr)
    return root


STOPS = {"p", "b", "t", "d", "k", "\u0261", "g"}


@lru_cache(maxsize=200000)
def _sub(q: str, c: str) -> float:
    if q == c:
        return 0.0
    f = _equiv_floor(q, c)
    vq, vc = matcher._vecs(q), matcher._vecs(c)
    if len(vq) == 0 or len(vc) == 0:
        return min(f, 0.6)
    import numpy as np
    d = min(1.0, float(np.abs(vq[0] - vc[0]).sum()) / (2.0 * matcher.N_FEATURES) / matcher.SHARPEN)
    d = min(f, d)
    # panphon underprices place-of-articulation swaps between stops (t~g
    # scores ~0.24) but the ear does not: floor them unless an equivalence
    # (voicing pair) applies.
    if q in STOPS and c in STOPS and f > 0.3 and d < 0.35:
        d = 0.35
    return d


# ------------------------------------------------------------ beam search

# Weight of bigram-LM score in beam ranking (0 = disabled).
# The beam key uses a baseline-relative score so that word boundaries with
# above-average LM probability REDUCE the key (preferred) while below-average
# bigrams INCREASE it (penalized).  Neutral baseline per word: -6 nats.
LM_BEAM_WEIGHT = 0.35
_LM_BASELINE = -6.0   # per-word log-prob that is "average fluency" in the beam key


def decode(query_ipa: str, root: Node, top_n: int = 5, max_words: int = MAX_WORDS,
           lm=None, lm_weight: float = LM_BEAM_WEIGHT) -> list[dict]:
    """Decode a canonical-IPA query into word sequences.

    When `lm` is a BigramLM instance the beam is steered toward grammatically
    fluent word sequences (sound × fluency) rather than purely phonetic cost.
    Bigram log-probabilities are accumulated per word boundary.  The beam key
    rewards above-average-fluency bigrams (lower key = preferred) and penalizes
    below-average ones.  The final `ranked` score also incorporates LM fluency.
    """
    q = matcher._segs(_canonical(query_ipa))
    if not q:
        return []
    nq = len(q)
    # State: (cost, qpos, node, plen, words, wordlen, zipfsum, matched, baddel, maxsub, lm_logp)
    # lm_logp: running sum of bigram log-probs (0.0 when lm=None or no words yet)
    start = (0.0, 0, root, 0, (), 0, 0.0, 0, 0, 0.0, 0.0)
    beams: list[list] = [[] for _ in range(nq + 1)]
    beams[0] = [start]
    results: list[tuple] = []

    # Beam key: lower is better.
    # Only adjust key once at least one complete word has been emitted.
    # excess = lm_logp - baseline * n_words: positive = above average (reward = lower key)
    # negative = below average (penalize = higher key).  Zero-word states use pure cost
    # so the LM does not bias the beam toward single long words over multi-word splits.
    if lm and lm_weight > 0:
        def _key(s):
            nw = len(s[4])   # completed words only
            if nw == 0:
                return s[0]
            excess = s[10] - _LM_BASELINE * nw
            return s[0] - lm_weight * excess / nw
    else:
        def _key(s):
            return s[0]

    for qpos in range(nq + 1):
        layer = heapq.nsmallest(BEAM, beams[qpos], key=_key)
        for cost, _qp, node, plen, words, wlen, zsum, matched, baddel, maxsub, lm_logp in layer:
            # finish: all query consumed and at a word end
            if qpos == nq:
                if node.words and len(words) + 1 <= max_words:
                    for w, z in node.words[:4]:
                        seq = words + (w,)
                        sim = 1.0 - cost / max(1, plen)
                        if lm:
                            last_lp = math.log(max(
                                lm.cond(words[-1], w) if words else lm._p_uni(w), 1e-9))
                            total_lm = lm_logp + last_lp
                            # bigram_lm calibration: good phrase ~-3.4, salad ~-11 per word
                            lm_fl = max(0.0, min(1.0, (total_lm / len(seq) + 15.0) / 11.0))
                        else:
                            lm_fl = 0.0
                        adj = sim + FREQ_BONUS * ((zsum + z) / len(seq)) / 7.0 \
                              - WORD_PENALTY * (len(seq) - 1) / max(1, plen) \
                              + (lm_weight * 0.4 * lm_fl if lm else 0.0)
                        results.append((adj, seq, sim, matched / nq, baddel, maxsub))
                continue
            qseg = q[qpos]
            # skip query segment (deletion)
            gq = _gap_cost(qseg)
            beams[qpos + 1].append((cost + gq, qpos + 1, node, plen + 1,
                                    words, wlen, zsum, matched,
                                    baddel + (1 if gq > 0.2 else 0), maxsub, lm_logp))
            for cseg, child in node.children.items():
                sc = _sub(qseg, cseg)
                if sc < 0.85:  # match
                    beams[qpos + 1].append((cost + sc, qpos + 1, child, plen + 1,
                                            words, wlen + 1, zsum, matched + 1,
                                            baddel, max(maxsub, sc), lm_logp))
                # insertion of trie segment (stay on query pos)
                ic = _gap_cost(cseg)
                if ic < 0.2:
                    beams[qpos].append((cost + ic, qpos, child, plen + 1,
                                        words, wlen + 1, zsum, matched, baddel, maxsub,
                                        lm_logp))
            # word boundary: emit word, return to root
            if node.words and len(words) + 1 < max_words and wlen >= MIN_WORD_SEGS:
                for w, z in node.words[:4]:
                    nwords = words + (w,)
                    if lm:
                        new_lp = math.log(max(
                            lm.cond(words[-1], w) if words else lm._p_uni(w), 1e-9))
                        new_lm = lm_logp + new_lp
                    else:
                        new_lm = 0.0
                    beams[qpos].append((cost, qpos, root, plen, nwords, 0,
                                        zsum + z, matched, baddel, maxsub, new_lm))
                    # liaison: previous word's spelling licenses z/t/n bridge
                    lc = LIAISON.get(w[-1])
                    if lc and qpos < nq:
                        bridge = _sub(q[qpos], lc)
                        if bridge < 0.3:
                            beams[qpos + 1].append((cost + bridge, qpos + 1, root,
                                                    plen + 1, nwords, 0, zsum + z,
                                                    matched + 1, baddel,
                                                    max(maxsub, bridge), new_lm))
        # prune in place to keep memory bounded
        if qpos < nq:
            beams[qpos + 1] = heapq.nsmallest(BEAM * 2, beams[qpos + 1], key=_key)

    results.sort(key=lambda r: -r[0])
    out, seen = [], set()
    for adj, seq, sim, coverage, baddel, maxsub in results:
        key = " ".join(seq)
        if key in seen:
            continue
        seen.add(key)
        out.append({"fr": key, "similarity": round(sim, 3),
                    "ranked": round(adj, 3), "words": len(seq),
                    "coverage": round(coverage, 2), "expensive_deletions": baddel,
                    "max_substitution": round(maxsub, 2)})
        if len(out) >= top_n:
            break
    return out


# ------------------------------------------------------------ augmentation

def augment_dictionary():
    """Decode every EN word whose best v3 score < 0.90; merge wins into v4."""
    from lexicon_g2p import load_en
    entries = json.load(open("dictionary-v3.json"))
    best = {}
    for e in entries:
        best[e["en"]] = max(best.get(e["en"], 0.0), e["score"])
    lex_en = load_en()
    lex_fr = load_fr()
    root = build_trie(min_zipf=1.5)

    def phrase_ipa(phrase: str) -> str:
        return " ".join((lex_fr.get(w) or [""])[0] for w in phrase.split())

    added = 0
    new_entries = []
    targets = sorted(best)
    # also EN words in the dictionary lexicon that got NO entry at all
    for i, w in enumerate(targets):
        if best.get(w, 0.0) >= 0.90:
            continue
        prons = lex_en.get(w)
        if not prons:
            continue
        cands = decode(prons[0], root, top_n=3)
        for c in cands:
            if (c["similarity"] >= 0.90 and c["words"] >= 2
                    and c["coverage"] >= 0.85 and c["expensive_deletions"] == 0
                    and c["max_substitution"] <= 0.45):
                tier = ("S" if c["max_substitution"] <= 0.25 and c["coverage"] >= 0.9
                        else "A")
                new_entries.append({
                    "en": w, "fr": c["fr"], "score": c["similarity"],
                    "tier": tier, "rank": 0, "multiword": True,
                    "decoder": True, "cognate": False, "loanword": False,
                    "coverage": c["coverage"],
                    "max_substitution": c["max_substitution"],
                    "en_ipa": prons[0], "fr_ipa": phrase_ipa(c["fr"]),
                    "en_freq_rank": None,
                })
                added += 1
                break
        if (i + 1) % 500 == 0:
            print(f"  decoded {i + 1}/{len(targets)} (+{added})", file=sys.stderr)

    merged = entries + new_entries
    merged.sort(key=lambda e: -e["score"])
    with open("dictionary-v4.json", "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=0)
    with open("decoder-additions.tsv", "w") as f:
        f.write("tier\tscore\tcoverage\tmax_sub\ten\tfr_phrase\ten_ipa\n")
        for e in sorted(new_entries, key=lambda x: (x["tier"], -x["score"])):
            f.write(f"{e['tier']}\t{e['score']}\t{e['coverage']}\t{e['max_substitution']}"
                    f"\t{e['en']}\t{e['fr']}\t{e['en_ipa']}\n")
    print(f"\ndecoder added {added} multiword S entries -> dictionary-v4.json, "
          f"decoder-additions.tsv")


def reverse_augment():
    """Decode common FR words into EN word sequences (fr -> en direction)."""
    from wordfreq import top_n_list
    lex_fr = load_fr()
    from lexicon_g2p import load_en
    lex_en = load_en()
    entries = json.load(open("dictionary-v4.json"))
    covered = {e["fr"] for e in entries if e["score"] >= 0.90 and not e.get("multiword")}
    root = build_trie(min_zipf=1.5, lang="en")

    def en_phrase_ipa(phrase):
        return " ".join((lex_en.get(w) or [""])[0] for w in phrase.split())

    targets = [w for w in top_n_list("fr", 12000)
               if w.isalpha() and len(w) > 2 and w in lex_fr and w not in covered]
    new, added = [], 0
    for i, w in enumerate(targets):
        cands = decode(lex_fr[w][0], root, top_n=3)
        for c in cands:
            if (c["similarity"] >= 0.90 and c["words"] >= 2
                    and c["coverage"] >= 0.85 and c["expensive_deletions"] == 0
                    and c["max_substitution"] <= 0.45):
                tier = ("S" if c["max_substitution"] <= 0.25 and c["coverage"] >= 0.9
                        else "A")
                new.append({"en": c["fr"], "fr": w, "score": c["similarity"],
                            "tier": tier, "rank": 0, "multiword": True,
                            "decoder": True, "reverse": True, "cognate": False,
                            "loanword": False, "coverage": c["coverage"],
                            "max_substitution": c["max_substitution"],
                            "en_ipa": en_phrase_ipa(c["fr"]).replace(" ", ""),
                            "fr_ipa": lex_fr[w][0], "en_freq_rank": None})
                added += 1
                break
        if (i + 1) % 1000 == 0:
            print(f"  reverse {i + 1}/{len(targets)} (+{added})", file=sys.stderr)
    with open("reverse-additions.json", "w") as f:
        json.dump(new, f, ensure_ascii=False, indent=0)
    with open("reverse-additions.tsv", "w") as f:
        f.write("tier\tscore\tfr\ten_phrase\tfr_ipa\n")
        for e in sorted(new, key=lambda x: (x["tier"], -x["score"])):
            f.write(f"{e['tier']}\t{e['score']}\t{e['fr']}\t{e['en']}\t{e['fr_ipa']}\n")
    print(f"reverse decoder added {added} fr->en entries -> reverse-additions.json/tsv")


if __name__ == "__main__":
    if "--augment" in sys.argv:
        augment_dictionary()
    elif "--reverse" in sys.argv:
        reverse_augment()
    else:
        from lexicon_g2p import load_en
        lex_en = load_en()
        root = build_trie(min_zipf=1.5)
        for arg in sys.argv[1:]:
            ipa = lex_en.get(arg.lower(), [None])[0]
            if ipa is None:
                import subprocess
                r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", arg],
                                   capture_output=True, text=True)
                from lexicon_g2p import clean_ipa
                ipa = clean_ipa(r.stdout.strip())
            print(f"\n{arg!r} [{ipa}]:")
            for c in decode(ipa, root, top_n=6):
                print(f"  {c['similarity']:.3f}  {c['fr']}")
