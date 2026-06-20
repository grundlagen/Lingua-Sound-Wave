"""Fragment weave: grow a shared phoneme stream from sound-blocks, then read it
off as real words in BOTH languages at once.  Unbounded length, recursive,
novelty-biased.

The idea you asked for
----------------------
`fragments.tsv` is an index of EN<->FR sound-blocks that already proved
themselves inside v5 alignments ("st"~"st" x372, "tr"~"tr" x183, "ri"~"ri"
x244).  Most high-count blocks are sound-IDENTICAL across the two languages.
So if we *chain* blocks into one long IPA stream, that stream is simultaneously
pronounceable as English and as French.  Decode it through the English word
trie -> a real English phrase; decode the same stream through the French word
trie -> a real French phrase that sounds the same.  Chain as long as you like:
the translation is not length-limited, it is grown.

  fragments  ->  shared IPA stream  ->  decode(EN trie) = EN phrase
                                    ->  decode(FR trie) = FR phrase
  keep when BOTH phrases are fluent (wordfreq gate) AND the pair is NOVEL.

Recursion (`--rounds N`): pairs that pass become *new mega-fragments* (their
whole IPA stream is added to the block bank), so the next round can chain them
into longer, still-fluent, still-novel compositions.  Certify -> densify ->
re-grow, applied to generation, not just to recycling existing edges.

Novelty bias (you asked to keep more of it novel)
-------------------------------------------------
A candidate is rewarded for being absent from dictionary-v5 (neither the EN
phrase nor the FR phrase is a known headword), for being non-cognate (EN and FR
word-sets disjoint), and for distinct new words; trivial/known/cognate pairs are
pushed down or dropped.

Needs espeak-ng + panphon + numpy + wordfreq + cmudict and the data/ lexica.
Run from research/homophone-bench/.

Usage:
  python fragment_weave.py                       # grow novel EN/FR pairs
  python fragment_weave.py --rounds 3 --certify  # recursive growth + seed bank
"""
from __future__ import annotations

import random
import sys
import time
from functools import lru_cache

from wordfreq import zipf_frequency

import phonetic_decoder as pd
from matcher import _canonical, _segs

# ---- knobs ----
FRAG_TOPK = 160        # how many of the most-attested blocks to chain from
MIN_LEN = 3            # minimum blocks per chain
MAX_LEN = 9            # grows each round; not a hard ceiling on the OUTPUT words
CHAINS_PER_LEN = 260   # breadth of the random/greedy chain sampler
WORD_ZIPF_GATE = 3.0   # every output word must be at least this common
DECODE_BEAM = 140      # smaller beam => faster generation (vs default 350)
KEEP = 40


@lru_cache(maxsize=200000)
def _zipf(w: str, lang: str) -> float:
    return zipf_frequency(w, lang)


# Optional word-bigram LM (bigram_lm.py).  When loaded, fluency() scores
# grammatical adjacency instead of mean unigram frequency, so "in the sea"
# beats "set could" even though both are made of common words.
_LM: dict = {}


def load_lm():
    import bigram_lm
    for lang in ("en", "fr"):
        try:
            _LM[lang] = bigram_lm.load(lang)
        except FileNotFoundError:
            print(f"no bigram-lm-{lang}.pkl; run `python bigram_lm.py /tmp/corpus`",
                  file=sys.stderr)


def fluency(words, lang) -> float:
    if not words:
        return 0.0
    if lang in _LM:
        return _LM[lang].fluency(list(words))
    return sum(min(_zipf(w, lang), 6.0) / 6.0 for w in words) / len(words)


def min_zipf(words, lang) -> float:
    return min((_zipf(w, lang) for w in words), default=0.0)


def load_blocks(path="fragments.tsv"):
    """Return [(ipa_block, count)] for sound-shared blocks (en_chunk == fr_chunk
    in canonical IPA -- the ones that are true cross-lingual sound units)."""
    blocks = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            count, en_chunk, fr_chunk = parts[0], parts[1], parts[2]
            if _canonical(en_chunk) == _canonical(fr_chunk):
                blocks.append((en_chunk, int(count)))
    blocks.sort(key=lambda b: -b[1])
    return blocks[:FRAG_TOPK]


def known_sets(path="dictionary-v5.json"):
    import json
    d = json.load(open(path))
    en = {e["en"] for e in d}
    fr = {e["fr"] for e in d}
    return en, fr


def novelty(en_phrase, fr_phrase, known_en, known_fr) -> float:
    ew, fw = en_phrase.split(), fr_phrase.split()
    if en_phrase == fr_phrase:
        return 0.0
    score = 1.0
    if en_phrase in known_en:           # already a dictionary headword
        score -= 0.5
    if fr_phrase in known_fr:
        score -= 0.3
    if set(ew) & set(fw):               # cognate-ish overlap
        score -= 0.3
    score += 0.06 * len(set(ew) | set(fw))   # reward distinct new words
    return max(0.0, min(1.5, score))


def best_decode(ipa_stream, root, lang, max_words):
    cands = pd.decode(ipa_stream, root, top_n=6, max_words=max_words)
    for c in cands:
        words = c["fr"].split()
        if c["expensive_deletions"] == 0 and min_zipf(words, lang) >= WORD_ZIPF_GATE:
            return c, words
    return None, None


def grow(blocks, en_root, fr_root, known_en, known_fr, *,
         max_len, deadline, mega=()):
    """Sample fragment chains of length MIN_LEN..max_len; for each, decode the
    shared stream into EN and FR words; keep fluent+novel pairs."""
    rng = random.Random(7)
    weights = [c for _, c in blocks]
    pool = [b for b, _ in blocks] + list(mega)
    wpool = weights + [max(weights)] * len(mega)   # mega-fragments are attractive
    results, seen = [], set()
    for L in range(MIN_LEN, max_len + 1):
        for _ in range(CHAINS_PER_LEN):
            if time.time() > deadline:
                break
            chain = rng.choices(pool, weights=wpool, k=L)
            ipa = "".join(chain)
            segs = _segs(_canonical(ipa))
            if not (4 <= len(segs) <= 22):
                continue
            max_words = max(2, len(segs))      # NOT length-limited: words follow stream
            en_c, ew = best_decode(ipa, en_root, "en", max_words)
            if not en_c:
                continue
            fr_c, fw = best_decode(ipa, fr_root, "fr", max_words)
            if not fr_c:
                continue
            en_p, fr_p = en_c["fr"], fr_c["fr"]   # decode() labels its output "fr"
            key = (en_p, fr_p)
            if key in seen:
                continue
            seen.add(key)
            nov = novelty(en_p, fr_p, known_en, known_fr)
            if nov <= 0.0:
                continue
            sound = 0.5 * (en_c["similarity"] + fr_c["similarity"])
            en_fl, fr_fl = fluency(ew, "en"), fluency(fw, "fr")
            joint = sound * en_fl * fr_fl * (0.6 + 0.4 * nov)
            results.append({
                "en": en_p, "fr": fr_p, "ipa": ipa, "blocks": L,
                "sound": round(sound, 3), "en_flu": round(en_fl, 3),
                "fr_flu": round(fr_fl, 3), "novelty": round(nov, 2),
                "joint": round(joint, 3), "words": (len(ew), len(fw)),
            })
    results.sort(key=lambda r: -r["joint"])
    return results


def main():
    rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 1
    certify = "--certify" in sys.argv
    budget = int(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 240

    if "--lm" in sys.argv:
        load_lm()
        print(f"bigram LM loaded for {sorted(_LM)}", file=sys.stderr)
    print("loading blocks + building common-word tries (min_zipf=3.0)...",
          file=sys.stderr)
    pd.BEAM = DECODE_BEAM
    blocks = load_blocks()
    known_en, known_fr = known_sets()
    en_root = pd.build_trie(min_zipf=3.0, lang="en")
    fr_root = pd.build_trie(min_zipf=3.0, lang="fr")

    mega, max_len = [], MAX_LEN
    all_gold = []
    for rd in range(1, rounds + 1):
        deadline = time.time() + budget
        res = grow(blocks, en_root, fr_root, known_en, known_fr,
                   max_len=max_len, deadline=deadline, mega=tuple(mega))
        print(f"\n===== round {rd}: {len(res)} novel fluent pairs "
              f"(max_len={max_len}) =====")
        for r in res[:KEEP]:
            print(f"  EN: {r['en']:34s} | FR: {r['fr']:30s} "
                  f"snd {r['sound']:.2f} enflu {r['en_flu']:.2f} "
                  f"frflu {r['fr_flu']:.2f} nov {r['novelty']:.2f} "
                  f"joint {r['joint']:.2f}")
        # recursion: promote the strongest streams to mega-fragments + grow longer
        strong = [r for r in res if r["sound"] >= 0.9 and r["en_flu"] >= 0.55
                  and r["fr_flu"] >= 0.55][:12]
        mega += [r["ipa"] for r in strong]
        all_gold += strong
        max_len += 3   # next round may chain longer, including the mega-fragments
        if not strong:
            break

    if certify and all_gold:
        with open("certified-phrase-pairs.tsv", "a", encoding="utf-8") as f:
            for r in all_gold:
                f.write(f"{r['en']}\t{r['fr']}\ten\tfr\t{r['sound']}"
                        f"\t{r['en_flu']}\t{r['fr_flu']}\t{r['novelty']}\n")
        print(f"\ncertified {len(all_gold)} novel pairs -> certified-phrase-pairs.tsv",
              file=sys.stderr)


if __name__ == "__main__":
    main()
