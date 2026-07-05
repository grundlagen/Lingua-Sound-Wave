"""Full-scale EN->FR homophone dictionary builder.

Same architecture as build_dictionary.py (block with phoneme bigrams, rank
with the benchmark-winning `combo` matcher) but engineered for full lexicons:

  - batched G2P: ONE espeak-ng process per chunk instead of one per word
    (words are terminated with "." so each is its own utterance — this
    blocks French liaison from corrupting isolated-word pronunciations);
  - inverted bigram index for blocking (no all-pairs scan);
  - unbounded feature caches (the 8k-entry default thrashes at this scale);
  - keeps the top 3 candidates per English word with score >= 0.62, so the
    B-tier ("loose, needs a human") is preserved for later use.

Run: python build_dictionary_full.py [n_en=8000] [n_fr=15000]
Outputs: dictionary-full.json, dictionary-full.tsv, tier counts on stdout.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache

from wordfreq import top_n_list

import matcher
from matcher import _canonical, _variants, _ngram_channel

# Unbounded caches: ~25k unique IPA strings plus variants exceed the defaults.
matcher._vecs = lru_cache(maxsize=None)(matcher._vecs.__wrapped__)
matcher._segs = lru_cache(maxsize=None)(matcher._segs.__wrapped__)

S, A, B = 0.90, 0.78, 0.62
SHORTLIST = 15
KEEP_PER_WORD = 3
CHUNK = 800

EXTRA_DROP = {"-", "_", "ˈ", "ˌ", "‿", ".", "|", "‖", " ", "\t"}


def clean(ipa: str) -> str:
    s = unicodedata.normalize("NFD", ipa)
    return "".join(ch for ch in s if ch not in EXTRA_DROP)


def batch_g2p(words: list[str], voice: str) -> dict[str, str]:
    """One espeak call per chunk; '.' per word keeps utterances separate."""
    out: dict[str, str] = {}
    for i in range(0, len(words), CHUNK):
        chunk = words[i:i + CHUNK]
        text = "".join(w + ".\n" for w in chunk)
        r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, "--stdin"],
                           input=text, capture_output=True, text=True, check=True)
        lines = [ln for ln in r.stdout.split("\n") if ln.strip()]
        if len(lines) == len(chunk):
            for w, ln in zip(chunk, lines):
                out[w] = clean(ln)
        else:  # rare misalignment: fall back to per-word for this chunk
            for w in chunk:
                r1 = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, w],
                                    capture_output=True, text=True, check=True)
                out[w] = clean(r1.stdout.strip())
        print(f"  g2p {voice}: {min(i + CHUNK, len(words))}/{len(words)}",
              file=sys.stderr)
    return out


def bigrams(ipa: str) -> frozenset:
    s = ("#",) + matcher._segs(_canonical(ipa)) + ("#",)
    return frozenset(s[i] + s[i + 1] for i in range(len(s) - 1))


@lru_cache(maxsize=None)
def feat_channel_cached(ipa_a: str, ipa_b: str) -> float:
    va = _variants(ipa_a)
    vb = _variants(ipa_b)
    return max(matcher._nw_sim(matcher._vecs(a), matcher._vecs(b)) for a in va for b in vb)


def combo(qi: str, ci: str) -> float:
    return 0.5 * _ngram_channel(qi, ci) + 0.5 * feat_channel_cached(qi, ci)


def load_dict_words(path: str) -> set[str]:
    """Lowercase non-proper entries from a system dictionary file."""
    words = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            # skip proper nouns (capitalized) and possessives/abbreviations
            if w and w == w.lower() and w.isalpha():
                words.add(w)
    return words


def main():
    t0 = time.time()
    n_en = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    n_fr = int(sys.argv[2]) if len(sys.argv) > 2 else 15000

    # Legitimacy filter: frequency lists are full of fragments, proper nouns
    # and code-switched foreign words deep down ("tou", "wii", "dwight").
    # Requiring membership in a real dictionary keeps every entry a
    # legitimate lexical unit — the "no un œuf" guarantee, word-level.
    en_dict = load_dict_words("/usr/share/dict/american-english")
    fr_dict = load_dict_words("/usr/share/dict/french")

    en = [w for w in top_n_list("en", n_en)
          if w.isalpha() and len(w) > 1 and w in en_dict]
    fr = [w for w in top_n_list("fr", n_fr)
          if w.isalpha() and len(w) > 1 and w in fr_dict]
    print(f"lexicons after dictionary filter: {len(en)} EN x {len(fr)} FR",
          file=sys.stderr)

    en_ipa = batch_g2p(en, "en-us")
    fr_ipa = batch_g2p(fr, "fr")

    fr_bg = [bigrams(fr_ipa[w]) for w in fr]
    index: dict[str, list[int]] = defaultdict(list)
    for j, bg in enumerate(fr_bg):
        for b in bg:
            index[b].append(j)

    entries = []
    for k, w in enumerate(en):
        qi = en_ipa[w]
        qb = bigrams(qi)
        if not qb:
            continue
        counts: Counter = Counter()
        for b in qb:
            counts.update(index[b])
        cands = sorted(
            ((2 * c / (len(qb) + len(fr_bg[j])), j) for j, c in counts.items()),
            reverse=True)[:SHORTLIST]
        scored = []
        for _dice, j in cands:
            cw = fr[j]
            if cw == w:
                continue  # identical spelling: loanword, trivial
            s = combo(qi, fr_ipa[cw])
            if s >= B:
                scored.append((s, cw))
        scored.sort(reverse=True)
        for rank, (s, cw) in enumerate(scored[:KEEP_PER_WORD]):
            tier = "S" if s >= S else "A" if s >= A else "B"
            entries.append({
                "en": w, "fr": cw, "score": round(s, 3), "tier": tier,
                "rank": rank, "en_ipa": qi, "fr_ipa": fr_ipa[cw],
                "en_freq_rank": k, "fr_freq_rank": fr.index(cw) if rank == 0 else None,
            })
        if (k + 1) % 500 == 0:
            print(f"  ranked {k + 1}/{len(en)} ({time.time() - t0:.0f}s)",
                  file=sys.stderr)

    entries.sort(key=lambda e: (-e["score"], e["en_freq_rank"]))
    counts = Counter(e["tier"] for e in entries)
    best = Counter(e["tier"] for e in entries if e["rank"] == 0)
    print(f"\n=== Full dictionary: {len(entries)} entries from {len(en)} EN words ===")
    print(f"  all entries:  S={counts['S']}  A={counts['A']}  B={counts['B']}")
    print(f"  best-per-word: S={best['S']}  A={best['A']}  B={best['B']}")
    print(f"  EN words with a strong (S/A) match: "
          f"{len({e['en'] for e in entries if e['tier'] in 'SA'})}")
    print(f"  took {time.time() - t0:.0f}s")

    with open("dictionary-full.json", "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=0)
    with open("dictionary-full.tsv", "w") as f:
        f.write("tier\tscore\ten\tfr\ten_ipa\tfr_ipa\n")
        for e in entries:
            f.write(f"{e['tier']}\t{e['score']}\t{e['en']}\t{e['fr']}"
                    f"\t{e['en_ipa']}\t{e['fr_ipa']}\n")
    print("wrote dictionary-full.json / dictionary-full.tsv")


if __name__ == "__main__":
    main()
