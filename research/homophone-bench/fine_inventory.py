"""Fine / mixed generation inventory: single phonemes + all 2-4 phoneme chunks +
the full rule set (every phoneme incl. odd stops and the EN<->FR equivalence
phones), arbiter-ranked.

Motivated by SINGLE_PHONEME_COMPOSITION.md: single-phoneme composition trebled
dual-decodable yield (16% -> 50%) because the attested 2-phoneme chunks are
consonant-cluster-biased and French avoids clusters. But the chunks are ATTESTED
cross-lingual units (naturalness). So the right pool is MIXED:
  - every single phoneme (full reach, incl. stops p/b/t/d/k/ɡ and the EQUIV
    cross-lingual phones), weighted by corpus occurrence;
  - the 2-4 phoneme attested chunks (high-confidence anchors);
and let the matcher arbiter rank whatever decodes in both languages.

Run: python fine_inventory.py
"""
from __future__ import annotations

import collections
import random
import sys

from matcher import _canonical, _segs
import matcher
import phonetic_decoder as pd
import fragment_weave as fw


def build_inventories():
    chunks = fw.load_blocks()                      # 2-4 phoneme attested shared chunks
    # single-phoneme inventory: every phoneme seen in the chunks + EQUIV phones +
    # the stop set (the "odd stops, rules"), weighted by chunk occurrence.
    ph = collections.Counter()
    for b, c in chunks:
        for s in _segs(_canonical(b)):
            ph[s] += c
    for a, b in matcher.EQUIV:                      # cross-lingual equivalence phones
        ph[a] += 1
        ph[b] += 1
    for s in matcher_stops():                       # ensure all stops present
        ph[s] += 1
    singles = list(ph.items())
    mixed = singles + [(b, c) for b, c in chunks]   # the FINE pool: phones + chunks
    return singles, chunks, mixed


def matcher_stops():
    try:
        return pd.STOPS
    except Exception:
        return {"p", "b", "t", "d", "k", "ɡ", "g"}


def yield_and_top(pool, label, en_root, fr_root, known_en, known_fr,
                  k=200, L=6, seed=3):
    rng = random.Random(seed)
    items = [b for b, _ in pool]
    wts = [c for _, c in pool]
    both, tried, results, seen = 0, 0, [], set()
    for _ in range(k):
        chain = rng.choices(items, weights=wts, k=L)
        ipa = "".join(chain)
        segs = _segs(_canonical(ipa))
        if not (4 <= len(segs) <= 18):
            continue
        tried += 1
        en_c, ew = fw.best_decode(ipa, en_root, "en", max(2, len(segs)))
        fr_c, frw = fw.best_decode(ipa, fr_root, "fr", max(2, len(segs)))
        if not (en_c and fr_c):
            continue
        both += 1
        en_p, fr_p = en_c["fr"], fr_c["fr"]
        if (en_p, fr_p) in seen:
            continue
        seen.add((en_p, fr_p))
        combo = matcher.homophone_score(en_p, "en", fr_p, "fr")["score"]
        nov = fw.novelty(en_p, fr_p, known_en, known_fr)
        results.append((combo * (0.6 + 0.4 * nov), combo, en_p, fr_p))
    results.sort(reverse=True)
    print(f"\n[{label}] dual-decodable {both}/{tried} ({100*both//max(1,tried)}%)")
    for score, combo, en_p, fr_p in results[:6]:
        print(f"   combo {combo:.2f}  EN: {en_p:24s} | FR: {fr_p}")
    return both, tried


def main():
    pd.BEAM = fw.DECODE_BEAM
    singles, chunks, mixed = build_inventories()
    print(f"inventories: {len(singles)} single phonemes, {len(chunks)} chunks, "
          f"{len(mixed)} mixed (fine).")
    en_root = pd.build_trie(min_zipf=3.0, lang="en")
    fr_root = pd.build_trie(min_zipf=3.0, lang="fr")
    known_en, known_fr = fw.known_sets()

    yield_and_top(chunks, "chunks only (current)", en_root, fr_root, known_en, known_fr)
    yield_and_top(singles, "single phonemes", en_root, fr_root, known_en, known_fr)
    yield_and_top(mixed, "FINE: phones + chunks", en_root, fr_root, known_en, known_fr)

    print("""
Reading: the FINE pool (every phoneme incl. stops/equiv-rules + the attested
chunks) gives the generator both reach (single phonemes dodge the cluster bias)
and anchors (attested chunks for natural transitions); the matcher arbiter ranks
whatever decodes in both. This is the --fine inventory mode SINGLE_PHONEME_
COMPOSITION.md called for. Quality of the carries still rides on the L2-coherence
model (the gate).""")


if __name__ == "__main__":
    main()
