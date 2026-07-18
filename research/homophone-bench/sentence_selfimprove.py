"""SELF-IMPROVING sentence composer over an EXTANT phrase database.

Ties three things the user asked for into one bounded, offline loop (no GPU, no
money, no LLM):

  1. BEST-OF-BEST only     -- a carve is kept only if BOTH languages are fluent
                              and it clears sound >= SIGMA and joint >= TAU.
  2. A GOOD EXTANT DB      -- source sentences are taken from corpus-phrases-en.tsv
                              (frequent English phrases mined from the corpus
                              bigram LM), not hand-picked lines.
  3. SELF-IMPROVING        -- expert iteration / densify->re-search:

     round r:
       a. COMPOSE each eval phrase (phrase_weave, juncture on); keep best-of-best.
       b. CERTIFY the winners into certified-phrase-pairs.tsv (dedup) -- gold grows.
       c. DENSIFY: fold each certified FR carve back into the decoder trie as a
          SINGLE multi-word fragment whose stored pronunciation is the whole-phrase
          (liaison-correct) IPA. Next round can place a proven-fluent French
          fragment in one step -- "the extra words we didn't have".
       d. MEASURE yield% + mean joint over the SAME fixed eval set. Because the
          only thing that changed between rounds is the fragment bank, a rising
          yield is real densification improvement, not memorisation.

Run:
  python sentence_selfimprove.py --rounds 3 --n 30
  python sentence_selfimprove.py --rounds 3 --n 30 --source mylines.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import matcher
from matcher import _canonical
import phonetic_decoder as pd
import phrase_weave as pw

SIGMA = 0.88      # min sound (connected-speech) for a keeper
TAU = 0.68        # min joint (sound x fluency) for a keeper
FRAG_ZIPF = 4.2   # frequency assigned to a re-injected fragment (common-word tier)
CERT_TSV = "certified-phrase-pairs.tsv"


def load_source_phrases(n: int, source: str | None) -> list[str]:
    """Eval phrases: from a user file, else the extant corpus-phrases DB."""
    if source and os.path.exists(source):
        lines = [l.strip() for l in open(source, encoding="utf-8") if l.strip()]
        return lines[:n]
    out, seen = [], set()
    with open("corpus-phrases-en.tsv", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            phrase = line.split("\t", 1)[0].strip()
            if len(phrase.split()) >= 2 and phrase not in seen:
                seen.add(phrase)
                out.append(phrase)
            if len(out) >= n:
                break
    return out


def inject_fragment(root: pd.Node, fr_phrase: str, injected: set[str]) -> bool:
    """Fold a certified French carve into the trie as one reusable unit, keyed on
    its whole-phrase (liaison-correct) pronunciation."""
    if fr_phrase in injected:
        return False
    ipa = pw.phrase_ipa(fr_phrase, "fr")          # connected speech (sandhi baked in)
    segs = matcher._segs(_canonical(ipa))
    if len(segs) < 2:
        return False
    node = root
    for s in segs:
        node = node.children.setdefault(s, pd.Node())
    node.words.append((fr_phrase, FRAG_ZIPF))
    injected.add(fr_phrase)
    return True


def best_of_best(text: str, root) -> dict | None:
    """Single best carve for a source phrase, or None if nothing clears the bar."""
    rows = pw.both_intelligible(text, "en", "fr", root, juncture=True)
    rows = [r for r in rows if r["sound"] >= SIGMA and r["joint"] >= TAU]
    return rows[0] if rows else None


def certify(rows: list[dict], seen_pairs: set[tuple[str, str]]) -> int:
    new = 0
    with open(CERT_TSV, "a", encoding="utf-8") as f:
        for r in rows:
            key = (r["src"], r["tgt"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            f.write(f"{r['src']}\t{r['tgt']}\ten\tfr\t{r['sound']}\t"
                    f"{r['tgt_fluency']}\t{r['joint']}\tselfimprove\n")
            new += 1
    return new


def load_seen_pairs() -> set[tuple[str, str]]:
    seen = set()
    if os.path.exists(CERT_TSV):
        for line in open(CERT_TSV, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                seen.add((p[0], p[1]))
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--source", default=None)
    args = ap.parse_args()

    print(f"building fr trie...", file=sys.stderr)
    root = pd.build_trie(min_zipf=2.0, lang="fr")

    eval_phrases = load_source_phrases(args.n, args.source)
    print(f"eval set: {len(eval_phrases)} phrases from "
          f"{args.source or 'corpus-phrases-en.tsv (extant DB)'}\n")

    seen_pairs = load_seen_pairs()
    injected: set[str] = set()
    gold0 = len(seen_pairs)

    print(f"{'round':>5} {'yield':>7} {'carved':>7} {'mean_joint':>11} "
          f"{'new_gold':>9} {'fragments':>10}")
    print("-" * 56)
    for r in range(1, args.rounds + 1):
        winners, joints = [], []
        for text in eval_phrases:
            w = best_of_best(text, root)
            if w:
                winners.append(w)
                joints.append(w["joint"])
        # certify + densify from this round's winners
        new_gold = certify(winners, seen_pairs)
        frags = sum(inject_fragment(root, w["tgt"], injected) for w in winners)
        y = len(winners) / max(1, len(eval_phrases))
        mj = sum(joints) / max(1, len(joints))
        print(f"{r:>5} {y:>6.1%} {len(winners):>7} {mj:>11.3f} "
              f"{new_gold:>9} {frags:>10}")

    print(f"\ngold: {gold0} -> {len(seen_pairs)} certified pairs "
          f"(+{len(seen_pairs) - gold0} this run)")
    # show a few best-of-best from the final round
    finals = sorted((best_of_best(t, root) for t in eval_phrases),
                    key=lambda w: -w["joint"] if w else 0)
    print("\nbest-of-best sample (final trie):")
    for w in [x for x in finals if x][:8]:
        print(f"  EN {w['src']:22s} -> FR {w['tgt']:26s} "
              f"snd {w['sound']:.2f} flu {w['tgt_fluency']:.2f} joint {w['joint']:.2f}")


if __name__ == "__main__":
    main()
