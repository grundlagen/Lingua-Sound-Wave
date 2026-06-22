"""Demonstration: end-to-end homophonic writing, sensical in BOTH languages.

This ties the whole stack together and shows progress toward the goal:

  fragments (attested EN<->FR sound-blocks)
     -> chain into a shared IPA stream
     -> LM-steered beam decode  =>  EN phrase  AND  FR phrase (same sound)
     -> ARBITER: re-score the produced pair with the AUC-0.993 matcher
        (matcher.homophone_score) -- the independent judge from RESULTS.md.

The arbiter is the honesty check: generation can fool its own decoder, but the
benchmark-winning combo scorer was tuned on hand-labeled data, never on the
generator's output, so a high combo score on a generated pair is real evidence
the two phrases actually sound alike.

Run from research/homophone-bench/.
"""
from __future__ import annotations

import time

import phonetic_decoder as pd
import fragment_weave as fw
import matcher


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def show_rules():
    banner("1. THE PHONETIC RULES (the AUC-0.993 arbiter)")
    print("""combo = 0.5*ngram_dice + 0.5*feat_nw_sharp  (RESULTS.md, AUC 0.993)
  ngram_dice    : Dice over exact phoneme bigrams  -- precision
  feat_nw_sharp : featural Needleman-Wunsch, panphon distance / 0.35 clamped,
                  floored by an EN<->FR equivalence table        -- recall

Equivalence floors that make cross-language homophones work:""")
    for pair, cost in [(("p", "b"), 0.20), (("t", "d"), 0.20), (("k", "ɡ"), 0.20),
                       (("i", "ɪ"), 0.10), (("e", "ɛ"), 0.10), (("θ", "s"), 0.25),
                       (("ŋ", "n"), 0.15), (("y", "i"), 0.20)]:
        f = matcher._equiv_floor(*pair)
        print(f"    {pair[0]} ~ {pair[1]}   sub-cost floor {f:.2f}")
    print("  rhotic: ʁ ʀ ɾ r -> ɹ      nasal split: ɑ̃ -> ɑn")
    print("  cheap to delete (offglide/schwa/h): ʊ ɪ j w ə ɚ h\n")
    print("Worked examples (text -> IPA -> combo):")
    for a, la, b, lb in [("shoe", "en", "chou", "fr"),
                         ("mayday", "en", "m'aider", "fr"),
                         ("the cat", "en", "des quatre", "fr"),
                         ("dog", "en", "chien", "fr")]:
        r = matcher.homophone_score(a, la, b, lb)
        verdict = "HOMOPHONE" if r["score"] >= 0.45 else "not alike"
        print(f"  {a!r:14s} ~ {b!r:14s} combo {r['score']:.2f} "
              f"(ng {r['ngram_dice']:.2f}/ft {r['featural_nw']:.2f}) "
              f"[{r['ipa_a']}|{r['ipa_b']}]  {verdict}")


def show_fragments():
    banner("2. THE FRAGMENTS (sound-blocks shared by EN and FR)")
    blocks = fw.load_blocks()
    print(f"{len(blocks)} sound-identical EN<->FR blocks loaded (top by attestation).")
    print("These are the same IPA in both languages, so a chain of them is")
    print("simultaneously pronounceable as English and as French:\n")
    for ipa, count in blocks[:12]:
        print(f"    {ipa:6s}  attested x{count}")


def produce(deadline_s=140):
    banner("3. PRODUCTION: generate bilingual homophone phrases (LM-steered)")
    fw.load_lm()
    pd.BEAM = fw.DECODE_BEAM
    phrase_seeds = fw.load_phrase_seeds()
    blocks = fw.load_blocks()
    known_en, known_fr = fw.known_sets()
    en_root = pd.build_trie(min_zipf=3.0, lang="en")
    fr_root = pd.build_trie(min_zipf=3.0, lang="fr")

    res = fw.grow(blocks, en_root, fr_root, known_en, known_fr,
                  max_len=fw.MAX_LEN, deadline=time.time() + deadline_s,
                  phrase_seeds=phrase_seeds)
    return res


def arbitrate(res, top=25):
    banner("4. ARBITER: re-score every produced pair with the AUC-0.993 matcher")
    print("combo >= 0.45 is the benchmark homophone threshold. The matcher never")
    print("saw the generator -- a high score here is independent evidence.\n")
    judged = []
    for r in res:
        a = matcher.homophone_score(r["en"], "en", r["fr"], "fr")
        r["combo"] = a["score"]
        r["arb_ng"] = a["ngram_dice"]
        r["arb_ft"] = a["featural_nw"]
        judged.append(r)
    # rank by a blend the user cares about: arbiter sound x both-side fluency
    judged.sort(key=lambda r: -(r["combo"] * r["en_flu"] * r["fr_flu"]))

    passed = [r for r in judged if r["combo"] >= 0.45]
    print(f"{len(passed)}/{len(judged)} produced pairs pass the arbiter (combo>=0.45).\n")
    print(f"{'EN phrase':24s} {'FR phrase (same sound)':24s} {'combo':>6s} "
          f"{'enflu':>6s} {'frflu':>6s}")
    print("-" * 72)
    for r in judged[:top]:
        flag = "  <-- both-fluent + sound-true" if (
            r["combo"] >= 0.6 and r["en_flu"] >= 0.7 and r["fr_flu"] >= 0.7) else ""
        print(f"{r['en']:24s} {r['fr']:24s} {r['combo']:6.2f} "
              f"{r['en_flu']:6.2f} {r['fr_flu']:6.2f}{flag}")
    return judged


def main():
    show_rules()
    show_fragments()
    res = produce()
    judged = arbitrate(res)

    banner("5. THE GOAL, MEASURED")
    sensical = [r for r in judged
                if r["combo"] >= 0.55 and r["en_flu"] >= 0.65 and r["fr_flu"] >= 0.65]
    print(f"Pairs that are sound-true (combo>=0.55) AND read as real phrases in")
    print(f"BOTH languages (en_flu & fr_flu >= 0.65): {len(sensical)}\n")
    for r in sensical[:12]:
        print(f"  EN  {r['en']}")
        print(f"  FR  {r['fr']}   (combo {r['combo']:.2f}, "
              f"enflu {r['en_flu']:.2f}, frflu {r['fr_flu']:.2f})")
        print()


if __name__ == "__main__":
    main()
