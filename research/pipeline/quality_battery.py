"""Quality battery: compose a varied sentence set, report generation quality.

One composer instance (inventory loaded once), N sentences; per sentence the
best line, its whole-line combo, and the provenance mix; then the aggregate:
combo distribution, source-tier usage, gap rate. The honest dashboard for
"how good is generation right now".

Run: python quality_battery.py --bench-dir ../homophone-bench [--sentences file]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from compose_lattice import SentenceComposer

DEFAULT_SENTENCES = [
    "she wandered through the deep forest at twilight",
    "the ocean remembers every vessel that ever sailed",
    "a gentle stream becomes a mighty rushing waterfall",
    "my heart is an open door",
    "the cat sat on the mat",
    "little lamb who made thee",
    "the moon was a ghostly galleon tossed upon cloudy seas",
    "and miles to go before i sleep",
    "one for sorrow two for joy",
    "the old man and the sea",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("."))
    ap.add_argument("--byt5-checkpoint", type=Path, default=None)
    ap.add_argument("--sentences", type=Path, default=None,
                    help="file of EN sentences, one per line")
    args = ap.parse_args()

    sentences = (args.sentences.read_text(encoding="utf-8").splitlines()
                 if args.sentences else DEFAULT_SENTENCES)
    sentences = [s.strip() for s in sentences if s.strip()]

    composer = SentenceComposer(args.bench_dir, args.pipeline_dir,
                                byt5_checkpoint=args.byt5_checkpoint)

    combos, sources = [], Counter()
    word_scores = []
    print()
    for sent in sentences:
        tokens, finals = composer.compose(sent, n_out=1)
        if not finals:
            print(f"!! no line for: {sent}")
            continue
        score, fr_line, path, r = finals[0]
        combos.append(score)
        for c in path:
            sources[c.source] += 1
            if c.source != "GAP":
                word_scores.append(max(c.sound, 0.0))
        srcs = " ".join(f"{c.source[0]}" for c in path)  # I/G/B/… per word
        print(f"{score:.3f}  EN {sent}")
        print(f"       FR {fr_line}")
        print(f"       [{srcs}]  (I=inv G=glue B=bridge G?=filler _=gap)\n")

    n = len(combos)
    combos.sort()
    print("=" * 64)
    print(f"sentences: {n}")
    print(f"whole-line combo: median {combos[n//2]:.3f} | "
          f"min {combos[0]:.3f} | max {combos[-1]:.3f} | "
          f">=0.60: {sum(c >= 0.60 for c in combos)}/{n}")
    if word_scores:
        word_scores.sort()
        print(f"word-level sound: median {word_scores[len(word_scores)//2]:.2f}")
    total = sum(sources.values())
    print("source mix: " + "  ".join(
        f"{s} {c} ({100*c/total:.0f}%)" for s, c in sources.most_common()))


if __name__ == "__main__":
    main()
