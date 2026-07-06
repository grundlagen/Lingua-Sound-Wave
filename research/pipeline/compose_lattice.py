"""STAGE 5 (structure) — the sentence lattice composer.

How words bridge together into a sentence. For each English token, build a
candidate column; the sentence is a path through the columns (the LATTICE);
beam-search the path; re-rank finalists by the whole-line combo, which is
scored on espeak's native connected French — so liaison/elision at the seams
is part of the objective, not an afterthought.

Candidate columns, in trust order:

  INVENTORY  verified pairs — gold tiers + every cycle's expansion output
  GLUE       zipf-glue homophones for function words (the/and/of...)
  INFLECT    inventory hits on the token's lemma or inflections
             (walk has no cell but walks does -> use walks' cell)
  BRIDGE     meaning-mates' sound cells (paraphrase-bridges): the word is
             SWAPPED for a meaning-mate that has a proven homophone — the
             French rail keeps the sense, the sound cell is the mate's
  GAP        nothing found: keep a placeholder, take a penalty

Beam state score = sum(word sound scores) + LAMBDA * mean fr zipf (fluency
proxy). Final: top-K full lines re-scored with matcher.homophone_score on
the whole sentence (the real objective).

Run:
  python compose_lattice.py --bench-dir <hb> --pipeline-dir . \
      "the ocean remembers every vessel that ever sailed"
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

LAMBDA_FLUENCY = 0.08
GAP_PENALTY = 0.15
BRIDGE_DISCOUNT = 0.92     # a bridge changes wording; slight preference for direct cells
PER_COLUMN = 6             # candidates kept per token
BEAM = 24
FINALISTS = 12


@dataclass
class Cand:
    fr: str
    sound: float           # word-level combo (from inventory) or glue score
    zipf: float
    source: str            # INVENTORY | GLUE | INFLECT | BRIDGE | GAP
    note: str = ""         # e.g. bridge mate used


def _load_matcher(bench: Path):
    spec = importlib.util.spec_from_file_location("matcher", bench / "matcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SentenceComposer:
    def __init__(self, bench_dir: Path, pipeline_dir: Path):
        self.matcher = _load_matcher(bench_dir)
        from wordfreq import zipf_frequency
        self.zipf = zipf_frequency

        # INVENTORY: en -> [(fr, score)] from every verified source
        inv: dict[str, dict[str, float]] = defaultdict(dict)
        ladder = pipeline_dir / "tier-ladder-cycle3.tsv"
        if not ladder.exists():
            ladder = bench_dir / "tier-ladder.tsv"
        with open(ladder, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["ladder"] in {"DUAL-S", "S", "STRICT-GOLD", "LOOP2",
                                   "LOOP1", "GOLD", "EXPANSION"}:
                    s = float(r["sound"]) if r.get("sound") else 0.75
                    en, fr = r["en"].strip(), r["fr"].strip()
                    if en and fr and en != fr:   # identity rows (abcd~abcd) are junk
                        inv[en][fr] = max(inv[en].get(fr, 0), s)
        for sub in ("out", "out-cycle2", "out-cycle3", "out-multiword"):
            p = pipeline_dir / sub / "expansion-verified.tsv"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    for r in csv.DictReader(f, delimiter="\t"):
                        s = float(r["score"])
                        inv[r["en"]][r["fr"]] = max(inv[r["en"]].get(r["fr"], 0), s)
        self.inv = inv

        # GLUE
        self.glue: dict[str, list[tuple[str, float]]] = defaultdict(list)
        gp = bench_dir / "zipf-glue.tsv"
        if gp.exists():
            with open(gp, encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    self.glue[r["en"]].append((r["fr"], float(r["sound"])))

        # BRIDGES: word -> [(mate, mate_fr)]
        self.bridge: dict[str, list[tuple[str, str]]] = defaultdict(list)
        bp = pipeline_dir / "out-paraphrase" / "paraphrase-bridges-en.tsv"
        if bp.exists():
            with open(bp, encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    self.bridge[r["word"]].append((r["mate"], r["mate_homophone"]))

        print(f"inventory: {len(self.inv)} EN words with cells | "
              f"glue: {len(self.glue)} | bridges: {len(self.bridge)}")

    # ---- candidate column for one token ------------------------------------
    def column(self, word: str) -> list[Cand]:
        w = word.lower()
        cands: list[Cand] = []
        for fr, s in self.inv.get(w, {}).items():
            cands.append(Cand(fr, s, self.zipf(fr, "fr"), "INVENTORY"))
        for fr, s in self.glue.get(w, []):
            cands.append(Cand(fr, s, self.zipf(fr, "fr"), "GLUE"))
        if not cands:
            try:
                from lemminflect import getAllInflections, getAllLemmas
                neigh = {l for pos in getAllLemmas(w).values() for l in pos}
                for lemma in list(neigh):
                    for forms in getAllInflections(lemma).values():
                        neigh.update(f.lower() for f in forms)
                neigh.discard(w)
                for n in neigh:
                    for fr, s in self.inv.get(n, {}).items():
                        cands.append(Cand(fr, s * 0.97, self.zipf(fr, "fr"),
                                          "INFLECT", note=f"via {n}"))
            except ImportError:
                pass
        if not cands:
            for mate, fr in self.bridge.get(w, [])[:8]:
                s = self.inv.get(mate, {}).get(fr, 0.6)
                cands.append(Cand(fr, s * BRIDGE_DISCOUNT, self.zipf(fr, "fr"),
                                  "BRIDGE", note=f"means '{mate}'"))
        if not cands:
            cands.append(Cand(w, -GAP_PENALTY, 0.0, "GAP"))
        # keep the best few, sound first, fluency as tiebreak
        cands.sort(key=lambda c: (-c.sound, -c.zipf))
        return cands[:PER_COLUMN]

    # ---- compose -----------------------------------------------------------
    def compose(self, sentence: str, n_out: int = 3):
        tokens = [t for t in sentence.lower().replace(",", " ").split() if t]
        cols = [self.column(t) for t in tokens]

        beams: list[tuple[float, list[Cand]]] = [(0.0, [])]
        for col in cols:
            nxt = []
            for score, path in beams:
                for c in col:
                    nxt.append((score + c.sound + LAMBDA_FLUENCY * c.zipf,
                                path + [c]))
            nxt.sort(key=lambda x: -x[0])
            beams = nxt[:BEAM]

        # re-rank finalists on the WHOLE line (junctures included via espeak)
        finals = []
        seen: set[str] = set()
        for _, path in beams[:FINALISTS]:
            fr_line = " ".join(c.fr for c in path)
            if fr_line in seen:
                continue
            seen.add(fr_line)
            r = self.matcher.homophone_score(sentence, "en", fr_line, "fr")
            finals.append((r["score"], fr_line, path, r))
        finals.sort(key=lambda x: -x[0])
        return tokens, finals[:n_out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("."))
    ap.add_argument("sentence", nargs="+")
    args = ap.parse_args()

    composer = SentenceComposer(args.bench_dir, args.pipeline_dir)
    sentence = " ".join(args.sentence)
    tokens, finals = composer.compose(sentence)

    print(f"\nEN: {sentence}")
    for score, fr_line, path, r in finals:
        print(f"\nFR: {fr_line}")
        print(f"    whole-line combo {score:.3f} "
              f"(ngram {r['ngram_dice']:.2f} / feat {r['featural_nw']:.2f})")
        print(f"    ipa EN {r['ipa_a']}")
        print(f"    ipa FR {r['ipa_b']}")
        for t, c in zip(tokens, path):
            tag = f" [{c.source}{' ' + c.note if c.note else ''}]"
            print(f"      {t:14} -> {c.fr:16} {max(c.sound,0):.2f}{tag}")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "qwen-finetune"))
        from sandhi_fr import spoken_stream
        if finals:
            print(f"\nspoken (sandhi): {spoken_stream(finals[0][1])}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
