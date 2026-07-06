"""Build ByT5 transducer training data: EN phoneme stream -> French words.

Per docs/sentence-generation-plan.md, the transducer is a byte-level seq2seq
(ByT5) that reads the English IPA stream (plus the text for context) and
writes the French carve. Training pairs come from every verified source, in
curriculum order (the `stage` field): words -> phrases -> composed lines.

Row format (HF seq2seq JSONL):
  {"src": "ipa: <EN IPA> ; text: <EN>", "tgt": "<FR>", "stage": 1, "score": 0.83}

Sources:
  stage 1  single-word verified pairs (all cycles, >= --min-score)
  stage 2  multi-word verified pairs (out-multiword + gold phrases)
  stage 3  composed lines from compose_lattice over PD-style seed sentences
           (optional, --compose-file: one EN sentence per line)

Train (example, transformers):
  python train_byt5.py --data byt5_train.jsonl --model google/byt5-small
  (input/target both plain text; ByT5 needs no tokenizer config)

Run: python build_byt5_data.py --bench-dir <hb> --pipeline-dir . --out byt5_train.jsonl
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path


def _load_matcher(bench: Path):
    spec = importlib.util.spec_from_file_location("matcher", bench / "matcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("."))
    ap.add_argument("--min-score", type=float, default=0.60)
    ap.add_argument("--compose-file", type=Path, default=None,
                    help="EN sentences to compose into stage-3 lines")
    ap.add_argument("--out", type=Path, default=Path("byt5_train.jsonl"))
    args = ap.parse_args()

    matcher = _load_matcher(args.bench_dir)
    g2p_cache: dict[str, str] = {}

    def en_ipa(text: str) -> str:
        if text not in g2p_cache:
            g2p_cache[text] = matcher.g2p(text, "en")
        return g2p_cache[text]

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(en: str, fr: str, score: float, stage: int):
        if en == fr or (en, fr) in seen or score < args.min_score:
            return
        seen.add((en, fr))
        try:
            ipa = en_ipa(en)
        except Exception:
            return
        rows.append({"src": f"ipa: {ipa} ; text: {en}", "tgt": fr,
                     "stage": stage, "score": round(score, 3)})

    # stage 1+2: verified pairs from every cycle
    for sub in ("out", "out-cycle2", "out-cycle3", "out-multiword"):
        p = args.pipeline_dir / sub / "expansion-verified.tsv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                stage = 2 if (" " in r["en"] or " " in r["fr"]) else 1
                add(r["en"], r["fr"], float(r["score"]), stage)

    # gold tiers from the enlarged ladder
    ladder = args.pipeline_dir / "tier-ladder-cycle3.tsv"
    if not ladder.exists():
        ladder = args.bench_dir / "tier-ladder.tsv"
    with open(ladder, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["ladder"] in {"DUAL-S", "S", "STRICT-GOLD", "LOOP2", "LOOP1",
                               "GOLD", "EXPANSION"}:
                s = float(r["sound"]) if r.get("sound") else 0.75
                stage = 2 if (" " in r["en"] or " " in r["fr"]) else 1
                add(r["en"].strip(), r["fr"].strip(), s, stage)

    # stage 3: composed whole lines (the composer is the data generator)
    if args.compose_file and args.compose_file.exists():
        from compose_lattice import SentenceComposer
        composer = SentenceComposer(args.bench_dir, args.pipeline_dir)
        for line in args.compose_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            _, finals = composer.compose(line, n_out=1)
            for score, fr_line, path, _ in finals:
                if all(c.source != "GAP" for c in path):
                    add(line, fr_line, score, 3)

    rows.sort(key=lambda r: (r["stage"], -r["score"]))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_stage = {s: sum(1 for r in rows if r["stage"] == s) for s in (1, 2, 3)}
    print(f"{len(rows)} rows -> {args.out}  (stage1 {by_stage[1]} words, "
          f"stage2 {by_stage[2]} phrases, stage3 {by_stage[3]} lines)")


if __name__ == "__main__":
    main()
