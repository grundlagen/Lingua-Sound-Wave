"""STAGE 5 (structure) — the sentence lattice composer.

How words bridge together into a sentence. For each English token, build a
candidate column; the sentence is a path through the columns (the LATTICE);
beam-search the path; re-rank finalists on the whole line matcher score, which is
scored on espeak's native connected French — so liaison/elision at the seams
is part of the objective, not an afterthought.

Candidate columns, in trust order:

  INVENTORY  verified pairs — gold tiers + every cycle's expansion output
  GLUE       zipf-glue homophones for function words (the/and/of...)
  INFLECT    inventory hits on the token's lemma or inflections
             (walk has no cell but walks does -> use walks' cell)
  BRIDGE     meaning-mates' sound cells (paraphrase-bridges): the word is
             SWAPPED for a meaning-mate that has a proven homophone — the
             French form keeps the sense, the sound cell is the mate's
  GAP_FILLER ByT5 transducer (if checkpoint available) or phonetic-decoder
             fallback — generates a fresh French candidate for the GAP
  GAP        nothing found: keep a placeholder, take a penalty

Beam state score = sum(word sound scores) + LAMBDA * mean fr zipf (fluency
proxy). Final: top-K full lines re-scored with matcher.homophone_score on
the whole sentence (the real objective).

Run:
  python compose_lattice.py --bench-dir <hb> --pipeline-dir . \
      "the ocean remembers every vessel that ever sailed"
  python compose_lattice.py --bench-dir <hb> --pipeline-dir . \
      --byt5-checkpoint ckpt/byt5-carver "she wandered..."
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
BRIDGE_DISCOUNT = 0.92
FILLER_DISCOUNT = 0.88     # generative fill — slightly more discount than bridge
PER_COLUMN = 6
BEAM = 24
FINALISTS = 12

# Function words that should trigger inventory fallback in GLUE tier
# (short, high-frequency words where S-tier may have arcane matches)
FUNC_WORDS = frozenset("""
the a an and or to in on at by as of for that you it was be are have he not but
from my we his me if one can will just like get out up so go see she said were
been had did does would could should may might must each any more some than very
too then now there here how what who when why also still even only over under
after before between both own same such without within while since once never
always often quite almost already other much many few well good new about people
really should
""".split())


@dataclass
class Cand:
    fr: str
    sound: float
    zipf: float
    source: str            # INVENTORY | GLUE | INFLECT | BRIDGE | GAP_FILLER | GAP
    note: str = ""


def _load_matcher(bench: Path):
    spec = importlib.util.spec_from_file_location("matcher", bench / "matcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SentenceComposer:
    def __init__(self, bench_dir: Path, pipeline_dir: Path,
                 byt5_checkpoint: Path | None = None):
        self.matcher = _load_matcher(bench_dir)
        from wordfreq import zipf_frequency
        self.zipf = zipf_frequency
        self.bench_dir = bench_dir
        self.pipeline_dir = pipeline_dir
        self.byt5_ckpt = byt5_checkpoint
        self._byt5_model = None
        self._byt5_tok = None
        self._pd_root = None
        self._pd = None

        # INVENTORY
        self.inv = self._load_inventory()

        # GLUE
        self.glue: dict[str, list[tuple[str, float]]] = defaultdict(list)
        gp = bench_dir / "zipf-glue.tsv"
        if gp.exists():
            with open(gp, encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    self.glue[r["en"]].append((r["fr"], float(r["sound"])))

        # BRIDGES
        self.bridge: dict[str, list[tuple[str, str]]] = defaultdict(list)
        bp = pipeline_dir / "out-paraphrase" / "paraphrase-bridges-en.tsv"
        if bp.exists():
            with open(bp, encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    self.bridge[r["word"]].append((r["mate"], r["mate_homophone"]))

        print(f"inventory: {len(self.inv)} EN words with cells | "
              f"glue: {len(self.glue)} | bridges: {len(self.bridge)}"
              + (f" | byt5: {self.byt5_ckpt}" if self.byt5_ckpt else ""))

    def _load_inventory(self) -> dict[str, dict[str, float]]:
        inv: dict[str, dict[str, float]] = defaultdict(dict)
        ladder = self.pipeline_dir / "tier-ladder-cycle3.tsv"
        if not ladder.exists():
            ladder = self.bench_dir / "tier-ladder.tsv"
        with open(ladder, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["ladder"] in {"DUAL-S", "S", "STRICT-GOLD", "LOOP2",
                                   "LOOP1", "GOLD", "EXPANSION"}:
                    s = float(r["sound"]) if r.get("sound") else 0.75
                    en, fr = r["en"].strip(), r["fr"].strip()
                    if en and fr and en != fr:  # identity rows are junk
                        inv[en][fr] = max(inv[en].get(fr, 0), s)
        for sub in ("out", "out-cycle2", "out-cycle3", "out-multiword"):
            p = self.pipeline_dir / sub / "expansion-verified.tsv"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    for r in csv.DictReader(f, delimiter="\t"):
                        s = float(r["score"])
                        inv[r["en"]][r["fr"]] = max(inv[r["en"]].get(r["fr"], 0), s)
        return inv

    # ---- ByT5 transducer (lazy-load; works without GPU) --------------------
    def _load_byt5(self) -> bool:
        if self._byt5_model is not None:
            return True
        if not self.byt5_ckpt or not self.byt5_ckpt.exists():
            return False
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._byt5_tok = AutoTokenizer.from_pretrained(str(self.byt5_ckpt))
            self._byt5_model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.byt5_ckpt)).to("cpu")
            return True
        except Exception:
            return False

    def _try_byt5(self, en_word: str) -> list[tuple[str, float]]:
        """Generate French candidates for a GAP word via trained ByT5."""
        if not self._load_byt5():
            return []
        try:
            ipa = self.matcher.g2p(en_word, "en")
            src = f"ipa: {ipa} ; text: {en_word}"
            enc = self._byt5_tok(src, return_tensors="pt", max_length=256,
                                 truncation=True)
            out = self._byt5_model.generate(**enc, max_length=48, num_beams=5,
                                            num_return_sequences=3,
                                            early_stopping=True)
            results = []
            for o in out:
                fr = self._byt5_tok.decode(o, skip_special_tokens=True).strip()
                if fr and fr != en_word:
                    s = self.matcher.homophone_score(en_word, "en", fr, "fr")["score"]
                    results.append((fr, s * FILLER_DISCOUNT))
            return results
        except Exception:
            return []

    def _load_pd(self) -> bool:
        if self._pd_root is not None:
            return True
        try:
            import sys
            sys.path.insert(0, str(self.bench_dir))
            pd_path = self.bench_dir / "phonetic_decoder.py"
            spec = importlib.util.spec_from_file_location("pd", pd_path)
            self._pd = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._pd)
            self._pd_root = self._pd.build_trie(min_zipf=2.0, lang="fr")
            return True
        except Exception as e:
            print(f"  (phonetic decoder unavailable: {e})")
            return False

    def _try_decoder(self, en_word: str) -> list[tuple[str, float]]:
        """Phonetic-decoder fallback: decode EN IPA into FR words."""
        if not self._load_pd():
            return []
        try:
            ipa = self.matcher.g2p(en_word, "en")
            cands = self._pd.decode(ipa, self._pd_root, top_n=5, max_words=2)
            results = []
            for c in cands:
                if c["similarity"] < 0.55:
                    continue
                fr = c["fr"]
                if fr and fr.lower() != en_word.lower():
                    score = c["similarity"] * FILLER_DISCOUNT
                    results.append((fr, score))
            return results
        except Exception:
            return []

    # ---- candidate column for one token ------------------------------------
    def column(self, word: str) -> list[Cand]:
        w = word.lower()
        cands: list[Cand] = []

        # Tier 1: INVENTORY
        for fr, s in self.inv.get(w, {}).items():
            cands.append(Cand(fr, s, self.zipf(fr, "fr"), "INVENTORY"))

        # Tier 2: GLUE — zipf-glue homophones, PLUS inventory fallback for
        # function words (the S-tier inventory has arcane matches that
        # brute-force zipf mining misses: about→ébattent, get→guette, etc.)
        glue_hits = list(self.glue.get(w, []))
        # Also check full inventory for function words — arcane matches
        # that aren't in zipf-glue but exist in S/GOLD/STRICT-GOLD tiers
        inv_fallbacks = []
        if not glue_hits and len(w) <= 6 and w in FUNC_WORDS:
            for fr, s in self.inv.get(w, {}).items():
                inv_fallbacks.append((fr, s))
            inv_fallbacks.sort(key=lambda x: -x[1])
            inv_fallbacks = inv_fallbacks[:4]
        for fr, s in glue_hits + inv_fallbacks:
            source = "GLUE" if (fr, s) in glue_hits else "GLUE(inv)"
            cands.append(Cand(fr, s, self.zipf(fr, "fr"), source))

        # Tier 3: INFLECT
        if not cands:
            try:
                from lemminflect import getAllInflections, getAllLemmas
                neigh = {lemma for pos in getAllLemmas(w).values() for lemma in pos}
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

        # Tier 4: BRIDGE
        if not cands:
            for mate, fr in self.bridge.get(w, [])[:8]:
                s = self.inv.get(mate, {}).get(fr, 0.6)
                cands.append(Cand(fr, s * BRIDGE_DISCOUNT, self.zipf(fr, "fr"),
                                  "BRIDGE", note=f"means '{mate}'"))

        # Tier 5: GAP_FILLER — ByT5 or phonetic-decoder
        if not cands:
            byt5_fills = self._try_byt5(w)
            for fr, s in byt5_fills:
                cands.append(Cand(fr, s, self.zipf(fr, "fr"),
                                  "GAP_FILLER", note="byt5"))
            if not byt5_fills:
                dec_fills = self._try_decoder(w)
                for fr, s in dec_fills:
                    cands.append(Cand(fr, s, self.zipf(fr, "fr"),
                                      "GAP_FILLER", note="decoder"))

        # Tier 6: GAP
        if not cands:
            cands.append(Cand(w, -GAP_PENALTY, 0.0, "GAP"))

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
    ap.add_argument("--byt5-checkpoint", type=Path, default=None,
                    help="trained ByT5 checkpoint for GAP-filling")
    ap.add_argument("sentence", nargs="+")
    args = ap.parse_args()

    composer = SentenceComposer(args.bench_dir, args.pipeline_dir,
                                args.byt5_checkpoint)
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
