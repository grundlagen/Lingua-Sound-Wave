"""Multi-granularity composer for v5 material.

Builds pattern lots from:
  - partial units: reusable fragment chunks;
  - whole units: single dictionary entries;
  - multi units: dictionary entries whose French side is a phrase.

Then composes English input lines into French sound-lines with deterministic
selection and QC gates for coverage, rhythm drift, junctions, and fragment
overuse.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean

import matcher
from lexicon_g2p import load_en

matcher._vecs = lru_cache(maxsize=None)(matcher._vecs.__wrapped__)
matcher._segs = lru_cache(maxsize=None)(matcher._segs.__wrapped__)

DEFAULT_LINES = [
    "two men set the test",
    "we do see the group",
    "the west wind is here",
    "those words prove the rhythm",
]

TOKEN_RE = re.compile(r"[A-Za-z']+")
TIER_RANK = {"S": 0, "A": 1, "B_safe": 2, "B": 3, "B_reservoir": 4}
RISKY_FUNCWORDS = {"of", "here", "than"}
GLUE_FUNCWORDS = {"the", "a", "an", "and", "or", "to", "in", "on", "at", "by", "as"}


def is_vowel(seg: str) -> bool:
    vec = matcher._vecs(seg)
    return len(vec) > 0 and vec[0][0] == 1


def sound_class(seg: str) -> str:
    return "V" if is_vowel(seg) else "C"


def tokenize(line: str) -> list[str]:
    return [m.group(0).lower().strip("'") for m in TOKEN_RE.finditer(line)]


def entry_kind(entry: dict) -> str:
    if entry.get("multiword") or " " in entry.get("fr", "") or " " in entry.get("en", ""):
        return "multi"
    return "whole"


def funcword_band(entry: dict) -> str:
    if not entry.get("funcword"):
        return ""
    if entry.get("en") in RISKY_FUNCWORDS:
        return "funcword_risky"
    if entry.get("en") in GLUE_FUNCWORDS:
        return "funcword_core"
    return "funcword_glue"


def load_entries() -> list[dict]:
    return json.load(open("dictionary-v5.json", encoding="utf-8"))


def load_fragments() -> list[dict]:
    rows = []
    with open("fragments.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row["count"] = int(row["count"])
            rows.append(row)
    return rows


def build_lots(entries: list[dict], fragments: list[dict]) -> dict:
    lots: dict[str, dict] = {}

    def add(lot_key: tuple, unit_id: str, score: float, preview: str) -> None:
        key = "|".join(str(x) for x in lot_key)
        lot = lots.setdefault(key, {
            "key": key,
            "kind": lot_key[0],
            "pivot": lot_key[1],
            "en_syll": lot_key[2],
            "fr_syll": lot_key[3],
            "direction": lot_key[4],
            "tier": lot_key[5],
            "count": 0,
            "best_score": 0.0,
            "examples": [],
        })
        lot["count"] += 1
        lot["best_score"] = max(lot["best_score"], score)
        if len(lot["examples"]) < 8:
            lot["examples"].append({"id": unit_id, "preview": preview, "score": score})

    for i, entry in enumerate(entries):
        if not entry.get("usable_for_composition") or entry.get("direction", "en_fr") != "en_fr":
            continue
        kind = entry_kind(entry)
        add(
            (kind, entry.get("pivot", ""), entry.get("en_syll", ""), entry.get("fr_syll", ""),
             entry.get("direction", "en_fr"), entry.get("tier", "")),
            f"dict:{i}",
            float(entry.get("score", 0.0)),
            f"{entry.get('en')}~{entry.get('fr')}",
        )

    for i, frag in enumerate(fragments):
        if frag["count"] < 2:
            continue
        en_segs = matcher._segs(matcher._canonical(frag["en_chunk"]))
        fr_segs = matcher._segs(matcher._canonical(frag["fr_chunk"]))
        if not en_segs or not fr_segs:
            continue
        pivot = "".join(sound_class(s) for s in en_segs)
        score = round(min(0.98, 0.55 + math.log1p(frag["count"]) / 10), 3)
        add(("partial", pivot, "", "", "en_fr", "fragment"), f"frag:{i}", score,
            f"{frag['en_chunk']}->{frag['fr_chunk']}")

    return {"parameters": {"fragment_min_count": 2}, "lots": sorted(lots.values(), key=lambda x: (-x["count"], x["key"]))}


def sort_entries(entries: list[dict], prefer_multi: bool = False) -> list[dict]:
    def key(e: dict):
        band = funcword_band(e)
        band_penalty = 1 if band == "funcword_risky" else 0
        source_penalty = 0 if e.get("source_stage") != "v5.2_generated_validated" else 1
        kind_penalty = 0 if entry_kind(e) == "multi" else 1
        return (
            kind_penalty if prefer_multi else 0,
            TIER_RANK.get(e.get("tier"), 9),
            band_penalty,
            source_penalty,
            -float(e.get("score", 0.0)),
            float(e.get("effective_gap_ratio", e.get("gap_ratio", 1.0))),
            e.get("fr", ""),
        )

    return sorted(entries, key=key)


def build_indexes(entries: list[dict], fragments: list[dict], prefer_multi: bool = False):
    by_en: dict[str, list[dict]] = defaultdict(list)
    for i, entry in enumerate(entries):
        if entry.get("usable_for_composition") and entry.get("direction", "en_fr") == "en_fr":
            item = dict(entry)
            item["_id"] = f"dict:{i}"
            item["_kind"] = entry_kind(entry)
            by_en[item["en"]].append(item)
    for key in list(by_en):
        by_en[key] = sort_entries(by_en[key], prefer_multi=prefer_multi)

    frag_by_en: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for i, frag in enumerate(fragments):
        if frag["count"] < 2:
            continue
        en_seq = tuple(matcher._segs(matcher._canonical(frag["en_chunk"])))
        fr_seq = tuple(matcher._segs(matcher._canonical(frag["fr_chunk"])))
        if not en_seq or not fr_seq:
            continue
        item = dict(frag)
        item["_id"] = f"frag:{i}"
        item["_en_seq"] = en_seq
        item["_fr_seq"] = fr_seq
        item["_score"] = round(min(0.90, 0.50 + math.log1p(item["count"]) / 12), 3)
        frag_by_en[en_seq].append(item)
    for key in list(frag_by_en):
        frag_by_en[key].sort(key=lambda x: (-x["count"], x["fr_chunk"]))
    return by_en, frag_by_en


def cover_with_fragments(word: str, lex_en: dict[str, list[str]], frag_by_en: dict[tuple[str, ...], list[dict]]) -> dict | None:
    prons = lex_en.get(word)
    if not prons:
        return None
    segs = tuple(matcher._segs(matcher._canonical(prons[0])))
    if not segs:
        return None
    dp: list[tuple[float, list[dict]] | None] = [None] * (len(segs) + 1)
    dp[0] = (0.0, [])
    for i in range(len(segs)):
        if dp[i] is None:
            continue
        score, parts = dp[i]
        for j in range(i + 2, min(len(segs), i + 4) + 1):
            frag = (frag_by_en.get(segs[i:j]) or [None])[0]
            if frag is None:
                continue
            candidate = (score + frag["_score"], parts + [frag])
            if dp[j] is None or candidate[0] > dp[j][0]:
                dp[j] = candidate
    if dp[-1] is None:
        return None
    score, parts = dp[-1]
    fr_sound = "".join(part["fr_chunk"] for part in parts)
    return {
        "id": "+".join(part["_id"] for part in parts),
        "kind": "partial",
        "en": word,
        "fr": f"<{fr_sound}>",
        "score": round(score / max(1, len(parts)), 3),
        "tier": "fragment",
        "source_stage": "fragment_cover",
        "chunk_recipe": "+".join(part["en_chunk"] for part in parts),
        "en_onset": f"{segs[0]}|{sound_class(segs[0])}",
        "en_coda": f"{segs[-1]}|{sound_class(segs[-1])}",
        "fr_onset": "",
        "fr_coda": "",
        "syllable_delta": 0,
    }


def compose_line(
    line: str,
    by_en: dict[str, list[dict]],
    frag_by_en: dict[tuple[str, ...], list[dict]],
    lex_en: dict[str, list[str]],
    prefer_partial: bool = False,
    partial_threshold: float = 0.72,
) -> dict:
    words = tokenize(line)
    units = []
    missing = []
    for word in words:
        fallback = cover_with_fragments(word, lex_en, frag_by_en)
        if prefer_partial and fallback and fallback["score"] >= partial_threshold:
            units.append(fallback)
            continue
        if by_en.get(word):
            units.append(by_en[word][0])
            continue
        if fallback:
            units.append(fallback)
        else:
            units.append({"kind": "missing", "_kind": "missing", "en": word, "fr": f"[{word}]", "score": 0.0, "tier": "missing"})
            missing.append(word)

    warnings = []
    rhythm_delta = 0
    previous = None
    for unit in units:
        rhythm_delta += int(unit.get("syllable_delta") or 0)
        if previous and unit.get("fr_onset") and previous.get("fr_coda"):
            if str(previous["fr_coda"]).endswith("|V") and str(unit["fr_onset"]).endswith("|V"):
                warnings.append(f"hiatus:{previous.get('fr')} {unit.get('fr')}")
        previous = unit if unit.get("tier") != "missing" else previous

    covered = [u for u in units if u.get("tier") != "missing"]
    partials = [u for u in covered if u.get("_kind", u.get("kind")) == "partial"]
    scores = [float(u.get("score", 0.0)) for u in covered]
    coverage = len(covered) / max(1, len(words))
    partial_ratio = len(partials) / max(1, len(words))
    avg_score = round(mean(scores), 3) if scores else 0.0
    usable = coverage >= 0.80 and partial_ratio <= 0.40 and abs(rhythm_delta) <= 2 and avg_score >= 0.78

    return {
        "en": line,
        "fr": " ".join(unit.get("fr", "") for unit in units),
        "coverage": round(coverage, 3),
        "partial_ratio": round(partial_ratio, 3),
        "avg_score": avg_score,
        "rhythm_delta": rhythm_delta,
        "usable_for_composition": usable,
        "missing": missing,
        "junction_warnings": warnings,
        "unit_kinds": [u.get("_kind", u.get("kind", entry_kind(u) if u.get("tier") != "missing" else "missing")) for u in units],
        "tier_path": [u.get("tier", "") for u in units],
        "units": units,
    }


def read_lines(args: argparse.Namespace) -> list[str]:
    lines = list(args.lines)
    if args.lines_file:
        lines.extend(x.strip() for x in Path(args.lines_file).read_text(encoding="utf-8").splitlines() if x.strip())
    return lines or DEFAULT_LINES


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pattern lots and compose v5 dual lines.")
    parser.add_argument("lines", nargs="*", help="English line(s) to compose.")
    parser.add_argument("--lines-file", help="Optional newline-delimited English line file.")
    parser.add_argument("--seed", type=int, default=17, help="Deterministic seed recorded for reproducibility.")
    parser.add_argument("--prefer-multi", action="store_true", help="Prefer multiword French rows over whole-word rows when both exist.")
    parser.add_argument("--prefer-partial", action="store_true", help="Prefer fragment-cover rows when their score clears --partial-threshold.")
    parser.add_argument("--partial-threshold", type=float, default=0.72, help="Minimum fragment-cover score used by --prefer-partial.")
    args = parser.parse_args()

    entries = load_entries()
    fragments = load_fragments()
    lots = build_lots(entries, fragments)
    by_en, frag_by_en = build_indexes(entries, fragments, prefer_multi=args.prefer_multi)
    lex_en = load_en()
    lines = [
        compose_line(
            line,
            by_en,
            frag_by_en,
            lex_en,
            prefer_partial=args.prefer_partial,
            partial_threshold=args.partial_threshold,
        )
        for line in read_lines(args)
    ]

    with open("composition-lots.json", "w", encoding="utf-8") as f:
        json.dump(lots, f, ensure_ascii=False, indent=2)
    with open("composition-lines.json", "w", encoding="utf-8") as f:
        json.dump({
            "parameters": {
                "seed": args.seed,
                "prefer_multi": args.prefer_multi,
                "prefer_partial": args.prefer_partial,
                "partial_threshold": args.partial_threshold,
            },
            "lines": lines,
        }, f, ensure_ascii=False, indent=2)
    with open("composition-lines.tsv", "w", encoding="utf-8") as f:
        f.write("usable\tcoverage\tpartial_ratio\tavg_score\trhythm_delta\ten\tfr\tunit_kinds\ttier_path\twarnings\tmissing\n")
        for row in lines:
            f.write("\t".join(str(x) for x in [
                int(row["usable_for_composition"]), row["coverage"], row["partial_ratio"],
                row["avg_score"], row["rhythm_delta"], row["en"], row["fr"],
                ",".join(row["unit_kinds"]), ",".join(row["tier_path"]),
                ";".join(row["junction_warnings"]), ",".join(row["missing"]),
            ]) + "\n")

    print(f"seed: {args.seed}")
    print(f"lots: {len(lots['lots'])} pattern groups")
    print(f"lines: {sum(1 for x in lines if x['usable_for_composition'])}/{len(lines)} passed QC")
    print("wrote composition-lots.json, composition-lines.json, composition-lines.tsv")


if __name__ == "__main__":
    main()
