"""Query tool for chain/loop/pair TSV data files.

    python query_chains.py search cross         # search all datasets for "cross"
    python query_chains.py chains cross          # chains starting/ending at "cross"
    python query_chains.py loops cross           # loops seeded by "cross"
    python query_chains.py pairs cross           # certified loop pairs containing "cross"
    python query_chains.py web cross troupe      # subchains connecting two nodes
    python query_chains.py stats                 # summary statistics for all datasets
    python query_chains.py top --dataset loops   # top entries by quality

Datasets loaded (all from this directory):
  chain-web.tsv          EN->FR chains (src, dst, hops, quality, chain)
  chain-web-full-S.tsv   full subchain web (a, b, hops, quality, subchain)
  chain-loops.tsv        loops returning to seed (seed, hops, quality, loop)
  chain-loops-S.tsv      loops (S-tier subset)
  loop-certified-pairs.tsv      certified EN-FR pairs (en, fr, certifications, example_loop)
  loop-certified-pairs-S.tsv    certified pairs (S-tier subset)
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- data models

@dataclass
class Chain:
    src: str
    dst: str
    hops: int
    quality: float
    chain: str

@dataclass
class Subchain:
    a: str
    b: str
    hops: int
    quality: float
    subchain: str

@dataclass
class Loop:
    seed: str
    hops: int
    quality: float
    loop: str

@dataclass
class CertifiedPair:
    en: str
    fr: str
    certifications: int
    example_loop: str

# ---------------------------------------------------------------- loading

def _read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

@dataclass
class Dataset:
    chains: list[Chain] = field(default_factory=list)
    chains_s: list[Chain] = field(default_factory=list)
    subchains: list[Subchain] = field(default_factory=list)
    loops: list[Loop] = field(default_factory=list)
    loops_s: list[Loop] = field(default_factory=list)
    pairs: list[CertifiedPair] = field(default_factory=list)
    pairs_s: list[CertifiedPair] = field(default_factory=list)

def load() -> Dataset:
    ds = Dataset()

    def _chains(path: Path) -> list[Chain]:
        return [Chain(r["src"], r["dst"], int(r["hops"]), float(r["quality"]), r["chain"])
                for r in _read_tsv(path)]

    def _subchains(path: Path) -> list[Subchain]:
        return [Subchain(r["a"], r["b"], int(r["hops"]), float(r["quality"]), r["subchain"])
                for r in _read_tsv(path)]

    def _loops(path: Path) -> list[Loop]:
        return [Loop(r["seed"], int(r["hops"]), float(r["quality"]), r["loop"])
                for r in _read_tsv(path)]

    def _pairs(path: Path) -> list[CertifiedPair]:
        return [CertifiedPair(r["en"], r["fr"], int(r["certifications"]), r["example_loop"])
                for r in _read_tsv(path)]

    p = DIR / "chain-web.tsv"
    if p.exists(): ds.chains = _chains(p)
    p = DIR / "chain-web-S.tsv"
    if p.exists(): ds.chains_s = _chains(p)
    p = DIR / "chain-web-full-S.tsv"
    if p.exists(): ds.subchains = _subchains(p)
    p = DIR / "chain-loops.tsv"
    if p.exists(): ds.loops = _loops(p)
    p = DIR / "chain-loops-S.tsv"
    if p.exists(): ds.loops_s = _loops(p)
    p = DIR / "loop-certified-pairs.tsv"
    if p.exists(): ds.pairs = _pairs(p)
    p = DIR / "loop-certified-pairs-S.tsv"
    if p.exists(): ds.pairs_s = _pairs(p)

    return ds

# ---------------------------------------------------------------- queries

def search_all(ds: Dataset, pattern: str, limit: int) -> None:
    pat = re.compile(pattern, re.IGNORECASE)
    hits = 0

    def emit(tag: str, quality: float, hops: int, text: str):
        nonlocal hits
        if hits >= limit:
            return
        hits += 1
        print(f"  [{tag}] q={quality:.3f} hops={hops}  {text}")

    for c in ds.chains:
        if pat.search(c.src) or pat.search(c.dst) or pat.search(c.chain):
            emit("chain", c.quality, c.hops, c.chain)
    for s in ds.subchains:
        if pat.search(s.a) or pat.search(s.b) or pat.search(s.subchain):
            emit("subchain", s.quality, s.hops, s.subchain)
    for lo in ds.loops:
        if pat.search(lo.seed) or pat.search(lo.loop):
            emit("loop", lo.quality, lo.hops, lo.loop)
    for p in ds.pairs:
        if pat.search(p.en) or pat.search(p.fr) or pat.search(p.example_loop):
            emit(f"pair(x{p.certifications})", 0, 0, f"{p.en} <-> {p.fr}  {p.example_loop}")
    if hits == 0:
        print(f"  (no results for /{pattern}/)")
    elif hits >= limit:
        print(f"  ... (capped at {limit}; use --limit to see more)")


def query_chains(ds: Dataset, word: str, limit: int, s_only: bool) -> None:
    source = ds.chains_s if s_only else ds.chains
    hits = [c for c in source if c.src.lower() == word.lower() or c.dst.lower() == word.lower()]
    hits.sort(key=lambda c: -c.quality)
    if not hits:
        print(f"  no chains for '{word}'")
        return
    for c in hits[:limit]:
        print(f"  q={c.quality:.3f} hops={c.hops}  {c.src} -> {c.dst}  {c.chain}")


def query_loops(ds: Dataset, word: str, limit: int, s_only: bool) -> None:
    source = ds.loops_s if s_only else ds.loops
    hits = [lo for lo in source if lo.seed.lower() == word.lower()]
    hits.sort(key=lambda lo: -lo.quality)
    if not hits:
        print(f"  no loops for '{word}'")
        return
    for lo in hits[:limit]:
        print(f"  q={lo.quality:.3f} hops={lo.hops}  {lo.loop}")


def query_pairs(ds: Dataset, word: str, limit: int, s_only: bool) -> None:
    source = ds.pairs_s if s_only else ds.pairs
    hits = [p for p in source if word.lower() in p.en.lower() or word.lower() in p.fr.lower()]
    hits.sort(key=lambda p: -p.certifications)
    if not hits:
        print(f"  no certified pairs for '{word}'")
        return
    for p in hits[:limit]:
        print(f"  certs={p.certifications}  {p.en} <-> {p.fr}  {p.example_loop}")


def query_web(ds: Dataset, a: str, b: str, limit: int) -> None:
    al, bl = a.lower(), b.lower()
    a_pats = {al, f"en:{al}", f"fr:{al}"}
    b_pats = {bl, f"en:{bl}", f"fr:{bl}"}

    def match(node: str, pats: set[str]) -> bool:
        nl = node.lower()
        return any(p == nl or (not p.startswith(("en:", "fr:")) and nl.endswith(f":{p}")) for p in pats)

    hits = [s for s in ds.subchains
            if (match(s.a, a_pats) and match(s.b, b_pats))
            or (match(s.a, b_pats) and match(s.b, a_pats))]
    hits.sort(key=lambda s: (-s.quality, s.hops))
    if not hits:
        print(f"  no subchains connecting '{a}' and '{b}'")
        return
    for s in hits[:limit]:
        print(f"  q={s.quality:.3f} hops={s.hops}  {s.a} -> {s.b}  {s.subchain}")


def show_stats(ds: Dataset) -> None:
    def stats(name: str, items: list, qual_fn):
        if not items:
            print(f"  {name:30s}  (empty)")
            return
        quals = [qual_fn(i) for i in items]
        print(f"  {name:30s}  {len(items):>7,} rows  "
              f"quality {min(quals):.3f} / {sum(quals)/len(quals):.3f} / {max(quals):.3f}")

    stats("chain-web.tsv", ds.chains, lambda c: c.quality)
    stats("chain-web-S.tsv", ds.chains_s, lambda c: c.quality)
    stats("chain-web-full-S.tsv", ds.subchains, lambda s: s.quality)
    stats("chain-loops.tsv", ds.loops, lambda lo: lo.quality)
    stats("chain-loops-S.tsv", ds.loops_s, lambda lo: lo.quality)
    stats("loop-certified-pairs.tsv", ds.pairs, lambda p: p.certifications)
    stats("loop-certified-pairs-S.tsv", ds.pairs_s, lambda p: p.certifications)


def show_top(ds: Dataset, dataset: str, limit: int) -> None:
    mapping = {
        "chains": ("chain-web.tsv", ds.chains, lambda c: (-c.quality, c.hops)),
        "chains-s": ("chain-web-S.tsv", ds.chains_s, lambda c: (-c.quality, c.hops)),
        "subchains": ("chain-web-full-S.tsv", ds.subchains, lambda s: (-s.quality, s.hops)),
        "loops": ("chain-loops.tsv", ds.loops, lambda lo: (-lo.quality, lo.hops)),
        "loops-s": ("chain-loops-S.tsv", ds.loops_s, lambda lo: (-lo.quality, lo.hops)),
        "pairs": ("loop-certified-pairs.tsv", ds.pairs, lambda p: -p.certifications),
        "pairs-s": ("loop-certified-pairs-S.tsv", ds.pairs_s, lambda p: -p.certifications),
    }
    if dataset not in mapping:
        print(f"  unknown dataset '{dataset}'; choices: {', '.join(mapping)}")
        return

    name, items, key = mapping[dataset]
    if not items:
        print(f"  {name} is empty")
        return
    items_sorted = sorted(items, key=key)
    print(f"  top {min(limit, len(items_sorted))} from {name}:")
    for item in items_sorted[:limit]:
        if isinstance(item, Chain):
            print(f"    q={item.quality:.3f} hops={item.hops}  {item.src} -> {item.dst}  {item.chain}")
        elif isinstance(item, Subchain):
            print(f"    q={item.quality:.3f} hops={item.hops}  {item.a} -> {item.b}  {item.subchain}")
        elif isinstance(item, Loop):
            print(f"    q={item.quality:.3f} hops={item.hops}  {item.loop}")
        elif isinstance(item, CertifiedPair):
            print(f"    certs={item.certifications}  {item.en} <-> {item.fr}  {item.example_loop}")

# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(
        description="Query chain/loop/pair TSV datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--limit", "-n", type=int, default=20, help="max results (default 20)")
    shared.add_argument("--s-only", "-s", action="store_true", help="restrict to S-tier datasets")

    sub = parser.add_subparsers(dest="cmd")

    p_search = sub.add_parser("search", parents=[shared], help="regex search across all datasets")
    p_search.add_argument("pattern", help="regex pattern to match")

    p_chains = sub.add_parser("chains", parents=[shared], help="chains starting/ending at a word")
    p_chains.add_argument("word")

    p_loops = sub.add_parser("loops", parents=[shared], help="loops seeded by a word")
    p_loops.add_argument("word")

    p_pairs = sub.add_parser("pairs", parents=[shared], help="certified pairs containing a word")
    p_pairs.add_argument("word")

    p_web = sub.add_parser("web", parents=[shared], help="subchains connecting two nodes")
    p_web.add_argument("a", help="first node (word or en:word / fr:word)")
    p_web.add_argument("b", help="second node")

    sub.add_parser("stats", parents=[shared], help="summary statistics for all datasets")

    p_top = sub.add_parser("top", parents=[shared], help="top entries by quality")
    p_top.add_argument("--dataset", "-d", default="loops",
                       help="which dataset (chains, chains-s, subchains, loops, loops-s, pairs, pairs-s)")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    print("loading data...", file=sys.stderr)
    ds = load()
    print("done.", file=sys.stderr)

    if args.cmd == "search":
        search_all(ds, args.pattern, args.limit)
    elif args.cmd == "chains":
        query_chains(ds, args.word, args.limit, args.s_only)
    elif args.cmd == "loops":
        query_loops(ds, args.word, args.limit, args.s_only)
    elif args.cmd == "pairs":
        query_pairs(ds, args.word, args.limit, args.s_only)
    elif args.cmd == "web":
        query_web(ds, args.a, args.b, args.limit)
    elif args.cmd == "stats":
        show_stats(ds)
    elif args.cmd == "top":
        show_top(ds, args.dataset, args.limit)


if __name__ == "__main__":
    main()
