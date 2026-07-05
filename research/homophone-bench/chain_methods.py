"""New chain methods: graduated tiers, density certification, deep fragments.

Standalone — reads dictionary-v5.json, fragments.tsv, and the chain TSVs.
Never imports or modifies existing scripts.

    python chain_methods.py all          # run everything, write new TSVs
    python chain_methods.py graduated    # graduated S/A+/A tier re-scoring
    python chain_methods.py density      # density-based certification
    python chain_methods.py shortchains  # extract strongest 2-hop evidence
    python chain_methods.py fragments    # exact deep fragment composition (3-6 chunks)

Outputs (all new files, prefixed 'new-'):
    new-chain-loops-graduated.tsv        loops re-scored with min-edge + graduated tiers
    new-certified-pairs-density.tsv      pairs certified by subchain density
    new-short-chain-evidence.tsv         2-hop chains ranked as strongest evidence
    new-fragment-compositions.tsv        novel 3-6 chunk fragment compositions
    new-methods-report.txt               summary report
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DIR = Path(__file__).resolve().parent
SEP = re.compile(r" ([≈=~]) ")


# ---------------------------------------------------------------- helpers

def parse_chain(s: str) -> list[tuple[str, str | None]]:
    parts = SEP.split(s)
    out = [(parts[0], None)]
    for i in range(1, len(parts), 2):
        out.append((parts[i + 1], parts[i]))
    return out


def load_json(name: str):
    with open(DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_tsv(name: str) -> list[dict]:
    p = DIR / name
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(name: str, rows: list[dict], fieldnames: list[str]):
    p = DIR / name
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {name}", file=sys.stderr)


# ---------------------------------------------------------------- data

class Data:
    def __init__(self):
        print("loading dictionary...", file=sys.stderr)
        self.entries = load_json("dictionary-v5.json")
        self.score_index: dict[tuple[str, str], float] = {}
        self.ipa_en: dict[str, str] = {}
        self.ipa_fr: dict[str, str] = {}
        for e in self.entries:
            key = (f"en:{e['en']}", f"fr:{e['fr']}")
            self.score_index[key] = e["score"]
            self.score_index[(key[1], key[0])] = e["score"]
            self.ipa_en[e["en"]] = e.get("en_ipa", "")
            self.ipa_fr[e["fr"]] = e.get("fr_ipa", "")
        self.usable = [e for e in self.entries if e.get("usable_for_composition")]
        print(f"  {len(self.entries)} entries, {len(self.usable)} usable, "
              f"{len(self.score_index)//2} score pairs", file=sys.stderr)

    def edge_score(self, a: str, b: str) -> float | None:
        return self.score_index.get((a, b))

    def score_chain_edges(self, chain_str: str) -> list[tuple[str, float, str]]:
        """Return [(edge_label, score, 'a->b'), ...] for each edge in a chain."""
        nodes = parse_chain(chain_str)
        edges = []
        for k in range(1, len(nodes)):
            node, lab = nodes[k]
            prev = nodes[k - 1][0]
            if lab == "≈":
                s = self.edge_score(prev, node)
                if s is None:
                    s = self.edge_score(node, prev)
                if s is None:
                    s = 0.88
                edges.append((lab, s, f"{prev}->{node}"))
            elif lab == "=":
                edges.append((lab, 0.95, f"{prev}->{node}"))
            else:
                edges.append((lab, 0.92, f"{prev}->{node}"))
        return edges


# ---------------------------------------------------------------- 1. graduated tiers

def graduated_tiers(data: Data, out: list[str]):
    """Re-score all loops with per-edge quality and assign graduated tiers."""
    out.append("=" * 72)
    out.append("METHOD 1: GRADUATED TIER RE-SCORING")
    out.append("=" * 72)

    loops_raw = load_tsv("chain-loops.tsv")
    if not loops_raw:
        out.append("  (no chain-loops.tsv found)")
        return

    results = []
    for lo in loops_raw:
        edges = data.score_chain_edges(lo["loop"])
        if not edges:
            continue
        scores = [s for _, s, _ in edges]
        sound_scores = [s for lab, s, _ in edges if lab == "≈"]

        mean_q = sum(scores) / len(scores)
        min_q = min(scores)
        min_sound = min(sound_scores) if sound_scores else 0.0
        max_sound = max(sound_scores) if sound_scores else 0.0

        if min_sound >= 0.95:
            tier = "S+"
        elif min_sound >= 0.90:
            tier = "S"
        elif min_sound >= 0.85:
            tier = "A+"
        elif min_sound >= 0.80:
            tier = "A"
        else:
            tier = "B"

        results.append(dict(
            seed=lo["seed"], hops=int(lo["hops"]),
            mean_quality=round(mean_q, 4),
            min_quality=round(min_q, 4),
            min_sound_edge=round(min_sound, 4),
            max_sound_edge=round(max_sound, 4),
            tier=tier,
            loop=lo["loop"]))

    tier_counts = Counter(r["tier"] for r in results)
    out.append(f"\nRe-scored {len(results)} loops with per-edge quality:")
    out.append(f"  Tier distribution:")
    for t in ["S+", "S", "A+", "A", "B"]:
        n = tier_counts.get(t, 0)
        out.append(f"    {t:3s}: {n:4d} loops ({100*n/max(1,len(results)):.1f}%)")

    results.sort(key=lambda r: (-{"S+": 4, "S": 3, "A+": 2, "A": 1, "B": 0}[r["tier"]],
                                 -r["min_sound_edge"], -r["mean_quality"]))

    write_tsv("new-chain-loops-graduated.tsv", results,
              ["seed", "hops", "mean_quality", "min_quality",
               "min_sound_edge", "max_sound_edge", "tier", "loop"])

    out.append(f"\n--- NEW: A+ tier loops (min sound edge 0.85-0.90) ---")
    aplus = [r for r in results if r["tier"] == "A+"]
    aplus.sort(key=lambda r: -r["min_sound_edge"])
    for r in aplus[:20]:
        out.append(f"  min_snd={r['min_sound_edge']:.3f} mean={r['mean_quality']:.3f} "
                   f"hops={r['hops']}  {r['loop'][:85]}")

    out.append(f"\n--- S+ tier (all sound edges >= 0.95) ---")
    splus = [r for r in results if r["tier"] == "S+"]
    for r in splus[:15]:
        out.append(f"  min_snd={r['min_sound_edge']:.3f} mean={r['mean_quality']:.3f} "
                   f"hops={r['hops']}  {r['loop'][:85]}")

    aplus_seeds = {r["seed"] for r in aplus}
    s_seeds = {r["seed"] for r in results if r["tier"] in ("S", "S+")}
    new_seeds = aplus_seeds - s_seeds
    out.append(f"\n  A+ tier adds {len(new_seeds)} NEW seeds not in S/S+ tier")
    out.append(f"  Total loop seeds: S+={len({r['seed'] for r in results if r['tier']=='S+'})} "
               f"S={len({r['seed'] for r in results if r['tier']=='S'})} "
               f"A+={len(aplus_seeds)} "
               f"A={len({r['seed'] for r in results if r['tier']=='A'})} "
               f"B={len({r['seed'] for r in results if r['tier']=='B'})}")

    # Extract certified pairs from A+ loops
    aplus_cert = Counter()
    aplus_example = {}
    for r in aplus:
        nodes = parse_chain(r["loop"])
        for k in range(1, len(nodes)):
            if nodes[k][1] == "≈":
                a, b = nodes[k - 1][0], nodes[k][0]
                en = a if a.startswith("en:") else b
                fr = b if b.startswith("fr:") else a
                if en.startswith("en:") and fr.startswith("fr:"):
                    pair = (en[3:], fr[3:])
                    aplus_cert[pair] += 1
                    aplus_example.setdefault(pair, r["loop"])

    existing_s = set()
    for p in load_tsv("loop-certified-pairs-S.tsv"):
        existing_s.add((p["en"], p["fr"]))

    new_aplus_pairs = [(p, n) for p, n in aplus_cert.items() if p not in existing_s]
    new_aplus_pairs.sort(key=lambda x: -x[1])
    out.append(f"\n  A+ tier produces {len(aplus_cert)} certified pairs, "
               f"{len(new_aplus_pairs)} NEW (not in S-tier)")
    out.append(f"\n--- Top NEW A+ certified pairs ---")
    for (en, fr), n in new_aplus_pairs[:20]:
        out.append(f"  certs={n:2d}  {en:20s} <-> {fr:20s}  {aplus_example[(en, fr)][:65]}")


# ---------------------------------------------------------------- 2. density certification

def density_certification(data: Data, out: list[str]):
    """Certify pairs by subchain passage density instead of loop membership."""
    out.append("\n" + "=" * 72)
    out.append("METHOD 2: DENSITY-BASED CERTIFICATION")
    out.append("=" * 72)

    subchains = load_tsv("chain-web-full-S.tsv")
    chains_s = load_tsv("chain-web-S.tsv")
    if not subchains:
        out.append("  (no chain-web-full-S.tsv found)")
        return

    # Build pair passage index: for each (en, fr) sound pair, count subchains containing both nodes
    pair_passages: dict[tuple[str, str], list] = defaultdict(list)

    for sc in subchains:
        nodes = parse_chain(sc["subchain"])
        # Collect all sound edges in this subchain
        for k in range(1, len(nodes)):
            if nodes[k][1] == "≈":
                a, b = nodes[k - 1][0], nodes[k][0]
                en_node = a if a.startswith("en:") else b
                fr_node = b if b.startswith("fr:") else a
                if en_node.startswith("en:") and fr_node.startswith("fr:"):
                    pair_passages[(en_node[3:], fr_node[3:])].append(
                        (float(sc["quality"]), int(sc["hops"]), sc["subchain"]))

    # Score each pair by density metrics
    results = []
    for (en, fr), passages in pair_passages.items():
        n = len(passages)
        if n < 2:
            continue
        best_q = max(q for q, _, _ in passages)
        mean_q = sum(q for q, _, _ in passages) / n
        min_hops = min(h for _, h, _ in passages)
        score = data.edge_score(f"en:{en}", f"fr:{fr}")
        if score is None:
            score = data.edge_score(f"fr:{fr}", f"en:{en}")

        best_example = max(passages, key=lambda x: (x[0], -x[1]))
        results.append(dict(
            en=en, fr=fr,
            pair_score=round(score, 4) if score else "",
            passages=n,
            best_subchain_quality=round(best_q, 4),
            mean_subchain_quality=round(mean_q, 4),
            min_hops=min_hops,
            density_score=round(n * mean_q, 2),
            example=best_example[2][:120]))

    results.sort(key=lambda r: -r["density_score"])

    write_tsv("new-certified-pairs-density.tsv", results,
              ["en", "fr", "pair_score", "passages", "best_subchain_quality",
               "mean_subchain_quality", "min_hops", "density_score", "example"])

    # Compare with loop-certified pairs
    loop_certified = {(p["en"], p["fr"]) for p in load_tsv("loop-certified-pairs-S.tsv")}
    density_only = [r for r in results if (r["en"], r["fr"]) not in loop_certified]
    both = [r for r in results if (r["en"], r["fr"]) in loop_certified]

    out.append(f"\nDensity-certified pairs (>= 2 subchain passages): {len(results)}")
    out.append(f"  Also loop-certified: {len(both)}")
    out.append(f"  Density-only (NEW):  {len(density_only)}")

    out.append(f"\n--- Top 20 density-certified pairs (overall) ---")
    for r in results[:20]:
        in_loop = "+" if (r["en"], r["fr"]) in loop_certified else " "
        out.append(f"  [{in_loop}] density={r['density_score']:6.1f}  passages={r['passages']:3d}  "
                   f"q={r['best_subchain_quality']:.3f}  {r['en']:20s} <-> {r['fr']}")

    out.append(f"\n--- Top 20 density-ONLY pairs (not loop-certified) ---")
    for r in density_only[:20]:
        out.append(f"  density={r['density_score']:6.1f}  passages={r['passages']:3d}  "
                   f"score={r['pair_score']}  {r['en']:20s} <-> {r['fr']:20s}  hops>={r['min_hops']}")


# ---------------------------------------------------------------- 3. short chain evidence

def short_chain_evidence(data: Data, out: list[str]):
    """Extract and rank 2-hop chains as strongest transfer evidence."""
    out.append("\n" + "=" * 72)
    out.append("METHOD 3: SHORT CHAIN (2-HOP) EVIDENCE")
    out.append("=" * 72)

    chains_s = load_tsv("chain-web-S.tsv")
    if not chains_s:
        out.append("  (no chain-web-S.tsv found)")
        return

    two_hop = []
    for c in chains_s:
        if int(c["hops"]) != 2:
            continue
        edges = data.score_chain_edges(c["chain"])
        sound_scores = [s for lab, s, _ in edges if lab == "≈"]
        meaning_scores = [s for lab, s, _ in edges if lab != "≈"]

        two_hop.append(dict(
            src=c["src"], dst=c["dst"],
            quality=round(float(c["quality"]), 4),
            sound_edge_score=round(sound_scores[0], 4) if sound_scores else 0.0,
            is_multiword_src=" " in c["src"],
            is_multiword_dst=" " in c["dst"],
            chain=c["chain"]))

    two_hop.sort(key=lambda r: -r["sound_edge_score"])

    write_tsv("new-short-chain-evidence.tsv", two_hop,
              ["src", "dst", "quality", "sound_edge_score",
               "is_multiword_src", "is_multiword_dst", "chain"])

    out.append(f"\n2-hop S-tier chains: {len(two_hop)}")
    mw_src = sum(1 for t in two_hop if t["is_multiword_src"])
    mw_dst = sum(1 for t in two_hop if t["is_multiword_dst"])
    out.append(f"  Multiword source: {mw_src}")
    out.append(f"  Multiword dest:   {mw_dst}")

    out.append(f"\n--- Top 20 by sound-edge score ---")
    for r in two_hop[:20]:
        mw = "*" if r["is_multiword_src"] or r["is_multiword_dst"] else " "
        out.append(f"  [{mw}] snd={r['sound_edge_score']:.3f}  {r['src']:22s} -> {r['dst']:22s}  {r['chain']}")

    # Unique src->dst pairs
    unique_transfers = {}
    for r in two_hop:
        key = (r["src"], r["dst"])
        if key not in unique_transfers or r["sound_edge_score"] > unique_transfers[key]["sound_edge_score"]:
            unique_transfers[key] = r
    out.append(f"\nUnique src->dst transfer pairs: {len(unique_transfers)}")

    # EN words reachable via 2-hop
    src_words = {r["src"] for r in two_hop}
    out.append(f"EN words with at least one 2-hop chain: {len(src_words)}")


# ---------------------------------------------------------------- 4. deep fragment composition

def deep_fragments(data: Data, out: list[str]):
    """Explore 3-6 chunk fragment compositions, EXACT IPA matching only.

    No fuzzy / edit-distance matching: a composition counts only when the
    concatenated FR-chunk IPA is an EXACT pronunciation of a real FR word.
    The honest result is that exact concatenation is sparse — which is the
    whole reason the 11-12 June schema routes growth through RE-MINING with
    phonetic_decoder.py (Lexique trie + beam search under the matcher's
    learned equivalence-floored costs), not through relaxed string matching.
    """
    out.append("\n" + "=" * 72)
    out.append("METHOD 4: DEEP FRAGMENT COMPOSITION (3-6 CHUNKS, EXACT ONLY)")
    out.append("=" * 72)

    try:
        from wordfreq import zipf_frequency
    except ImportError:
        out.append("  (wordfreq not available — skipping)")
        return

    # Build fragment inventory: EN IPA chunk -> FR IPA chunk(s)
    frag_rows = load_tsv("fragments.tsv")
    if not frag_rows:
        out.append("  (no fragments.tsv found)")
        return

    # Index: en_ipa_chunk -> [(fr_ipa_chunk, count, examples)]
    chunk_map: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for row in frag_rows:
        chunk_map[row["en_chunk"]].append(
            (row["fr_chunk"], int(row["count"]), row["examples"]))

    out.append(f"\nFragment inventory: {len(frag_rows)} chunks, "
               f"{len(chunk_map)} unique EN chunks")

    # Build EN IPA index for target words
    en_words_ipa: dict[str, str] = {}
    existing_en = set()
    for e in data.entries:
        ipa = e.get("en_ipa", "")
        if ipa:
            en_words_ipa[e["en"]] = ipa
        if e.get("usable_for_composition"):
            existing_en.add(e["en"])

    # Build FR IPA index for EXACT verification only.
    fr_ipa_to_word: dict[str, list[str]] = defaultdict(list)
    for e in data.entries:
        fr_ipa = e.get("fr_ipa", "")
        if fr_ipa:
            fr_ipa_to_word[fr_ipa].append(e["fr"])

    # Get common EN words not yet in dictionary
    common_en = set()
    for word in en_words_ipa:
        if zipf_frequency(word, "en") >= 3.0:
            common_en.add(word)
    target_words = common_en - existing_en
    also_existing = common_en & existing_en

    out.append(f"Common EN words (zipf >= 3.0): {len(common_en)}")
    out.append(f"Already in dictionary (usable): {len(also_existing)}")
    out.append(f"Target words (common but not usable): {len(target_words)}")

    # Try to compose each target word's IPA from 2-6 fragments
    en_chunks = sorted(chunk_map.keys(), key=len, reverse=True)

    def decompose(ipa: str, max_chunks: int = 6) -> list[list[str]]:
        """Find all ways to decompose IPA into fragment chunks, up to max_chunks."""
        results = []

        def _search(remaining: str, path: list[str], depth: int):
            if not remaining:
                if len(path) >= 2:
                    results.append(list(path))
                return
            if depth >= max_chunks:
                return
            if len(results) >= 50:
                return
            for chunk in en_chunks:
                if remaining.startswith(chunk):
                    path.append(chunk)
                    _search(remaining[len(chunk):], path, depth + 1)
                    path.pop()

        _search(ipa, [], 0)
        return results

    def compose_fr(en_chunks_list: list[str]) -> list[tuple[str, float]]:
        """For a list of EN IPA chunks, find FR IPA compositions with quality."""
        if not en_chunks_list:
            return []
        # Get FR options for each chunk
        options_per_chunk = []
        for chunk in en_chunks_list:
            frs = chunk_map.get(chunk, [])
            if not frs:
                return []
            options_per_chunk.append(frs)

        # Take best FR chunk for each position (by count)
        compositions = []
        best_per_pos = [max(opts, key=lambda x: x[1]) for opts in options_per_chunk]
        fr_ipa = "".join(fr for fr, _, _ in best_per_pos)
        min_count = min(c for _, c, _ in best_per_pos)
        avg_count = sum(c for _, c, _ in best_per_pos) / len(best_per_pos)
        compositions.append((fr_ipa, min_count, avg_count))
        return compositions

    # Run composition on target words — EXACT IPA matching only.
    novel_compositions = []
    tested = 0
    for word in sorted(target_words):
        ipa = en_words_ipa.get(word, "")
        if not ipa or len(ipa) < 3:
            continue
        tested += 1
        if tested > 3000:
            break

        decomps = decompose(ipa, max_chunks=6)
        for chunks in decomps:
            n_chunks = len(chunks)
            if n_chunks < 3:
                continue
            fr_options = compose_fr(chunks)
            for fr_ipa, min_count, avg_count in fr_options:
                fr_words = fr_ipa_to_word.get(fr_ipa, [])
                if fr_words:
                    for fw in fr_words[:2]:
                        novel_compositions.append(dict(
                            en=word, en_ipa=ipa, fr=fw, fr_ipa=fr_ipa,
                            n_chunks=n_chunks, chunk_recipe="+".join(chunks),
                            min_fragment_count=min_count,
                            avg_fragment_count=round(avg_count, 1),
                            en_zipf=round(zipf_frequency(word, "en"), 2),
                            fr_zipf=round(zipf_frequency(fw, "fr"), 2)))
                    break

    # Deduplicate by (en, fr)
    seen = set()
    unique = []
    for nc in novel_compositions:
        key = (nc["en"], nc["fr"])
        if key not in seen:
            seen.add(key)
            unique.append(nc)
    novel_compositions = unique

    novel_compositions.sort(key=lambda r: (-r["n_chunks"], -r["min_fragment_count"]))

    write_tsv("new-fragment-compositions.tsv", novel_compositions,
              ["en", "en_ipa", "fr", "fr_ipa", "n_chunks", "chunk_recipe",
               "min_fragment_count", "avg_fragment_count", "en_zipf", "fr_zipf"])

    # Stats by chunk count
    by_chunks = Counter(nc["n_chunks"] for nc in novel_compositions)
    out.append(f"\nTested {tested} target words for 3-6 chunk decomposition")
    out.append(f"Found {len(novel_compositions)} EXACT novel compositions:")
    for n in sorted(by_chunks):
        out.append(f"  {n} chunks: {by_chunks[n]}")

    out.append(f"\n--- Compositions by chunk count (deepest first) ---")
    for nc in novel_compositions[:30]:
        out.append(f"  {nc['n_chunks']}ch  {nc['en']:20s} ({nc['en_ipa']:15s}) -> "
                   f"{nc['fr']:20s} ({nc['fr_ipa']:15s})  "
                   f"recipe={nc['chunk_recipe']}  min_frag={nc['min_fragment_count']}")

    # Specifically highlight multiword possibilities
    # Can we find EN multiword -> FR multiword via fragments?
    out.append(f"\n--- Fragment composition depth analysis ---")
    out.append(f"Current state: all {len([e for e in data.entries if e.get('chunk_recipe')])} "
               f"fragment entries use exactly 2 chunks")
    out.append(f"This method found {by_chunks.get(3,0)} 3-chunk, "
               f"{by_chunks.get(4,0)} 4-chunk, {by_chunks.get(5,0)} 5-chunk, "
               f"{by_chunks.get(6,0)} 6-chunk compositions")
    if novel_compositions:
        avg_en_zipf = sum(nc["en_zipf"] for nc in novel_compositions) / len(novel_compositions)
        avg_fr_zipf = sum(nc["fr_zipf"] for nc in novel_compositions) / len(novel_compositions)
        out.append(f"Average EN zipf: {avg_en_zipf:.2f}, FR zipf: {avg_fr_zipf:.2f}")

    # Multiword-to-multiword exploration
    out.append(f"\n--- Multiword-to-multiword fragment bridging ---")
    out.append(f"Can we chain EN multiwords to FR multiwords via fragments?")

    # Take existing multiword usable entries and see if their IPA decomposes further
    mw_entries = [e for e in data.usable if e.get("multiword") and len(e["en"].split()) >= 2]
    mw_redecomposed = 0
    mw_examples = []
    for e in mw_entries[:200]:
        ipa = e.get("en_ipa", "")
        if not ipa:
            continue
        decomps = decompose(ipa, max_chunks=6)
        deep = [d for d in decomps if len(d) >= 3]
        if deep:
            mw_redecomposed += 1
            best = max(deep, key=len)
            fr_options = compose_fr(best)
            fr_ipa_str = "".join(fr for fr, _, _ in [max(chunk_map.get(c, [("?", 0, "")]), key=lambda x: x[1]) for c in best] if fr != "?")
            mw_examples.append(dict(
                en=e["en"], en_ipa=ipa, n_chunks=len(best),
                recipe="+".join(best),
                original_fr=e["fr"], score=e["score"]))

    out.append(f"\nExisting 2-word EN entries re-decomposed into 3+ chunks: "
               f"{mw_redecomposed} / {min(200, len(mw_entries))}")
    mw_examples.sort(key=lambda x: -x["n_chunks"])
    for ex in mw_examples[:15]:
        out.append(f"  {ex['n_chunks']}ch  {ex['en']:25s} ({ex['en_ipa']}) -> "
                   f"{ex['original_fr']:20s}  recipe={ex['recipe']}")

    out.append(f"\n>> FINDING (multi-to-multi, 5-6 words?): fragments are frozen at")
    out.append(f"   2 chunks and EXACT concatenation past that is sparse — most 3-6")
    out.append(f"   chunk recipes produce an IPA string no FR word exactly pronounces.")
    out.append(f"   This is NOT solved by relaxing the match (fuzzy/edit-distance is")
    out.append(f"   off the table). The 11-12 June schema already has the principled")
    out.append(f"   generalization: phonetic_decoder.py replaces pairwise/2-chunk")
    out.append(f"   blocking with a Lexique pronunciation TRIE + beam search, where")
    out.append(f"   substitution costs are floored by the matcher's LEARNED phoneme")
    out.append(f"   equivalences (learn_costs.py -> learned-costs.json) and word")
    out.append(f"   boundaries are free acoustically (MAX_WORDS controls depth).")
    out.append(f"   That is true multi-to-multi: 'remember' -> whole FR phrases, not")
    out.append(f"   chunk-pair lookups.")
    out.append(f"")
    out.append(f">> RE-MINING is the growth step (commit b5a985a's honest negative):")
    out.append(f"   promoting/relabelling certified pairs CANNOT compound — loops are")
    out.append(f"   built from edges that were already usable, so certification only")
    out.append(f"   re-labels existing topology. New edges come only from re-mining:")
    out.append(f"     1. learn costs from certified alignments   (learn_costs.py, done)")
    out.append(f"     2. re-run decode/augment with learned-cost matcher + 2.1x CMUdict")
    out.append(f"        trie:  phonetic_decoder.py --augment && --reverse   <- GROWTH")
    out.append(f"     3. new entries -> denser graph -> re-weave (weave.py)")
    out.append(f"     4. new loops certify new pairs -> goto 1")
    out.append(f"   These deep-fragment recipes are useful as EN-side SEGMENTATION")
    out.append(f"   seeds for that decoder pass, not as a standalone matcher.")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("section", nargs="?", default="all",
                        choices=["all", "graduated", "density", "shortchains", "fragments"])
    args = parser.parse_args()

    data = Data()
    out: list[str] = ["CHAIN METHODS REPORT", ""]
    run = args.section

    if run in ("all", "graduated"):
        graduated_tiers(data, out)
    if run in ("all", "density"):
        density_certification(data, out)
    if run in ("all", "shortchains"):
        short_chain_evidence(data, out)
    if run in ("all", "fragments"):
        deep_fragments(data, out)

    report = "\n".join(out)
    print(report)

    report_path = DIR / "new-methods-report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nwritten to {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
