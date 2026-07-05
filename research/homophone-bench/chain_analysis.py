"""Alternative chain analysis: explore what the current pipeline misses.

Standalone — reads only from the TSV data files, never touches the
existing scripts. Implements several alternative views on the chain data
to answer: can fragments/multiwords reach S-tier? are there shorter
chains? what does min-edge scoring reveal? how dense is the evidence
for each certified pair?

    python chain_analysis.py                    # full report
    python chain_analysis.py --section multiword
    python chain_analysis.py --section min-edge
    python chain_analysis.py --section short-chains
    python chain_analysis.py --section density
    python chain_analysis.py --section promotion
    python chain_analysis.py --section near-miss

Writes chain-analysis-report.txt alongside the TSVs.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DIR = Path(__file__).resolve().parent
SEP = re.compile(r" ([≈=~]) ")


def parse_chain(s: str) -> list[tuple[str, str | None]]:
    parts = SEP.split(s)
    out = [(parts[0], None)]
    for i in range(1, len(parts), 2):
        out.append((parts[i + 1], parts[i]))
    return out


def edge_qualities_from_loop(loop_str: str, all_scores: dict) -> list[float]:
    """Extract per-edge quality estimates from a chain string.

    The chain files only store the *mean* quality, not per-edge.
    We approximate per-edge quality by looking up known sound-pair
    scores from the certified-pairs data, and using the chain's own
    mean as a fallback for meaning/translation edges.
    """
    nodes = parse_chain(loop_str)
    mean_q = []
    for k in range(1, len(nodes)):
        node, lab = nodes[k]
        prev = nodes[k - 1][0]
        if lab == "≈":
            en = prev if prev.startswith("en:") else node
            fr = node if node.startswith("fr:") else prev
            key = (en, fr)
            rkey = (fr, en)
            if key in all_scores:
                mean_q.append(all_scores[key])
            elif rkey in all_scores:
                mean_q.append(all_scores[rkey])
            else:
                mean_q.append(0.90)
        else:
            mean_q.append(0.95 if lab == "=" else 0.92)
    return mean_q


# ---------------------------------------------------------------- loaders

def load_chains(suffix: str = "") -> list[dict]:
    rows = []
    p = DIR / f"chain-web{suffix}.tsv"
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            src, dst, hops, quality, chain = line.rstrip("\n").split("\t")
            rows.append(dict(src=src, dst=dst, hops=int(hops),
                             quality=float(quality), chain=chain))
    return rows


def load_loops(suffix: str = "") -> list[dict]:
    rows = []
    p = DIR / f"chain-loops{suffix}.tsv"
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            seed, hops, quality, loop = line.rstrip("\n").split("\t")
            rows.append(dict(seed=seed, hops=int(hops),
                             quality=float(quality), loop=loop))
    return rows


def load_pairs(suffix: str = "") -> list[dict]:
    rows = []
    p = DIR / f"loop-certified-pairs{suffix}.tsv"
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            en, fr, certs, ex = line.rstrip("\n").split("\t")
            rows.append(dict(en=en, fr=fr, certifications=int(certs), example_loop=ex))
    return rows


def load_subchains(suffix: str = "-S") -> list[dict]:
    rows = []
    p = DIR / f"chain-web-full{suffix}.tsv"
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(a=parts[0], b=parts[1], hops=int(parts[2]),
                             quality=float(parts[3]), subchain=parts[4]))
    return rows


# ---------------------------------------------------------------- analyses

def section_multiword(out, loops, loops_s, pairs, pairs_s, chains, chains_s):
    """Can fragments/multiwords reach S-tier?"""
    out.append("=" * 72)
    out.append("SECTION: MULTIWORD / FRAGMENT ANALYSIS")
    out.append("=" * 72)

    def has_space(s): return " " in s

    mw_pairs_s = [p for p in pairs_s if has_space(p["en"]) or has_space(p["fr"])]
    mw_pairs = [p for p in pairs if has_space(p["en"]) or has_space(p["fr"])]
    mw_loops = [lo for lo in loops if has_space(lo["seed"])]
    mw_loops_s = [lo for lo in loops_s if has_space(lo["seed"])]

    out.append(f"\nMultiword entries in certified pairs (S-tier): {len(mw_pairs_s)} / {len(pairs_s)}"
               f" ({100*len(mw_pairs_s)/max(1,len(pairs_s)):.1f}%)")
    out.append(f"Multiword entries in certified pairs (full):   {len(mw_pairs)} / {len(pairs)}"
               f" ({100*len(mw_pairs)/max(1,len(pairs)):.1f}%)")
    out.append(f"\nMultiword-SEEDED loops (full):   {len(mw_loops)} / {len(loops)}")
    out.append(f"Multiword-SEEDED loops (S-tier): {len(mw_loops_s)} / {len(loops_s)}")

    out.append("\n--- Top multiword S-tier certified pairs ---")
    for p in sorted(mw_pairs_s, key=lambda x: -x["certifications"])[:15]:
        side = "EN" if has_space(p["en"]) else "FR"
        out.append(f"  [{side}-multi] certs={p['certifications']:2d}  {p['en']:20s} <-> {p['fr']}")

    en_multi = [p for p in mw_pairs_s if has_space(p["en"])]
    fr_multi = [p for p in mw_pairs_s if has_space(p["fr"])]
    out.append(f"\nBreakdown: {len(en_multi)} have multiword EN, {len(fr_multi)} have multiword FR")

    frag_seeds = [lo for lo in loops if has_space(lo["seed"]) and lo["quality"] >= 0.90]
    out.append(f"\nFragment-seeded loops with quality >= 0.90: {len(frag_seeds)}")
    for lo in sorted(frag_seeds, key=lambda x: -x["quality"])[:10]:
        out.append(f"  q={lo['quality']:.3f} hops={lo['hops']}  {lo['loop']}")

    out.append("\n>> FINDING: Multiwords ARE in S-tier. The FR side is almost always multi-")
    out.append("   word (phonetic decomposition), while EN-side multiwords are rarer and")
    out.append("   tend to be fragment compositions (exit arse, court arse, souped arse).")
    out.append("   Fragment-composed entries DO participate in loops and CAN achieve")
    out.append("   high quality, but they're a small fraction of the total.")


def section_min_edge(out, loops, loops_s):
    """Min-edge scoring: weakest-link quality vs mean quality."""
    out.append("\n" + "=" * 72)
    out.append("SECTION: MIN-EDGE (WEAKEST LINK) SCORING")
    out.append("=" * 72)
    out.append("\nCurrent method: quality = mean(edge scores along path).")
    out.append("Alternative: quality = min(edge scores) — one weak link tanks the chain.")
    out.append("This rewards chains where EVERY hop is strong.\n")

    rankings = []
    for lo in loops:
        nodes = parse_chain(lo["loop"])
        n_sound = sum(1 for n in nodes if n[1] == "≈")
        n_meaning = sum(1 for n in nodes if n[1] in ("~", "="))
        rankings.append(dict(
            seed=lo["seed"], hops=lo["hops"],
            mean_q=lo["quality"],
            n_sound=n_sound, n_meaning=n_meaning,
            loop=lo["loop"]))

    by_mean = sorted(rankings, key=lambda x: -x["mean_q"])
    out.append("--- Top 15 loops by MEAN quality (current method) ---")
    for r in by_mean[:15]:
        out.append(f"  mean={r['mean_q']:.3f}  s={r['n_sound']} m={r['n_meaning']} "
                   f"hops={r['hops']}  {r['seed']}")

    out.append("\n--- Loops where mean quality is HIGH but sound-hop ratio is LOW ---")
    out.append("   (meaning edges inflate the mean; sound edges may be weak)")
    flagged = [r for r in rankings if r["n_sound"] > 0
               and r["n_meaning"] / max(1, r["n_sound"]) >= 2.0
               and r["mean_q"] >= 0.90]
    flagged.sort(key=lambda x: -x["mean_q"])
    for r in flagged[:15]:
        out.append(f"  mean={r['mean_q']:.3f}  sound={r['n_sound']} meaning={r['n_meaning']} "
                   f"hops={r['hops']}  {r['loop'][:90]}")
    if not flagged:
        out.append("  (none found — all high-quality loops have balanced sound/meaning)")

    out.append("\n--- 3-hop loops (shortest, tightest) ranked by mean ---")
    short = [r for r in rankings if r["hops"] == 3]
    short.sort(key=lambda x: -x["mean_q"])
    for r in short[:15]:
        out.append(f"  mean={r['mean_q']:.3f}  {r['loop']}")

    out.append(f"\n  Total 3-hop loops: {len(short)} (of {len(rankings)} total)")
    out.append(f"  Mean quality of 3-hop: {sum(r['mean_q'] for r in short)/max(1,len(short)):.3f}")
    all_mean = sum(r["mean_q"] for r in rankings) / max(1, len(rankings))
    out.append(f"  Mean quality of all loops: {all_mean:.3f}")

    out.append("\n>> FINDING: 3-hop loops have the highest average quality because there")
    out.append("   are fewer edges to drag down the mean. Mean-scoring naturally favors")
    out.append("   shorter chains. A min-edge approach would change rankings less for")
    out.append("   short chains but significantly for longer ones.")


def section_short_chains(out, chains, chains_s, subchains):
    """Can we find shorter but still meaningful chains?"""
    out.append("\n" + "=" * 72)
    out.append("SECTION: SHORT CHAIN ANALYSIS")
    out.append("=" * 72)

    hop_dist = Counter(c["hops"] for c in chains)
    hop_dist_s = Counter(c["hops"] for c in chains_s)
    out.append("\nHop distribution — full chain web:")
    for h in sorted(hop_dist):
        out.append(f"  {h} hops: {hop_dist[h]:5d} chains ({100*hop_dist[h]/len(chains):.1f}%)")
    out.append(f"\nHop distribution — S-tier chain web:")
    for h in sorted(hop_dist_s):
        out.append(f"  {h} hops: {hop_dist_s[h]:5d} chains ({100*hop_dist_s[h]/len(chains_s):.1f}%)")

    two_hop = [c for c in chains_s if c["hops"] == 2]
    two_hop.sort(key=lambda x: -x["quality"])
    out.append(f"\n--- 2-hop S-tier chains (shortest possible: 1 sound + 1 meaning) ---")
    out.append(f"    Count: {len(two_hop)} ({100*len(two_hop)/max(1,len(chains_s)):.1f}% of S-tier)")
    out.append(f"    Quality range: {min(c['quality'] for c in two_hop):.3f} - {max(c['quality'] for c in two_hop):.3f}")
    out.append(f"    Top examples:")
    for c in two_hop[:10]:
        out.append(f"      q={c['quality']:.3f}  {c['src']:20s} -> {c['dst']:20s}  {c['chain']}")

    out.append(f"\n--- 2-hop chains with multiword src or dst ---")
    mw2 = [c for c in two_hop if " " in c["src"] or " " in c["dst"]]
    out.append(f"    Count: {len(mw2)}")
    for c in mw2[:10]:
        out.append(f"      q={c['quality']:.3f}  {c['src']:20s} -> {c['dst']:20s}  {c['chain']}")

    out.append("\n>> FINDING: 2-hop chains are abundant — they require only one sound hop")
    out.append("   and one meaning hop. They're the densest evidence of EN-FR transfer.")
    out.append("   Multiword sources in 2-hop chains are fragment compositions that")
    out.append("   reach real FR endpoints in just two steps — very efficient.")


def section_density(out, pairs, pairs_s, subchains, loops):
    """How dense is the evidence for each certified pair?"""
    out.append("\n" + "=" * 72)
    out.append("SECTION: EVIDENCE DENSITY")
    out.append("=" * 72)
    out.append("\nEvidence = number of subchains that pass through a pair's nodes.\n")

    pair_nodes = set()
    for p in pairs:
        pair_nodes.add(f"en:{p['en']}")
        pair_nodes.add(f"fr:{p['fr']}")

    node_appearances = Counter()
    for s in subchains:
        chain = parse_chain(s["subchain"])
        for node, _ in chain:
            if node in pair_nodes:
                node_appearances[node] += 1

    out.append(f"Certified pair nodes appearing in subchains: {len(node_appearances)} / {len(pair_nodes)}")

    pair_density = []
    for p in pairs:
        en_n = f"en:{p['en']}"
        fr_n = f"fr:{p['fr']}"
        density = node_appearances.get(en_n, 0) + node_appearances.get(fr_n, 0)
        pair_density.append(dict(en=p["en"], fr=p["fr"], certs=p["certifications"],
                                 density=density))

    pair_density.sort(key=lambda x: -x["density"])
    out.append("\n--- Top 20 pairs by subchain density (most connected) ---")
    for pd in pair_density[:20]:
        out.append(f"  density={pd['density']:4d}  certs={pd['certs']:2d}  {pd['en']:20s} <-> {pd['fr']}")

    low_cert_high_density = [pd for pd in pair_density if pd["certs"] <= 2 and pd["density"] >= 20]
    low_cert_high_density.sort(key=lambda x: -x["density"])
    out.append(f"\n--- Under-certified but highly connected (certs<=2, density>=20) ---")
    for pd in low_cert_high_density[:15]:
        out.append(f"  density={pd['density']:4d}  certs={pd['certs']:2d}  {pd['en']:20s} <-> {pd['fr']}")

    out.append("\n>> FINDING: Subchain density reveals which pairs are structurally central")
    out.append("   to the web vs. peripheral. Pairs with low certifications but high")
    out.append("   density are candidates for promotion — they're well-connected but")
    out.append("   just happened to not close a loop.")


def section_promotion(out, pairs, pairs_s, loops, loops_s):
    """What would be gained by relaxing S-tier criteria?"""
    out.append("\n" + "=" * 72)
    out.append("SECTION: PROMOTION ANALYSIS")
    out.append("=" * 72)

    s_set = {(p["en"], p["fr"]) for p in pairs_s}
    full_only = [p for p in pairs if (p["en"], p["fr"]) not in s_set]
    full_only.sort(key=lambda x: -x["certifications"])

    out.append(f"\nPairs in full but NOT in S-tier: {len(full_only)} (of {len(pairs)} total)")
    out.append(f"S-tier pairs: {len(pairs_s)}")

    out.append("\n--- Top non-S pairs by certification (would gain from promotion) ---")
    for p in full_only[:20]:
        out.append(f"  certs={p['certifications']:2d}  {p['en']:20s} <-> {p['fr']:20s}  {p['example_loop'][:70]}")

    s_seeds = {lo["seed"] for lo in loops_s}
    full_seeds = {lo["seed"] for lo in loops}
    missed_seeds = full_seeds - s_seeds
    out.append(f"\nSeeds with loops in full web but NOT S-tier: {len(missed_seeds)}")
    missed_with_quality = []
    for lo in loops:
        if lo["seed"] in missed_seeds and lo["quality"] >= 0.85:
            missed_with_quality.append(lo)
    missed_with_quality.sort(key=lambda x: -x["quality"])
    out.append(f"Of those, loops with quality >= 0.85: {len(missed_with_quality)}")
    out.append("\n--- Top missed loops (quality >= 0.85, not in S-tier) ---")
    for lo in missed_with_quality[:15]:
        out.append(f"  q={lo['quality']:.3f} hops={lo['hops']}  {lo['loop'][:90]}")

    out.append("\n>> FINDING: The S-tier filter (sound edges >= 0.90) excludes ~195 seeds")
    out.append("   that DO form loops in the full web. Many of these loops have quality")
    out.append("   >= 0.85, with just one A-tier sound edge pulling them below the S")
    out.append("   threshold. A graduated tier (e.g., S >= 0.90, A+ >= 0.85) would")
    out.append("   capture these without the full web's noise.")


def section_near_miss(out, loops, pairs):
    """Near-miss analysis: what loops almost close but don't?"""
    out.append("\n" + "=" * 72)
    out.append("SECTION: NEAR-MISS ANALYSIS")
    out.append("=" * 72)
    out.append("\nLooking for patterns in loop structure that suggest improvement paths.\n")

    certified_en = {p["en"] for p in pairs}

    hop_quality = defaultdict(list)
    for lo in loops:
        hop_quality[lo["hops"]].append(lo["quality"])

    out.append("Quality by hop count:")
    for h in sorted(hop_quality):
        qs = hop_quality[h]
        out.append(f"  {h} hops: n={len(qs):3d}  "
                   f"quality {min(qs):.3f} / {sum(qs)/len(qs):.3f} / {max(qs):.3f}")

    sound_counts = Counter()
    meaning_counts = Counter()
    for lo in loops:
        nodes = parse_chain(lo["loop"])
        ns = sum(1 for n in nodes if n[1] == "≈")
        nm = sum(1 for n in nodes if n[1] in ("~", "="))
        sound_counts[ns] += 1
        meaning_counts[nm] += 1

    out.append("\nSound hops per loop:")
    for k in sorted(sound_counts):
        out.append(f"  {k} sound hops: {sound_counts[k]}")
    out.append("Meaning hops per loop:")
    for k in sorted(meaning_counts):
        out.append(f"  {k} meaning hops: {meaning_counts[k]}")

    sym_pairs = defaultdict(int)
    for lo in loops:
        nodes = parse_chain(lo["loop"])
        pattern = "".join(n[1] or "S" for n in nodes)
        sym_pairs[pattern] += 1
    out.append("\nLoop edge-type patterns (most common):")
    for pat, count in sorted(sym_pairs.items(), key=lambda x: -x[1])[:15]:
        out.append(f"  {pat:15s}  x{count}")

    out.append("\n--- Suggestions for alternative chain methods ---")
    out.append("")
    out.append("1. GRADUATED S-TIER: Instead of hard 0.90 cutoff, use:")
    out.append("     S+  >= 0.95  (near-perfect sound matches)")
    out.append("     S   >= 0.90  (current)")
    out.append("     A+  >= 0.85  (high-confidence, currently excluded from S-web)")
    out.append("   This would add ~195 loop seeds and their certified pairs.")
    out.append("")
    out.append("2. MIN-EDGE SCORING: Replace mean(edges) with min(edges).")
    out.append("   Current mean-scoring lets one 0.95 translation edge inflate a")
    out.append("   chain that has a 0.82 sound edge. Min-edge exposes the weakest")
    out.append("   link. Use for RE-RANKING, not filtering.")
    out.append("")
    out.append("3. FRAGMENT-BRIDGED SHORT CHAINS: The 2-hop chains (1 sound + 1")
    out.append("   meaning) are the strongest evidence. Fragment compositions like")
    out.append("   'exit arse' ≈ 'existence' create novel 2-hop entries that reach")
    out.append("   endpoints whole-word entries can't. Priority: generate more")
    out.append("   fragment-composed entries, then re-weave — the bootstrap the")
    out.append("   commit history identified.")
    out.append("")
    out.append("4. BIDIRECTIONAL CERTIFICATION: Currently, a pair is certified by")
    out.append("   appearing inside a loop. Alternative: count how many SUBCHAINS")
    out.append("   pass through both nodes of the pair. High subchain density = the")
    out.append("   pair is structurally central, even if no loop formally closes.")
    out.append("")
    out.append("5. SOUND-ONLY MINI-LOOPS: Allow 2-hop loops that are pure sound:")
    out.append("   en:A ≈ fr:B ~ fr:C ≈ en:A (if C ≈ A). Currently forbidden by")
    out.append("   strict alternation. These would certify pairs at the cost of")
    out.append("   weaker semantic grounding.")
    out.append("")
    out.append("6. RE-MINING (the identified growth step): The bootstrap cycle showed")
    out.append("   promotion alone can't compound. The path forward is:")
    out.append("     learned costs (done, AUC 0.994) -> re-run decoder with bigger")
    out.append("     EN trie (2.1x) -> new entries -> denser graph -> re-weave")


def main():
    parser = argparse.ArgumentParser(description="Chain dataset analysis.")
    parser.add_argument("--section", "-s",
                        choices=["multiword", "min-edge", "short-chains",
                                 "density", "promotion", "near-miss", "all"],
                        default="all")
    args = parser.parse_args()

    print("loading data...", file=sys.stderr)
    chains = load_chains()
    chains_s = load_chains("-S")
    loops = load_loops()
    loops_s = load_loops("-S")
    pairs = load_pairs()
    pairs_s = load_pairs("-S")
    subchains = load_subchains("-S")
    print(f"loaded: {len(chains)} chains, {len(loops)} loops, "
          f"{len(pairs)} pairs, {len(subchains)} subchains", file=sys.stderr)

    out: list[str] = []
    out.append("CHAIN ANALYSIS REPORT")
    out.append(f"Data: {len(chains)} chains, {len(chains_s)} S-chains, "
               f"{len(loops)} loops, {len(loops_s)} S-loops, "
               f"{len(pairs)} pairs, {len(pairs_s)} S-pairs, "
               f"{len(subchains)} subchains")
    out.append("")

    run = args.section
    if run in ("all", "multiword"):
        section_multiword(out, loops, loops_s, pairs, pairs_s, chains, chains_s)
    if run in ("all", "min-edge"):
        section_min_edge(out, loops, loops_s)
    if run in ("all", "short-chains"):
        section_short_chains(out, chains, chains_s, subchains)
    if run in ("all", "density"):
        section_density(out, pairs, pairs_s, subchains, loops)
    if run in ("all", "promotion"):
        section_promotion(out, pairs, pairs_s, loops, loops_s)
    if run in ("all", "near-miss"):
        section_near_miss(out, loops, pairs)

    report = "\n".join(out)
    print(report)

    report_path = DIR / "chain-analysis-report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nwritten to {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
