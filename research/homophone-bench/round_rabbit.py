"""Round-rabbit semantic/sound lattice.

This builds the structure the poem generator was missing:

1. Collapse meaning edges into semantic components.
2. From every semantic component, walk the usable sound graph outward.
3. Record how many homophonic hops away every attached string is.
4. Preserve one-to-many / many-to-one substitutions by attaching all v5 rows
   at each EN/FR node, including multiword French phrases.

The output is not a poem by itself. It is the lattice a poem generator should
use when it wants "same meaning, then N homophonic connections away, with all
strings attached".
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from statistics import mean

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ']+")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if ch.isalnum() and not unicodedata.combining(ch))


def tokens(text: str) -> list[str]:
    return [m.group(0).lower().strip("'") for m in TOKEN_RE.finditer(text)]


def node_text(node: str) -> str:
    return node.split(":", 1)[1] if ":" in node else node


def node_lang(node: str) -> str:
    return node.split(":", 1)[0] if ":" in node else ""


def entry_kind(entry: dict) -> str:
    if entry.get("multiword") or " " in entry.get("fr", "") or " " in entry.get("en", ""):
        return "multi"
    return "whole"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class Attachment:
    en: str
    fr: str
    score: float
    tier: str
    kind: str
    source_stage: str
    chunk_recipe: str


@dataclass
class LatticeRow:
    component_id: str
    semantic_strings: list[str]
    component_weight: float
    homophonic_hops: int
    node: str
    path: list[str]
    path_edge_scores: list[float]
    attached_count: int
    attachments: list[Attachment]
    best_attachment_score: float
    has_multi: bool
    has_generated: bool
    rank_score: float


def load_inputs() -> tuple[dict, list[dict]]:
    graph = json.load(open("mapping-web.json", encoding="utf-8"))
    entries = json.load(open("dictionary-v5.json", encoding="utf-8"))
    return graph, entries


def semantic_components(graph: dict) -> tuple[dict[str, list[str]], dict[str, float]]:
    uf = UnionFind()
    weights: dict[str, list[float]] = defaultdict(list)
    for edge in graph["edges"]:
        if edge["type"] != "meaning_edge":
            continue
        source, target = edge["source"], edge["target"]
        uf.union(source, target)
        weight = float(edge.get("meaning_weight", 0.0))
        weights[source].append(weight)
        weights[target].append(weight)

    components: dict[str, list[str]] = defaultdict(list)
    for node in list(uf.parent):
        components[uf.find(node)].append(node)

    component_weights: dict[str, float] = {}
    for root, nodes in components.items():
        vals = [w for node in nodes for w in weights.get(node, [])]
        component_weights[root] = round(mean(vals), 3) if vals else 0.0
    return {root: sorted(nodes) for root, nodes in components.items()}, component_weights


def fragment_bridges(graph: dict, entries: list[dict]):
    """Segmentation + fragment adjacency so the lattice can TUNNEL through
    subword chunks: en:word -> sound:en:chunk -> [fragment] -> sound:fr:chunk
    -> fr:word. This makes fragment-built routes first-class lattice paths,
    not just whole-word sound edges (the missing wiring)."""
    import matcher
    # chunk inventory from the typed web's fragment_edges
    frag_pairs = []          # (en_chunk, fr_chunk, weight, edge_id)
    en_chunks, fr_chunks = set(), set()
    for e in graph["edges"]:
        if e["type"] != "fragment_edge":
            continue
        enc = e["source"].split("sound:en:", 1)[-1]
        frc = e["target"].split("sound:fr:", 1)[-1]
        w = min(0.95, 0.6 + 0.05 * int(e.get("count", 2)))
        frag_pairs.append((enc, frc, w, e["id"]))
        en_chunks.add(enc); fr_chunks.add(frc)

    extra: dict[str, list[tuple[str, float, str]]] = defaultdict(list)

    def seg_links(word_node, ipa, chunkset, chunk_prefix):
        canon = matcher._canonical(ipa or "")
        for ch in chunkset:
            if len(ch) >= 2 and ch in canon:
                cnode = f"sound:{chunk_prefix}:{ch}"
                extra[word_node].append((cnode, 0.9, f"seg:{word_node}:{ch}"))
                extra[cnode].append((word_node, 0.9, f"seg:{cnode}"))

    for entry in entries:
        if not entry.get("usable_for_composition") or entry.get("direction", "en_fr") != "en_fr":
            continue
        seg_links(f"en:{entry.get('en','')}", entry.get("en_ipa", ""), en_chunks, "en")
        seg_links(f"fr:{entry.get('fr','')}", (entry.get("fr_ipa") or "").replace(" ", ""), fr_chunks, "fr")

    for enc, frc, w, eid in frag_pairs:
        a, b = f"sound:en:{enc}", f"sound:fr:{frc}"
        extra[a].append((b, w, f"frag:{eid}"))
        extra[b].append((a, w, f"frag:{eid}"))
    return extra


def build_sound_graph(graph: dict, with_fragments: bool = False, entries: list | None = None) -> dict[str, list[tuple[str, float, str]]]:
    adjacency: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for edge in graph["edges"]:
        if edge["type"] != "sound_edge" or not edge.get("usable"):
            continue
        source, target = edge["source"], edge["target"]
        score = float(edge.get("score", 0.0))
        adjacency[source].append((target, score, edge["id"]))
        adjacency[target].append((source, score, edge["id"]))
    if with_fragments and entries is not None:
        for node, links in fragment_bridges(graph, entries).items():
            adjacency[node].extend(links)
    for node in list(adjacency):
        adjacency[node].sort(key=lambda x: (-x[1], x[0]))
    return adjacency


def attachments_by_node(entries: list[dict]) -> dict[str, list[Attachment]]:
    out: dict[str, list[Attachment]] = defaultdict(list)
    for entry in entries:
        if not entry.get("usable_for_composition") or entry.get("direction", "en_fr") != "en_fr":
            continue
        attachment = Attachment(
            en=entry.get("en", ""),
            fr=entry.get("fr", ""),
            score=float(entry.get("score", 0.0)),
            tier=entry.get("tier", ""),
            kind=entry_kind(entry),
            source_stage=entry.get("source_stage", "v5.1_reviewed"),
            chunk_recipe=entry.get("chunk_recipe", ""),
        )
        out[f"en:{attachment.en}"].append(attachment)
        out[f"fr:{attachment.fr}"].append(attachment)
        for token in tokens(attachment.fr):
            out[f"fr:{token}"].append(attachment)
    for node in list(out):
        # De-dupe rows attached through phrase tokens.
        seen = set()
        unique = []
        for attachment in sorted(out[node], key=lambda a: (-a.score, a.en, a.fr)):
            key = (attachment.en, attachment.fr, attachment.source_stage)
            if key in seen:
                continue
            seen.add(key)
            unique.append(attachment)
        out[node] = unique
    return out


def component_matches_seed(nodes: list[str], seed_norms: set[str]) -> bool:
    if not seed_norms:
        return True
    node_norms = {norm(node_text(node)) for node in nodes}
    token_norms = {norm(token) for node in nodes for token in tokens(node_text(node))}
    return bool(seed_norms & (node_norms | token_norms))


def semantic_label(nodes: list[str]) -> list[str]:
    labels = []
    for node in nodes:
        text = node_text(node)
        labels.append(f"{node_lang(node)}:{text}")
    return labels[:16]


def bfs_component(
    component_id: str,
    nodes: list[str],
    component_weight: float,
    sound_graph: dict[str, list[tuple[str, float, str]]],
    attachments: dict[str, list[Attachment]],
    max_hops: int,
    per_component: int,
) -> list[LatticeRow]:
    queue = deque()
    best: dict[str, tuple[int, list[str], list[float]]] = {}
    for node in nodes:
        queue.append((node, 0, [node], []))
        best[node] = (0, [node], [])

    while queue:
        node, hops, path, edge_scores = queue.popleft()
        if hops >= max_hops:
            continue
        for nxt, score, edge_id in sound_graph.get(node, [])[:80]:
            next_state = (hops + 1, path + [nxt], edge_scores + [score])
            old = best.get(nxt)
            if old is not None and old[0] <= next_state[0]:
                continue
            best[nxt] = next_state
            queue.append((nxt, *next_state))

    rows: list[LatticeRow] = []
    semantic_strings = semantic_label(nodes)
    for node, (hops, path, edge_scores) in best.items():
        attached = attachments.get(node, [])
        if not attached:
            continue
        shown = attached[:12]
        best_score = max((a.score for a in attached), default=0.0)
        has_multi = any(a.kind == "multi" for a in attached)
        has_generated = any(a.source_stage == "v5.2_generated_validated" for a in attached)
        edge_mean = mean(edge_scores) if edge_scores else 1.0
        rank = (
            0.45 * component_weight
            + 0.35 * edge_mean
            + 0.15 * best_score
            + (0.04 if has_multi else 0.0)
            + (0.03 if has_generated else 0.0)
            - 0.08 * hops
        )
        rows.append(LatticeRow(
            component_id=component_id,
            semantic_strings=semantic_strings,
            component_weight=component_weight,
            homophonic_hops=hops,
            node=node,
            path=path,
            path_edge_scores=[round(x, 3) for x in edge_scores],
            attached_count=len(attached),
            attachments=shown,
            best_attachment_score=round(best_score, 3),
            has_multi=has_multi,
            has_generated=has_generated,
            rank_score=round(rank, 4),
        ))
    rows.sort(key=lambda r: (-r.rank_score, r.homophonic_hops, r.node))
    return rows[:per_component]


def write_outputs(rows: list[LatticeRow], parameters: dict) -> None:
    payload = {"parameters": parameters, "rows": [asdict(row) for row in rows]}
    with open("round-rabbit.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open("round-rabbit.tsv", "w", encoding="utf-8") as f:
        f.write(
            "component_id\tcomponent_weight\thomophonic_hops\tnode\trank_score"
            "\tsemantic_strings\tpath\tpath_edge_scores\tattached_count"
            "\thas_multi\thas_generated\tattachments\n"
        )
        for row in rows:
            f.write("\t".join(str(x) for x in [
                row.component_id,
                row.component_weight,
                row.homophonic_hops,
                row.node,
                row.rank_score,
                " | ".join(row.semantic_strings),
                " -> ".join(row.path),
                ",".join(str(x) for x in row.path_edge_scores),
                row.attached_count,
                int(row.has_multi),
                int(row.has_generated),
                "; ".join(
                    f"{a.en}->{a.fr}[{a.kind}:{a.tier}:{a.score:g}]"
                    + (f":{a.chunk_recipe}" if a.chunk_recipe else "")
                    for a in row.attachments
                ),
            ]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic component -> homophonic radius lattice.")
    parser.add_argument("--with-fragments", action="store_true", help="tunnel through subword fragment chunks")
    parser.add_argument("--seed", default="", help="Optional seed words; only matching semantic components are expanded.")
    parser.add_argument("--max-hops", type=int, default=2, help="Maximum homophonic sound-edge hops from each semantic component.")
    parser.add_argument("--components", type=int, default=20, help="Maximum semantic components to expand.")
    parser.add_argument("--per-component", type=int, default=18, help="Maximum lattice rows per component.")
    parser.add_argument("--min-component-weight", type=float, default=0.70, help="Minimum average meaning-edge weight.")
    args = parser.parse_args()

    graph, entries = load_inputs()
    components, component_weights = semantic_components(graph)
    sound_graph = build_sound_graph(graph, with_fragments=args.with_fragments, entries=entries)
    attachments = attachments_by_node(entries)
    seed_norms = {norm(token) for token in tokens(args.seed)}

    selected = []
    missing_seed_words = sorted(seed_norms)
    for component_id, nodes in components.items():
        if component_weights[component_id] < args.min_component_weight:
            continue
        if not component_matches_seed(nodes, seed_norms):
            continue
        node_norms = {norm(node_text(node)) for node in nodes}
        token_norms = {norm(token) for node in nodes for token in tokens(node_text(node))}
        missing_seed_words = [word for word in missing_seed_words if word not in node_norms and word not in token_norms]
        selected.append((component_id, nodes))

    selected.sort(key=lambda item: (-component_weights[item[0]], item[0]))
    selected = selected[:args.components]

    rows: list[LatticeRow] = []
    for component_id, nodes in selected:
        rows.extend(bfs_component(
            component_id,
            nodes,
            component_weights[component_id],
            sound_graph,
            attachments,
            args.max_hops,
            args.per_component,
        ))
    rows.sort(key=lambda r: (-r.rank_score, r.component_id, r.homophonic_hops, r.node))

    parameters = {
        "seed": args.seed,
        "missing_seed_words": missing_seed_words,
        "max_hops": args.max_hops,
        "components": args.components,
        "per_component": args.per_component,
        "min_component_weight": args.min_component_weight,
        "selected_components": len(selected),
        "rows": len(rows),
    }
    write_outputs(rows, parameters)

    print(f"selected_components: {len(selected)} rows: {len(rows)}")
    if missing_seed_words:
        print(f"seed words not found in meaning components: {', '.join(missing_seed_words)}")
    for row in rows[:12]:
        attached = "; ".join(f"{a.en}->{a.fr}" for a in row.attachments[:4])
        print(f"{row.homophonic_hops} hops {row.node:24s} score={row.rank_score:.3f} {attached}")
    print("wrote round-rabbit.json, round-rabbit.tsv")


if __name__ == "__main__":
    main()
