"""Typed mapping web for v5 sound and meaning walks.

Sound, fragment, and meaning edges stay separate. Walks can therefore say
whether a result is a pure sound echo, a semantic echo, or a sound->meaning
permutation instead of collapsing everything into one mushy score.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ']+")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if ch.isalnum() and not unicodedata.combining(ch))


def tokens(text: str) -> list[str]:
    return [m.group(0).lower().strip("'") for m in TOKEN_RE.finditer(text)]


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def add_node(nodes: dict[str, dict], node_id: str, label: str, kind: str) -> None:
    nodes.setdefault(node_id, {"id": node_id, "label": label, "kind": kind})


def load_fragments() -> list[dict]:
    with open("fragments.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        row["count"] = int(row["count"])
    return rows


def find_muse_files() -> dict:
    roots = [Path.cwd(), Path.home() / "Downloads"]
    generated_names = {"muse-status.json", "mapping-web.json", "mapping-walks.tsv"}
    found = []
    for root in roots:
        if not root.exists():
            continue
        pattern = "**/*" if root == Path.cwd() else "*"
        for path in root.glob(pattern):
            if not path.is_file() or path.name in generated_names:
                continue
            if "muse" in path.name.lower() or "nemotron" in path.name.lower():
                found.append(str(path))
    return {
        "searched": [str(x) for x in roots],
        "found": found,
        "muse_embedding_found": any("muse" in Path(x).name.lower() for x in found),
        "note": "No MUSE-named embedding/file found; Nemotron PDF is tracked as an external reference if present.",
    }


def main() -> None:
    entries = json.load(open("dictionary-v5.json", encoding="utf-8"))
    fragments = load_fragments()
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    sound_by_en: dict[str, list[dict]] = defaultdict(list)
    meaning_from: dict[str, list[dict]] = defaultdict(list)

    def edge(edge_type: str, source: str, target: str, **payload) -> dict:
        item = {"type": edge_type, "source": source, "target": target, **payload}
        edges.append(item)
        if edge_type == "meaning_edge":
            meaning_from[source].append(item)
        return item

    for i, entry in enumerate(entries):
        if entry.get("direction", "en_fr") != "en_fr":
            continue
        en_id = f"en:{entry.get('en')}"
        fr_id = f"fr:{entry.get('fr')}"
        add_node(nodes, en_id, entry.get("en", ""), "en_word")
        add_node(nodes, fr_id, entry.get("fr", ""), "fr_phrase")
        sound_edge = edge(
            "sound_edge",
            en_id,
            fr_id,
            id=f"sound:{i}",
            score=float(entry.get("score", 0.0)),
            tier=entry.get("tier", ""),
            usable=bool(entry.get("usable_for_composition")),
            source_stage=entry.get("source_stage", "v5.1_reviewed"),
            chunk_recipe=entry.get("chunk_recipe", ""),
        )
        if sound_edge["usable"]:
            sound_by_en[en_id].append(sound_edge)

        en_norm = norm(entry.get("en", ""))
        fr_norms = [norm(t) for t in tokens(entry.get("fr", ""))]
        flags = [name for name in ("cognate", "loanword") if entry.get(name)]
        surface = any(en_norm == fr_norm for fr_norm in fr_norms) or any(similarity(en_norm, fr_norm) >= 0.82 for fr_norm in fr_norms)
        if flags or (surface and len(en_norm) >= 4):
            weight = 0.95 if "cognate" in flags else 0.88 if "loanword" in flags else 0.72
            reasons = flags or ["surface_similarity"]
            edge("meaning_edge", en_id, fr_id, id=f"meaning:{i}:fwd", meaning_weight=weight, reasons=reasons)
            edge("meaning_edge", fr_id, en_id, id=f"meaning:{i}:rev", meaning_weight=weight, reasons=reasons)

    for i, frag in enumerate(fragments):
        if frag["count"] < 2:
            continue
        en_id = f"sound:en:{frag['en_chunk']}"
        fr_id = f"sound:fr:{frag['fr_chunk']}"
        add_node(nodes, en_id, frag["en_chunk"], "en_sound_chunk")
        add_node(nodes, fr_id, frag["fr_chunk"], "fr_sound_chunk")
        edge("fragment_edge", en_id, fr_id, id=f"fragment:{i}", count=frag["count"], examples=frag["examples"])

    for sound_edges in sound_by_en.values():
        sound_edges.sort(key=lambda e: (-e["score"], e["target"]))

    walks = []
    for first in edges:
        if first["type"] != "sound_edge" or not first.get("usable"):
            continue
        for meaning in meaning_from.get(first["target"], []):
            echo_en = meaning["target"]
            next_sound = (sound_by_en.get(echo_en) or [None])[0]
            walks.append({
                "start": first["source"],
                "sound_fr": first["target"],
                "echo_en": echo_en,
                "echo_fr": next_sound["target"] if next_sound else "",
                "sound_score": first["score"],
                "meaning_weight": meaning["meaning_weight"],
                "next_sound_score": next_sound["score"] if next_sound else 0.0,
                "walk_type": "sound->meaning->sound" if next_sound else "sound->meaning",
                "trace": " -> ".join(x for x in [first["source"], first["target"], echo_en, next_sound["target"] if next_sound else ""] if x),
            })
    walks.sort(key=lambda w: (-(w["sound_score"] + w["meaning_weight"] + w["next_sound_score"]), w["trace"]))
    walks = walks[:500]

    muse_status = find_muse_files()
    graph = {
        "schema": "typed_mapping_web.v1",
        "muse_status": muse_status,
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "sound_edges": sum(1 for e in edges if e["type"] == "sound_edge"),
            "fragment_edges": sum(1 for e in edges if e["type"] == "fragment_edge"),
            "meaning_edges": sum(1 for e in edges if e["type"] == "meaning_edge"),
            "walks": len(walks),
        },
        "nodes": sorted(nodes.values(), key=lambda x: x["id"]),
        "edges": edges,
        "walks": walks,
    }
    with open("mapping-web.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    with open("mapping-walks.tsv", "w", encoding="utf-8") as f:
        f.write("walk_type\tsound_score\tmeaning_weight\tnext_sound_score\tstart\tsound_fr\techo_en\techo_fr\ttrace\n")
        for row in walks:
            f.write("\t".join(str(row[x]) for x in [
                "walk_type", "sound_score", "meaning_weight", "next_sound_score",
                "start", "sound_fr", "echo_en", "echo_fr", "trace",
            ]) + "\n")
    with open("muse-status.json", "w", encoding="utf-8") as f:
        json.dump(muse_status, f, ensure_ascii=False, indent=2)

    print(f"nodes: {graph['counts']['nodes']} edges: {graph['counts']['edges']}")
    print(f"walks: {len(walks)}")
    print(f"MUSE embedding found: {muse_status['muse_embedding_found']}")
    print("wrote mapping-web.json, mapping-walks.tsv, muse-status.json")


if __name__ == "__main__":
    main()
