"""Recursive semantic-sound poem generator over v5.

This is deliberately not a lookup composer. It uses:
  - dictionary-v5.json: whole and multiword sound rows;
  - fragments.tsv / chunk_recipe: fragment-generated provenance;
  - mapping-web.json: typed sound and meaning graph;
  - wordfreq: cheap French/English plausibility prior.

The generator runs several passes. Each pass expands theme seeds through the
typed graph, searches line templates recursively, then feeds the chosen EN/FR
tokens back into the next pass. The goal is a poem that is acoustically tied
across languages while still carrying semantic echoes in both.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from statistics import mean

from wordfreq import zipf_frequency

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ']+")

DEFAULT_THEME = "west air breeze rose bell dream prove"
DEFAULT_PATTERNS = [
    ("breeze proves rose", ["breeze", "prove3", "rose"]),
    ("bell dream", ["bell", "dream"]),
    ("west air rose", ["west", "air", "rose"]),
    ("breeze proves dream", ["breeze", "prove3", "dream"]),
]

FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "by",
    "to", "for", "with", "from", "as", "of", "is", "are", "was", "were",
}

TAG_WORDS = {
    "nature": {
        "air", "airs", "aire", "aires", "breeze", "brise", "sea", "rain",
        "rose", "roses", "west", "ouest", "light", "wind", "river", "rime",
        "rimes", "dream", "moon", "night",
    },
    "direction": {"west", "ouest", "western"},
    "flower": {"rose", "roses"},
    "sound": {"bell", "bells", "belle", "belles", "air", "airs", "rime", "rimes"},
    "dream": {"dream", "dreams", "rime", "rimes"},
    "action": {"prove", "proves", "proved", "prouve", "prouvent", "see", "sees", "set", "sets"},
    "color": {"red", "blue", "rose", "roses", "raid", "raide", "blouse"},
}


def norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if ch.isalnum() and not unicodedata.combining(ch))


def tokens(text: str) -> list[str]:
    return [m.group(0).lower().strip("'") for m in TOKEN_RE.finditer(text)]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def entry_kind(entry: dict) -> str:
    if entry.get("multiword") or " " in entry.get("fr", "") or " " in entry.get("en", ""):
        return "multi"
    return "whole"


def zipf_avg(words: list[str], lang: str) -> float:
    vals = [zipf_frequency(w, lang) for w in words if w]
    return mean(vals) if vals else 0.0


@dataclass(frozen=True)
class Candidate:
    id: str
    en: str
    fr: str
    score: float
    tier: str
    kind: str
    source_stage: str
    chunk_recipe: str
    meaning_weight: float
    seed_weight: float
    walk_weight: float
    fr_plausibility: float
    en_plausibility: float
    tags: tuple[str, ...]
    base_score: float
    why: tuple[str, ...] = field(default_factory=tuple)


def load_mapping_web() -> dict:
    return json.load(open("mapping-web.json", encoding="utf-8"))


def graph_adjacency(graph: dict) -> dict[str, list[tuple[str, float, str]]]:
    adjacency: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for edge in graph["edges"]:
        typ = edge["type"]
        if typ == "meaning_edge":
            weight = float(edge.get("meaning_weight", 0.75))
        elif typ == "sound_edge":
            weight = 0.62 * float(edge.get("score", 0.0))
        elif typ == "fragment_edge":
            weight = min(0.45, 0.18 + math.log1p(float(edge.get("count", 1))) / 20)
        else:
            continue
        adjacency[edge["source"]].append((edge["target"], weight, typ))
        if typ in {"meaning_edge", "fragment_edge"}:
            adjacency[edge["target"]].append((edge["source"], weight * 0.95, typ))
    return adjacency


def recursive_seed_weights(graph: dict, seed_words: list[str], depth: int) -> dict[str, float]:
    adjacency = graph_adjacency(graph)
    weights: dict[str, float] = {}
    frontier: list[tuple[str, float, int]] = []
    for word in seed_words:
        for prefix in ("en", "fr"):
            node = f"{prefix}:{word}"
            weights[node] = max(weights.get(node, 0.0), 1.0)
            frontier.append((node, 1.0, 0))
    while frontier:
        node, weight, dist = frontier.pop(0)
        if dist >= depth:
            continue
        for nxt, edge_weight, typ in adjacency.get(node, []):
            decay = 0.86 if typ == "meaning_edge" else 0.68 if typ == "sound_edge" else 0.50
            nweight = weight * edge_weight * decay
            if nweight > weights.get(nxt, 0.0) and nweight >= 0.08:
                weights[nxt] = nweight
                frontier.append((nxt, nweight, dist + 1))
    return weights


def meaning_tables(graph: dict) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    meaning: dict[tuple[str, str], float] = {}
    walks: dict[tuple[str, str], float] = {}
    for edge in graph["edges"]:
        if edge["type"] == "meaning_edge":
            meaning[(edge["source"], edge["target"])] = max(
                meaning.get((edge["source"], edge["target"]), 0.0),
                float(edge.get("meaning_weight", 0.0)),
            )
    for walk in graph["walks"]:
        pairs = [
            (walk["start"], walk["sound_fr"]),
            (walk["echo_en"], walk["echo_fr"]),
        ]
        value = float(walk["sound_score"]) + float(walk["meaning_weight"]) + float(walk["next_sound_score"])
        for pair in pairs:
            walks[pair] = max(walks.get(pair, 0.0), value / 3.0)
    return meaning, walks


def infer_tags(en: str, fr: str, meaning_weight: float, chunk_recipe: str) -> tuple[str, ...]:
    words = {norm(x) for x in [en, *tokens(fr)] if x}
    tags = set()
    for tag, tag_words in TAG_WORDS.items():
        if words & {norm(w) for w in tag_words}:
            tags.add(tag)
    if tags & {"nature", "flower", "sound", "dream", "color"}:
        tags.add("image")
    if meaning_weight > 0.0:
        tags.add("echo")
    if chunk_recipe:
        tags.add("fragment_generated")
    return tuple(sorted(tags))


def build_candidates(graph: dict, seed_weights: dict[str, float], limit_per_en: int = 8) -> list[Candidate]:
    entries = json.load(open("dictionary-v5.json", encoding="utf-8"))
    meaning, walks = meaning_tables(graph)
    by_en_count: dict[str, int] = defaultdict(int)
    candidates: list[Candidate] = []
    for i, entry in enumerate(entries):
        if not entry.get("usable_for_composition") or entry.get("direction", "en_fr") != "en_fr":
            continue
        en = entry.get("en", "")
        fr = entry.get("fr", "")
        if not en or not fr or en in FUNCTION_WORDS:
            continue
        if not re.fullmatch(r"[a-z][a-z']+", en):
            continue
        if by_en_count[en] >= limit_per_en:
            continue
        en_id = f"en:{en}"
        fr_id = f"fr:{fr}"
        sound_score = float(entry.get("score", 0.0))
        meaning_weight = max(meaning.get((en_id, fr_id), 0.0), meaning.get((fr_id, en_id), 0.0))
        walk_weight = walks.get((en_id, fr_id), 0.0)
        seed_weight = max(seed_weights.get(en_id, 0.0), seed_weights.get(fr_id, 0.0))
        fr_words = tokens(fr)
        en_words = tokens(en)
        fr_plaus = min(1.0, zipf_avg(fr_words, "fr") / 6.0)
        en_plaus = min(1.0, zipf_avg(en_words, "en") / 6.0)
        tags = infer_tags(en, fr, meaning_weight, entry.get("chunk_recipe", ""))
        if not tags and seed_weight < 0.12 and meaning_weight <= 0.0:
            continue
        kind = entry_kind(entry)
        why = []
        if meaning_weight:
            why.append(f"meaning={meaning_weight:.2f}")
        if seed_weight:
            why.append(f"seed={seed_weight:.2f}")
        if walk_weight:
            why.append(f"walk={walk_weight:.2f}")
        if entry.get("chunk_recipe"):
            why.append(f"chunk={entry['chunk_recipe']}")
        if kind == "multi":
            why.append("multi")
        base = (
            0.34 * sound_score
            + 0.24 * min(1.0, meaning_weight)
            + 0.20 * min(1.0, seed_weight)
            + 0.10 * min(1.0, walk_weight)
            + 0.08 * fr_plaus
            + 0.04 * en_plaus
        )
        if kind == "multi":
            base += 0.035
        if entry.get("source_stage") == "v5.2_generated_validated":
            base += 0.025
        if entry.get("chunk_recipe"):
            base += 0.025
        candidate = Candidate(
            id=f"dict:{i}",
            en=en,
            fr=fr,
            score=sound_score,
            tier=entry.get("tier", ""),
            kind=kind,
            source_stage=entry.get("source_stage", "v5.1_reviewed"),
            chunk_recipe=entry.get("chunk_recipe", ""),
            meaning_weight=round(meaning_weight, 3),
            seed_weight=round(seed_weight, 3),
            walk_weight=round(walk_weight, 3),
            fr_plausibility=round(fr_plaus, 3),
            en_plausibility=round(en_plaus, 3),
            tags=tags,
            base_score=round(base, 4),
            why=tuple(why),
        )
        candidates.append(candidate)
        by_en_count[en] += 1
    candidates.sort(key=lambda c: (-c.base_score, -c.score, c.en, c.fr))
    return candidates


def role_score(candidate: Candidate, role: str) -> float:
    tags = set(candidate.tags)
    en = norm(candidate.en)
    fr_words = {norm(x) for x in tokens(candidate.fr)}
    words = {en, *fr_words}

    aliases = {
        "nature_singular": "nature",
        "sound_singular": "sound",
        "sound_plural": "sound",
        "action3": "action",
        "prove3": "action",
        "breeze": "nature",
        "bell": "sound",
        "west": "direction",
        "air": "nature",
        "rose": "flower",
    }
    base_role = aliases.get(role, role)
    score = 0.0
    if base_role in tags:
        score += 1.0
    if base_role in TAG_WORDS and words & {norm(w) for w in TAG_WORDS[base_role]}:
        score += 0.75
    if role == "nature_singular" and not candidate.en.endswith("s"):
        score += 0.20
    if role == "sound_singular" and not candidate.en.endswith("s"):
        score += 0.20
    if role == "sound_plural" and candidate.en.endswith("s"):
        score += 0.20
    if role == "action3" and candidate.en.endswith("s"):
        score += 0.30
    if role == "action3" and candidate.en in {"prove", "see", "set"}:
        score -= 0.12
    if role == "prove3":
        if candidate.en == "proves":
            score += 1.40
        elif candidate.en == "prove":
            score -= 0.80
        else:
            score -= 1.00
    if role == "breeze":
        if candidate.en == "breeze":
            score += 1.50
        elif "brise" in fr_words:
            score += 0.45
        else:
            score -= 0.65
    if role == "bell":
        if candidate.en == "bell":
            score += 1.45
        elif "belle" in fr_words or "belles" in fr_words:
            score += 0.45
        else:
            score -= 0.65
    if role == "west":
        if candidate.en == "west":
            score += 1.45
        elif "ouest" in fr_words:
            score += 0.45
        else:
            score -= 0.65
    if role == "air":
        if candidate.en == "air":
            score += 1.45
        elif fr_words & {"air", "airs", "aire", "aires"}:
            score += 0.45
        else:
            score -= 0.65
    if role == "rose":
        if candidate.en == "rose":
            score += 1.45
        elif fr_words & {"rose", "roses"}:
            score += 0.45
        else:
            score -= 0.65
    if role == "dream":
        if candidate.en == "dream":
            score += 1.25
        elif fr_words & {"rime", "rimes"}:
            score += 0.25
        if candidate.kind == "multi" and candidate.en != "dream":
            score -= 0.55
    return score


def relation_score(prev: Candidate, current: Candidate) -> float:
    if prev.en == current.en or prev.fr == current.fr:
        return -1.0
    shared_tags = set(prev.tags) & set(current.tags)
    score = 0.05 * len(shared_tags)
    if prev.fr.split()[-1][-1:] == current.fr.split()[0][:1]:
        score += 0.03
    if similarity(norm(prev.fr), norm(current.fr)) > 0.55:
        score += 0.04
    if prev.kind != current.kind:
        score += 0.04
    return score


def search_line(candidates: list[Candidate], pattern: list[str], used: set[str], beam_size: int) -> tuple[list[Candidate], float]:
    pools: list[list[Candidate]] = []
    for role in pattern:
        pool = [
            c for c in candidates
            if role_score(c, role) > 0.75
        ][:140]
        if not pool:
            pool = candidates[:140]
        pools.append(pool)

    beam: list[tuple[list[Candidate], float]] = [([], 0.0)]
    for pos, role in enumerate(pattern):
        next_beam: list[tuple[list[Candidate], float]] = []
        for seq, score in beam:
            prev = seq[-1] if seq else None
            local_used = {c.en for c in seq}
            for candidate in pools[pos]:
                if candidate.en in local_used:
                    continue
                rscore = role_score(candidate, role)
                if rscore <= 0.0:
                    continue
                transition = relation_score(prev, candidate) if prev else 0.0
                if transition <= -0.9:
                    continue
                novelty_penalty = 0.08 if candidate.en in used else 0.0
                next_beam.append((
                    seq + [candidate],
                    score + candidate.base_score + 0.13 * rscore + transition - novelty_penalty,
                ))
        next_beam.sort(key=lambda x: -x[1])
        beam = next_beam[:beam_size]
    if not beam:
        return [], 0.0
    return beam[0]


def make_poem(candidates: list[Candidate], beam_size: int) -> dict:
    lines = []
    used: set[str] = set()
    for title, pattern in DEFAULT_PATTERNS:
        seq, score = search_line(candidates, pattern, used, beam_size)
        if not seq:
            continue
        used.update(c.en for c in seq)
        lines.append({
            "pattern": title,
            "score": round(score / max(1, len(seq)), 3),
            "en": " ".join(c.en for c in seq),
            "fr": " ".join(c.fr for c in seq),
            "units": [asdict(c) for c in seq],
            "semantic_mean": round(mean(c.meaning_weight for c in seq), 3),
            "sound_mean": round(mean(c.score for c in seq), 3),
            "kinds": [c.kind for c in seq],
        })
    return {
        "en": "\n".join(line["en"] for line in lines),
        "fr": "\n".join(line["fr"] for line in lines),
        "lines": lines,
        "semantic_mean": round(mean(line["semantic_mean"] for line in lines), 3) if lines else 0.0,
        "sound_mean": round(mean(line["sound_mean"] for line in lines), 3) if lines else 0.0,
    }


def next_seed_words(poem: dict, previous: list[str], keep: int) -> list[str]:
    scored: dict[str, float] = {word: max(0.25, 1.0 - i * 0.05) for i, word in enumerate(previous)}
    for line in poem["lines"]:
        for unit in line["units"]:
            score = float(unit["meaning_weight"]) + float(unit["seed_weight"]) + float(unit["walk_weight"]) + float(unit["score"]) / 2
            for word in [unit["en"], *tokens(unit["fr"])]:
                n = norm(word)
                if len(n) > 2:
                    scored[n] = max(scored.get(n, 0.0), score)
    return [word for word, _score in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:keep]]


def write_outputs(result: dict) -> None:
    with open("recursive-poem.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open("recursive-poem.tsv", "w", encoding="utf-8") as f:
        f.write("pass\tline\tpattern\tscore\tsound_mean\tsemantic_mean\ten\tfr\tkinds\tunits\n")
        final_pass = result["passes"][-1]
        for i, line in enumerate(final_pass["poem"]["lines"], 1):
            f.write("\t".join(str(x) for x in [
                final_pass["pass"], i, line["pattern"], line["score"],
                line["sound_mean"], line["semantic_mean"], line["en"], line["fr"],
                ",".join(line["kinds"]),
                "; ".join(f"{u['en']}->{u['fr']}[{u['tier']}:{u['score']}]" for u in line["units"]),
            ]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursive poem generator over v5 sound and meaning maps.")
    parser.add_argument("--theme", default=DEFAULT_THEME, help="Seed words for the first semantic graph expansion.")
    parser.add_argument("--passes", type=int, default=3, help="Recursive generate-and-reseed passes.")
    parser.add_argument("--graph-depth", type=int, default=3, help="Typed graph expansion depth.")
    parser.add_argument("--beam", type=int, default=48, help="Beam size for each line search.")
    parser.add_argument("--seed-keep", type=int, default=14, help="Seed words kept after each pass.")
    args = parser.parse_args()

    graph = load_mapping_web()
    seed_words = [norm(w) for w in tokens(args.theme)]
    history = []
    for pass_no in range(1, args.passes + 1):
        seed_weights = recursive_seed_weights(graph, seed_words, args.graph_depth)
        candidates = build_candidates(graph, seed_weights)
        poem = make_poem(candidates, args.beam)
        history.append({
            "pass": pass_no,
            "seed_words": seed_words,
            "candidate_count": len(candidates),
            "poem": poem,
        })
        seed_words = next_seed_words(poem, seed_words, args.seed_keep)

    result = {
        "parameters": {
            "theme": args.theme,
            "passes": args.passes,
            "graph_depth": args.graph_depth,
            "beam": args.beam,
            "seed_keep": args.seed_keep,
        },
        "passes": history,
        "final": history[-1]["poem"],
    }
    write_outputs(result)

    final = result["final"]
    print(f"passes: {args.passes}")
    print(f"final sound_mean: {final['sound_mean']} semantic_mean: {final['semantic_mean']}")
    print("\nEN")
    print(final["en"])
    print("\nFR")
    print(final["fr"])
    print("\nwrote recursive-poem.json, recursive-poem.tsv")


if __name__ == "__main__":
    main()
