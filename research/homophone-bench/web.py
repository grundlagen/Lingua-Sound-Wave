"""The mapping centre: one graph over EN and FR words with two edge types.

  sound edges    usable v5 entries (weight = phonetic score; cognate edges
                 carry cognate=true so an "alternate version" can include or
                 exclude them — kept separate by type, as requested)
  meaning edges  MUSE bilingual dictionary (en<->fr translations)

Walking the graph alternating edge types produces semantic-phonetic
permutations: an English word's French sound-twin has *meanings*, whose
English translations are what the word "sounds like it means" in French —
and longer walks return synonym-like echoes.

Outputs:
  mapping-web.json   the full graph (nodes, typed weighted edges)
  web-demo.txt       sample traversals
Usage: python web.py [seed words...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

entries = json.load(open("dictionary-v5.json"))

# ---- build edges ----
sound = defaultdict(list)      # node -> [(node, score, cognate)]
for e in entries:
    if not e.get("usable_for_composition"):
        continue
    a, b = f"en:{e['en']}", f"fr:{e['fr']}"
    sound[a].append((b, e["score"], bool(e.get("cognate"))))
    sound[b].append((a, e["score"], bool(e.get("cognate"))))

meaning = defaultdict(set)     # en:w <-> fr:w from MUSE
try:
    with open("/tmp/muse-en-fr.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                meaning[f"en:{parts[0]}"].add(f"fr:{parts[1]}")
                meaning[f"fr:{parts[1]}"].add(f"en:{parts[0]}")
except FileNotFoundError:
    print("no MUSE file; meaning edges empty", file=sys.stderr)

nodes = set(sound) | set(meaning)
print(f"web: {len(nodes)} nodes, "
      f"{sum(len(v) for v in sound.values()) // 2} sound edges, "
      f"{sum(len(v) for v in meaning.values()) // 2} meaning edges",
      file=sys.stderr)

with open("mapping-web.json", "w") as f:
    json.dump({
        "sound": {k: [[n, s, c] for n, s, c in v] for k, v in sound.items()},
        "meaning": {k: sorted(v) for k, v in meaning.items()},
    }, f, ensure_ascii=False)


def walk(start: str, max_hops: int = 4, beam: int = 12):
    """Alternate sound/meaning steps; return EN words reached, with paths."""
    results = []
    frontier = [(1.0, start, [start], None)]   # (conf, node, path, last_type)
    visited = {start}
    for _hop in range(max_hops):
        nxt = []
        for conf, node, path, last in frontier:
            if last != "sound":
                for n, s, cog in sorted(sound.get(node, []), key=lambda x: -x[1])[:6]:
                    if n not in visited:
                        visited.add(n)
                        tag = "≈cog" if cog else "≈"
                        nxt.append((conf * s, n, path + [f"{tag}{n}"], "sound"))
            if last != "meaning":
                for n in list(meaning.get(node, []))[:6]:
                    if n not in visited:
                        visited.add(n)
                        nxt.append((conf * 0.95, n, path + [f"={n}"], "meaning"))
        nxt.sort(key=lambda x: -x[0])
        frontier = nxt[:beam * 4]
        for conf, node, path, last in frontier:
            if node.startswith("en:") and node != start and last == "meaning":
                results.append((conf, node[3:], " ".join(path)))
    results.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for conf, w, path in results:
        if w in seen:
            continue
        seen.add(w)
        out.append((conf, w, path))
        if len(out) >= beam:
            break
    return out


def main():
    seeds = sys.argv[1:] or ["sea", "pain", "more", "two", "bell", "share"]
    lines = []
    for s in seeds:
        lines.append(f"\n=== en:{s} — what it sounds like it means ===")
        for conf, w, path in walk(f"en:{s}"):
            lines.append(f"  {conf:.2f}  {w:16s} {path}")
    text = "\n".join(lines)
    print(text)
    with open("web-demo.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
