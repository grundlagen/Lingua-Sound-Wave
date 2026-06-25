# Webbing density of the dual-atom alphabet (vertical & horizontal)

`webbing.py` measures how the 813 loop-tiles interconnect — the precondition for
composing pages, and where the real wall is.

## The two directions

- **HORIZONTAL** — tile A followed by tile B in a line: the EN words chain AND the
  FR words chain. Advances both poems left-to-right.
- **VERTICAL** — tiles that can occupy the SAME slot (share a word, or their EN
  words are `≈` sound-twins and their FR words too): alternate rungs stacked at one
  position, giving the search room.

## The finding: horizontal both-rail chaining is ZERO

Measured on **observed** corpus bigrams (not the smoothed `cond`, which is always
>0 and falsely reported full connectivity):

| direction | result |
|---|---|
| horizontal, ≥1 rail attested | 1,263 edges, avg out-degree 1.55, **31%** of tiles followable |
| horizontal, **BOTH rails attested** | **0** edges |
| vertical (interchangeable rungs) | 3,172 edges, **~3.9 options per slot** |

There is **no** pair of dual-atoms you can set side-by-side and have the English
read as real English *and* the French read as real French at the same time. That
zero is exactly why the ladder lines scored ~0.14 fluency — the alphabet does not
self-chain on both rails.

## What this means for composition (data-driven)

1. **You cannot compose from tile-adjacency alone.** Lines must be webbed through
   **connector words** (drawn from the full graph, not the tile set) that bridge
   the rails between atoms.
2. **The connectors are identifiable.** The hub tiles are high-frequency English
   discourse words with French-fragment renderings — the natural joints:
   `suggests≈on geste`, `nonetheless≈on laisse`, `conclusion≈on clown`,
   `where≈air`, `anywhere≈un air`, `mostly≈but lie`.
3. **Vertical stacking is the usable slack** — ~3.9 alternate rungs per slot is
   what lets a search satisfy a horizontal constraint by swapping the rung.
4. **Grow the alphabet** (French-anchor the FR rail + mine more loops) to lift the
   31% / raise the chance any pair co-chains.

## Growth test (re-weave on the French-anchored augmented dictionary)

Folded 1,496 French-anchored sound pairs into the v7 dictionary and re-ran the
**unedited** weave: loops 1,109→1,222, alphabet **813→896 tiles (+10%)**.
Re-measuring webbing on the bigger alphabet:

| metric | 813 tiles | 896 tiles |
|---|---|---|
| horizontal followable | 31% | 34% |
| **both-rail attested transitions** | **0** | **1** |
| vertical rungs/slot | 3.90 | 4.01 |

The verdict is now empirical: growth helps but is **sublinear** — +10% alphabet
bought *one* gold transition. You cannot loop-mine your way to dense both-rail
chaining; you'd need orders-of-magnitude more atoms. **The connector layer is
mandatory.** Vertical stacking (~4 rungs/slot, and rising) is the dependable
resource; the dense thematic clusters (e.g. the `feel/feels/feeling ≈ file/files`
family) are the only places tiles self-bind.

## Honest reframing

The wall is not concept, it's **coverage of co-chainable rungs**. The right
architecture is: dual-atoms as anchors, a dense connector layer between them
(full-web words + the hub fragments), vertical rung-swap for flexibility, and a
real L2 model to choose among them. Tile-to-tile alone is too sparse.
