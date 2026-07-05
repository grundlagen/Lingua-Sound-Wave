# Fine / mixed generation inventory — built and measured

`fine_inventory.py` builds three generation pools and ranks all output by the
matcher arbiter: chunks-only (current), single-phoneme (full inventory incl.
stops p/b/t/d/k/ɡ and the EN↔FR equivalence phones), and FINE = phones + chunks.

## Result (random streams → real words in BOTH languages, arbiter-ranked)

| pool | dual-decodable yield | best combo (top carve) |
|---|---|---|
| chunks only (current) | 25/200 (12%) | 0.66 (end intend / haines traînent haine) |
| **single phonemes (full)** | **91/200 (45%)** | **0.82 (peat lee / bit lis; repel / rappelle)** |
| FINE: phones + chunks | 62/200 (31%) | 0.78 (ls loop / relais loupe) |

## The honest refinement

The **full single-phoneme inventory wins on both yield and top quality** — it
beats the mixed (fine) pool. Adding the attested chunks back *lowers* yield
(45% → 31%), because their consonant-cluster bias (st, ks, tɹ, kspl) dilutes the
pool with streams French can't carve. The chunks' theoretical naturalness benefit
did not show up in the matcher combo.

So the "fine inventory" answer is sharper than expected: for **generation**, the
right pool is the **full single-phoneme set** (every phoneme, stops and
equivalence-rule phones included), arbiter-ranked — not the chunks, and not the
mix. The chunks remain useful as a *decoding/anchoring* index (fragments.tsv) and
for naturalness a coherence model would judge, but they should not drive
generation.

## Caveat

This measures yield × sound-combo, not L2 coherence. A real coherence model might
value the chunks' attested transitions; on the axes we can measure today, pure
single-phoneme composition is best. As always, the carries' *verse* quality still
rides on the L2-coherence model (the gating component).

## Net for the inventory question

- Generation should compose at the **finest grain (single phonemes, full rule
  set)** — measured win.
- Matching already runs at single-phoneme grain.
- Chunks = anchors/index, not the generative driver.
