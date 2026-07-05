/**
 * meaning-map-atlas.ts — build the Meaning Map from the bank, hunt the chains,
 * and print the convergence atlas.
 *
 * Run it (no build, no services needed):
 *
 *   node --experimental-strip-types \
 *     artifacts/api-server/src/lib/meaning-map-atlas.ts
 *
 * or, from inside artifacts/api-server:  pnpm run meaning-map
 *
 * Everything here is deterministic and offline. Swap `defaultPhoneticScore`
 * for the LLM `phonemeChainScore` to upgrade the hunt to production quality.
 */

import {
  MeaningMap,
  defaultPhoneticScore,
  whittle,
  type MapNode,
} from "./meaning-map.ts";
import {
  WORDS,
  SEMANTIC_LINKS,
  HOMOPHONIC_LINKS,
} from "./meaning-bank.ts";

/** Assemble the graph: vocabulary, then the three edge layers. */
export function buildMap(): MeaningMap {
  const map = new MeaningMap();

  // 1. nodes
  for (const w of WORDS) {
    map.addNode({ text: w.text, lang: w.lang, gloss: w.gloss, concept: w.concept });
  }

  // 2. semantic edges derived from shared concept (translations + synonyms)
  const byConcept = new Map<string, MapNode[]>();
  for (const node of map.nodes.values()) {
    if (!node.concept) continue;
    const list = byConcept.get(node.concept) ?? [];
    list.push(node);
    byConcept.set(node.concept, list);
  }
  for (const group of byConcept.values()) {
    for (let i = 0; i < group.length; i += 1) {
      for (let j = i + 1; j < group.length; j += 1) {
        const a = group[i];
        const b = group[j];
        const sameLang = a.lang === b.lang;
        // cross-language = literal translation (1.0); same-language = synonym (0.9)
        map.addEdge(a.id, b.id, "semantic", sameLang ? 0.9 : 1.0, sameLang ? "synonym" : "translation");
      }
    }
  }

  // 3. the synonym/sense layer on top (near-synonyms across concepts)
  for (const link of SEMANTIC_LINKS) {
    map.addEdge(link.a, link.b, "semantic", link.weight, link.reason);
  }

  // 4. curated homophonic floor
  for (const link of HOMOPHONIC_LINKS) {
    map.addEdge(link.a, link.b, "homophonic", link.weight, link.reason);
  }

  // 5. the hunt: auto-mine more sound-alikes across the whole vocabulary
  const mined = map.mineHomophonic(defaultPhoneticScore, 0.6);
  if (process.env.MEANING_MAP_VERBOSE) {
    console.error(`[mine] added ${mined} auto-phonetic edges`);
  }

  return map;
}

function label(map: MeaningMap, nodeId: string): string {
  const n = map.nodes.get(nodeId);
  return n ? `${n.lang}:${n.text}` : nodeId;
}

function pathStr(map: MeaningMap, steps: { from: string; to: string; weight: number }[]): string {
  if (steps.length === 0) return "(self)";
  const parts = [label(map, steps[0].from)];
  for (const s of steps) parts.push(`→ ${label(map, s.to)} [${s.weight.toFixed(2)}]`);
  return parts.join(" ");
}

function main(): void {
  const map = buildMap();
  console.log("╔══════════════════════════════════════════════════════════════╗");
  console.log("║   THE MEANING MAP — convergence atlas (EN ⇄ FR)               ║");
  console.log("╚══════════════════════════════════════════════════════════════╝");
  console.log(
    `nodes: ${map.nodes.size}  semantic edges: ${map.countEdges("semantic")}  homophonic edges: ${map.countEdges("homophonic")}\n`,
  );

  const convergences = map.findConvergences();
  console.log(`── CONVERGENCES (sound-path ∧ meaning-path agree) — ${convergences.length} found ──\n`);
  for (const c of convergences) {
    const tag = c.kind === "cognate" ? "◆ cognate   " : "◇ transitive";
    console.log(
      `${tag}  ${label(map, c.from)}  ≈⇔  ${label(map, c.to)}   combined=${c.combined.toFixed(2)} (mean=${c.meaning.toFixed(2)} sound=${c.sound.toFixed(2)})`,
    );
    if (c.kind === "transitive" || c.meaningPath.length > 1) {
      console.log(`              meaning: ${pathStr(map, c.meaningPath)}`);
      console.log(`              sound  : ${pathStr(map, c.soundPath)}`);
    }
  }

  const sirens = map.findSirens(0.6);
  console.log(`\n── SIRENS (sound says yes, sense says no) — ${sirens.length} flagged ──\n`);
  for (const s of sirens) {
    console.log(`  ⚠  ${label(map, s.from)}  ≈  ${label(map, s.to)}   sound=${s.sound.toFixed(2)}  — ${s.note}`);
  }

  console.log(`\n── WHITTLING (the atlas converging over rising thresholds) ──\n`);
  const rounds = whittle(convergences);
  for (const r of rounds) {
    const names = r.survivors.map((c) => `${label(map, c.from)}≈${label(map, c.to)}`).join(", ");
    console.log(`  θ≥${r.threshold.toFixed(2)} → ${r.survivors.length.toString().padStart(2)} survive : ${names}`);
  }

  const gold = rounds[rounds.length - 1]?.survivors ?? [];
  console.log(`\n★ GOLD CORE — the map's most-perfect homophonic translations:`);
  for (const c of gold) {
    console.log(`    ${label(map, c.from)}  =  ${label(map, c.to)}   (${c.combined.toFixed(2)})`);
  }
  console.log("");
}

// run only when invoked directly
const invokedDirectly =
  typeof process !== "undefined" &&
  Array.isArray(process.argv) &&
  /meaning-map-atlas\.ts$/.test(process.argv[1] ?? "");
if (invokedDirectly) main();

export { main as runAtlas };
