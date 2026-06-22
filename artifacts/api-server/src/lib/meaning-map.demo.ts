/**
 * Runnable demo / smoke test for the Meaning Map.
 *
 *   node --experimental-strip-types meaning-map.demo.ts
 *
 * It builds a tiny bilingual graph from two pair banks — a homophone bank
 * (sound-alikes) and a semantic bank (translations + synonyms) — then asks the
 * map for *convergences*: phrases reachable by BOTH a sound chain and a sense
 * chain. The data below is small and hand-built to illustrate the machinery; in
 * production the homophone edges come from the reservoir and the semantic edges
 * from a translation/synonym source.
 *
 * The two intended outcomes:
 *   1. A GEM surfaces — a French phrase that both sounds like and means an
 *      English one — found by composing chains, not by direct mining.
 *   2. A sound-only pair (sounds alike, means something unrelated) is correctly
 *      NOT reported as a gem. That rejection is the "whittling down".
 */

import { MeaningGraph, type Convergence } from "./meaning-map.ts";

const g = new MeaningGraph();

// --- Homophone bank (cross-lingual SOUND similarity) -----------------------
// EN ~ FR pairs that sound alike when spoken. Weight = phonetic similarity.
g.ingestHomophonePair({ lang: "en", text: "new" }, { lang: "fr", text: "neuf", gloss: "new / nine" }, 0.80);
g.ingestHomophonePair({ lang: "en", text: "more" }, { lang: "fr", text: "mort", gloss: "dead" }, 0.90);
g.ingestHomophonePair({ lang: "en", text: "sick" }, { lang: "fr", text: "cygne", gloss: "swan" }, 0.78);
g.ingestHomophonePair({ lang: "en", text: "sea" }, { lang: "fr", text: "si", gloss: "if / so" }, 0.88);
// a two-hop sound chain: fresh ~ frais, and frais ~ fret (freight)
g.ingestHomophonePair({ lang: "en", text: "fresh" }, { lang: "fr", text: "frais", gloss: "fresh / cool" }, 0.72);

// --- Semantic bank (same MEANING: translations + synonyms) -----------------
// Cross-lingual translations. Weight = translation confidence.
g.ingestSemanticPair({ lang: "en", text: "new" }, { lang: "fr", text: "neuf", gloss: "new" }, 0.97, "translation");
g.ingestSemanticPair({ lang: "en", text: "more" }, { lang: "fr", text: "plus", gloss: "more" }, 0.95, "translation");
g.ingestSemanticPair({ lang: "en", text: "fresh" }, { lang: "fr", text: "frais", gloss: "fresh" }, 0.93, "translation");
g.ingestSemanticPair({ lang: "en", text: "sea" }, { lang: "fr", text: "mer", gloss: "sea" }, 0.99, "translation");
// Within-language synonyms layered on top.
g.ingestSemanticPair({ lang: "en", text: "new" }, { lang: "en", text: "novel" }, 0.65, "synonym");
g.ingestSemanticPair({ lang: "en", text: "more" }, { lang: "en", text: "extra" }, 0.60, "synonym");
g.ingestSemanticPair({ lang: "fr", text: "neuf" }, { lang: "fr", text: "récent" }, 0.55, "synonym");

console.log(`graph: ${g.nodeCount} nodes, ${g.edgeCount} edges\n`);

const fmt = (c: Convergence): string => {
  const sound = c.soundPath.nodes.map((id) => g.getNode(id)!.text).join(" ~ ");
  const sense = c.sensePath.nodes.map((id) => g.getNode(id)!.text).join(" = ");
  return (
    `  ${c.origin.lang}:"${c.origin.text}"  ⇄  ${c.target.lang}:"${c.target.text}"` +
    `  [score ${c.score.toFixed(3)}, gate ${c.gate.toFixed(3)}]\n` +
    `      sound: ${sound}  (${c.soundPath.strength.toFixed(2)})\n` +
    `      sense: ${sense}  (${c.sensePath.strength.toFixed(2)})`
  );
};

const gems = g.findConvergences();
console.log(`CONVERGENCES (gems — sound AND sense meet on the same phrase): ${gems.length}`);
for (const c of gems) console.log(fmt(c));

// Sanity assertions so this doubles as a smoke test.
const newNeuf = gems.find((c) => c.target.text === "neuf" || c.origin.text === "neuf");
const moreMort = gems.find(
  (c) =>
    (c.origin.text === "more" && c.target.text === "mort") ||
    (c.origin.text === "mort" && c.target.text === "more"),
);

console.log("\nchecks:");
console.log(`  ✔ new⇄neuf surfaces as a gem (sounds like AND means): ${Boolean(newNeuf)}`);
console.log(
  `  ✔ more⇄mort rejected (sounds alike but "mort"=dead ≠ "more"): ${!moreMort}`,
);

const clusters = g.meaningClusters();
const roots = new Set(clusters.values());
console.log(`\nmeaning clusters (same-meaning groups over strong semantic edges): ${roots.size}`);

if (!newNeuf || moreMort) {
  console.error("\nDEMO FAILED: expected gem missing or sound-only pair leaked through.");
  process.exit(1);
}
console.log("\nDEMO OK");
