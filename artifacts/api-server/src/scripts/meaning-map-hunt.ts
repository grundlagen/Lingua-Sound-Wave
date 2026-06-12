/**
 * Meaning Map hunt — offline demonstration.
 *
 * Builds a small EN↔FR bank of MEANING edges (translations + synonyms) and
 * SOUND edges (cross-lingual sound-alikes), then asks the MeaningMap where the
 * two chains converge: phrases that both *mean* and *sound* the same across
 * the languages. Runs with no model — sound weights come from the crude
 * offline phonetic similarity. Swap in phonemeChainScore + an LLM/embedding
 * meaning judge for the real thing.
 *
 *   pnpm --filter @workspace/scripts run meaning-hunt
 *   # or: npx tsx artifacts/api-server/src/scripts/meaning-map-hunt.ts
 */

import { MeaningMap, crudePhoneticSimilarity, type Lang } from "../lib/meaning-map";

// ── Meaning bank: [enText, frText, weight, source] translations ──────────────
const TRANSLATIONS: [string, string, number, string][] = [
  // Cognates — the obvious floor of resonance (sound AND meaning coincide).
  ["table", "table", 0.98, "translation"],
  ["nation", "nation", 0.98, "translation"],
  ["important", "important", 0.97, "translation"],
  ["machine", "machine", 0.97, "translation"],
  ["animal", "animal", 0.97, "translation"],
  ["image", "image", 0.97, "translation"],
  ["village", "village", 0.96, "translation"],
  ["fruit", "fruit", 0.95, "translation"],
  ["mountain", "montagne", 0.95, "translation"],
  ["river", "rivière", 0.95, "translation"],
  ["liberty", "liberté", 0.96, "translation"],
  // Non-cognates — same meaning, different sound (no resonance on their own).
  ["happy", "heureux", 0.92, "translation"],
  ["pretty", "joli", 0.9, "translation"],
  ["sea", "mer", 0.95, "translation"],
  ["flesh", "chair", 0.93, "translation"],
  ["chair", "chaise", 0.95, "translation"],
  ["to begin", "commencer", 0.92, "translation"],
  ["to finish", "terminer", 0.92, "translation"],
  ["satisfied", "satisfait", 0.9, "translation"],
  ["water", "eau", 0.95, "translation"],
];

// ── Synonyms overlaid on top (EN↔EN and FR↔FR) — the chains that do the work ─
const SYNONYMS: [string, Lang, string, Lang, number][] = [
  // "happy" cluster: the rare latinate synonym "content" is the bridge.
  ["happy", "en", "content", "en", 0.85],
  ["content", "en", "satisfied", "en", 0.8],
  ["content", "fr", "satisfait", "fr", 0.82],
  // "to begin" cluster: English keeps a latinate twin "commence".
  ["to begin", "en", "commence", "en", 0.85],
  // "to finish" cluster: English "terminate".
  ["to finish", "en", "terminate", "en", 0.85],
  ["pretty", "en", "lovely", "en", 0.8],
  // "jolly" pulls into the happy cluster (jolly → merry → happy).
  ["jolly", "en", "merry", "en", 0.9],
  ["merry", "en", "happy", "en", 0.9],
  // A deliberately WEAK semantic adjacency bridging two clusters (happy↔pretty:
  // both positive/pleasant, but not synonyms). Sub-threshold, so it does NOT
  // fuse the clusters — it only makes them *reachable* for the hunt. This is
  // exactly what creates a frontier: jolly[happy] ≈sound≈ joli[pretty].
  ["happy", "en", "pretty", "en", 0.5],
];

// French "content" really does mean "happy/pleased" — add that translation so
// the synonym chain can close into a perfect resonance.
const EXTRA_TRANSLATIONS: [string, string, number, string][] = [
  ["content", "satisfait", 0.6, "weak-link"], // FR internal: deliberately weak — a frontier
  ["content", "content", 0.95, "translation"], // EN content ↔ FR content (pleased)
  ["commence", "commencer", 0.95, "translation"],
  ["terminate", "terminer", 0.95, "translation"],
];

// ── Sound bank: cross-lingual sound-alikes (weights computed offline) ────────
const SOUND_PAIRS: [string, Lang, string, Lang][] = [
  // cognate sound-alikes
  ["table", "en", "table", "fr"],
  ["nation", "en", "nation", "fr"],
  ["important", "en", "important", "fr"],
  ["machine", "en", "machine", "fr"],
  ["animal", "en", "animal", "fr"],
  ["image", "en", "image", "fr"],
  ["village", "en", "village", "fr"],
  ["fruit", "en", "fruit", "fr"],
  ["mountain", "en", "montagne", "fr"],
  ["river", "en", "rivière", "fr"],
  ["liberty", "en", "liberté", "fr"],
  // the hunted ones — only visible through synonym chains
  ["content", "en", "content", "fr"], // happy, via EN synonym "content"
  ["commence", "en", "commencer", "fr"], // to begin, via EN synonym "commence"
  ["terminate", "en", "terminer", "fr"], // to finish, via EN synonym "terminate"
  // frontier — nearly identical sound, meanings adjacent but not equal
  ["jolly", "en", "joli", "fr"], // merry vs pretty (even etymological cousins)
  // false friend — sounds identical, means something else (must be rejected)
  ["chair", "en", "chair", "fr"], // EN seat vs FR flesh
];

function build(): MeaningMap {
  const map = new MeaningMap({ meaningThreshold: 0.7, soundThreshold: 0.45, maxMeaningHops: 4, hopDecay: 0.65 });

  for (const [en, fr, w, src] of [...TRANSLATIONS, ...EXTRA_TRANSLATIONS]) {
    map.addMeaningEdge(en, "en", fr, "fr", w, src);
  }
  for (const [a, la, b, lb, w] of SYNONYMS) {
    map.addMeaningEdge(a, la, b, lb, w, "synonym");
  }
  for (const [a, la, b, lb] of SOUND_PAIRS) {
    const w = crudePhoneticSimilarity(a, b);
    map.addSoundEdge(a, la, b, lb, w, "sound-alike");
  }
  return map;
}

function fmt(n: number): string {
  return n.toFixed(2);
}

function chainText(map: MeaningMap, ids: string[]): string {
  return ids
    .map((id) => {
      const n = map.getNode(id);
      return n ? `${n.text}[${n.lang}]` : id;
    })
    .join(" → ");
}

function main(): void {
  const map = build();
  const s = map.stats();
  console.log("═══ The Meaning Map — hunt for sound∧meaning resonances ═══\n");
  console.log(`nodes=${s.nodes}  meaningEdges=${s.meaningEdges}  soundEdges=${s.soundEdges}  meaningClusters=${s.clusters}\n`);

  const res = map.resonances();
  const perfect = res.filter((r) => r.perfect);
  const frontier = res.filter((r) => !r.perfect && r.meaningHops !== null);
  const rejected = res.filter((r) => r.meaningHops === null);

  console.log(`── PERFECT resonances (sound ∧ meaning coincide) — ${perfect.length} ──`);
  for (const r of perfect) {
    const via = r.chain && r.chain.hops > 1 ? `  via ${chainText(map, r.chain.path)}` : "";
    console.log(
      `  ${fmt(r.score)}  ${r.a.text}[${r.a.lang}] ≈sound≈ ${r.b.text}[${r.b.lang}]` +
        `  (sound=${fmt(r.sound.weight)}, meaningHops=${r.meaningHops})${via}`,
    );
  }

  console.log(`\n── FRONTIER (sound-alike, meaning NEARLY closes — hunt next) — ${frontier.length} ──`);
  for (const r of frontier) {
    console.log(
      `  ${fmt(r.score)}  ${r.a.text}[${r.a.lang}] ≈sound≈ ${r.b.text}[${r.b.lang}]` +
        `  (sound=${fmt(r.sound.weight)}, meaningHops=${r.meaningHops})` +
        (r.chain ? `  via ${chainText(map, r.chain.path)}` : ""),
    );
  }

  console.log(`\n── REJECTED (sound-alike but meaning unreachable — false friends) — ${rejected.length} ──`);
  for (const r of rejected) {
    console.log(`  ${fmt(r.sound.weight)}  ${r.a.text}[${r.a.lang}] ≈sound≈ ${r.b.text}[${r.b.lang}]  (no meaning path)`);
  }

  console.log("\nThe hunt: PERFECT rows are the map of meaning so far. FRONTIER rows are");
  console.log("the next routine's targets — confirm one weak link and they promote to PERFECT.");
}

main();
