/**
 * Meaning Map — the convergence layer over the homophone reservoir.
 *
 * The reservoir gives us pairs that SOUND alike across EN↔FR (phonetic
 * similarity) together with a one-line English gloss of each side's MEANING.
 * The meaning map fuses those two signals:
 *
 *   • phonetic similarity — do the two phrases sound alike? (from the reservoir)
 *   • semantic similarity — do their glosses mean the same? (from lib/semantic)
 *
 * A pair that scores high on BOTH is a "resonance": a cross-lingual phrase that
 * is simultaneously a homophone and a (near-)translation — the rare point where
 * sound and sense meet across the language barrier. Ranking the reservoir by
 * resonance whittles the corpus down to those meeting points.
 *
 * This module is PURE (no I/O, no LLM, no DB) so it is trivially testable: the
 * phonetic score arrives from the reservoir and the semantic score is injected
 * by the caller (see routes/meaning.ts, which wires in lib/semantic).
 */

// ---- Resonance: fusing sound and sense ------------------------------------

export type Quadrant = "resonant" | "homophone" | "translation" | "weak";

export interface ResonanceThresholds {
  /** Phonetic-similarity "high" cutoff. */
  phon: number;
  /** Semantic-similarity "high" cutoff. */
  sem: number;
}

export const DEFAULT_THRESHOLDS: ResonanceThresholds = { phon: 0.75, sem: 0.6 };

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(x) ? x : 0));
}

/**
 * Combined sound+sense score. Geometric mean, so a single low factor tanks the
 * result — mirroring the hybrid scorer's "both judges must agree" philosophy.
 */
export function resonanceScore(phonetic: number, semantic: number): number {
  return Math.sqrt(clamp01(phonetic) * clamp01(semantic));
}

/**
 * Place a pair in the sound×sense plane:
 *   resonant    — sounds alike AND means alike  (the gems)
 *   homophone   — sounds alike, means different (classic Mots-d'Heures effect)
 *   translation — means alike, sounds different (ordinary cross-lingual sense)
 *   weak        — neither
 */
export function classifyPair(
  phonetic: number,
  semantic: number,
  t: ResonanceThresholds = DEFAULT_THRESHOLDS,
): Quadrant {
  const hiP = phonetic >= t.phon;
  const hiS = semantic >= t.sem;
  if (hiP && hiS) return "resonant";
  if (hiP && !hiS) return "homophone";
  if (!hiP && hiS) return "translation";
  return "weak";
}

export interface ScoredPair {
  id: number;
  enPhrase: string;
  frPhrase: string;
  enGloss: string;
  frGloss: string;
  phonetic: number;
  semantic: number;
}

export interface ResonancePair extends ScoredPair {
  resonance: number;
  quadrant: Quadrant;
}

/** The whittle-down: annotate every pair with resonance + quadrant, strongest first. */
export function rankResonance(
  pairs: ScoredPair[],
  t: ResonanceThresholds = DEFAULT_THRESHOLDS,
): ResonancePair[] {
  return pairs
    .map((p) => ({
      ...p,
      resonance: resonanceScore(p.phonetic, p.semantic),
      quadrant: classifyPair(p.phonetic, p.semantic, t),
    }))
    .sort((a, b) => b.resonance - a.resonance);
}

export interface QuadrantCounts {
  resonant: number;
  homophone: number;
  translation: number;
  weak: number;
}

export function countQuadrants(pairs: ResonancePair[]): QuadrantCounts {
  const c: QuadrantCounts = { resonant: 0, homophone: 0, translation: 0, weak: 0 };
  for (const p of pairs) c[p.quadrant]++;
  return c;
}

// ---- The meaning-map graph ------------------------------------------------
//
// Nodes are phrases (lang-tagged). Two edge kinds:
//   • PHONE edges — the two sides of a reservoir pair (a sound bridge, EN↔FR).
//   • SENSE edges — phrases whose glosses denote the same meaning (synonym /
//                   translation). v1 derives these cheaply: (a) a pair whose
//                   two sides already mean the same (semantic ≥ threshold) and
//                   (b) any two phrases that share a normalized gloss. Deeper
//                   LLM-judged synonymy is the documented next step.
//
// "Meaning islands" are connected components over SENSE edges — clusters of
// phrases that all mean the same thing regardless of language. A PHONE edge
// whose endpoints live in DIFFERENT islands is a "bridge" (a sound-link that
// joins two distinct meanings — the classic homophonic-translation surprise);
// a PHONE edge WITHIN one island is a "resonance" (sound and sense agree).

export interface MapNode {
  key: string;
  phrase: string;
  lang: "en" | "fr";
  gloss: string;
  /** Connected-component id over SENSE edges. */
  island: number;
}

export interface MapEdge {
  kind: "phone" | "sense";
  a: string;
  b: string;
  weight: number;
  /** Source reservoir row, for phone edges. */
  pairId?: number;
}

export interface MeaningMap {
  nodes: MapNode[];
  edges: MapEdge[];
  islandCount: number;
  /** Phone edges spanning two islands. */
  bridges: MapEdge[];
  /** Phone edges inside a single island (sound and sense agree). */
  resonances: MapEdge[];
}

export function normalize(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9 ]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function nodeKey(lang: "en" | "fr", phrase: string): string {
  return `${lang}:${normalize(phrase)}`;
}

/** Minimal union-find with path compression, keyed by node key. */
class UnionFind {
  private parent = new Map<string, string>();

  find(x: string): string {
    const p = this.parent.get(x);
    if (p === undefined) {
      this.parent.set(x, x);
      return x;
    }
    if (p !== x) {
      const root = this.find(p);
      this.parent.set(x, root);
      return root;
    }
    return x;
  }

  union(a: string, b: string): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent.set(ra, rb);
  }
}

export function buildMeaningMap(
  pairs: ScoredPair[],
  t: ResonanceThresholds = DEFAULT_THRESHOLDS,
): MeaningMap {
  const nodeMap = new Map<string, MapNode>();
  const phoneEdges: MapEdge[] = [];
  const senseEdges: MapEdge[] = [];
  const byGloss = new Map<string, string[]>();
  const uf = new UnionFind();

  const ensureNode = (lang: "en" | "fr", phrase: string, gloss: string): string => {
    const k = nodeKey(lang, phrase);
    if (!nodeMap.has(k)) {
      nodeMap.set(k, { key: k, phrase, lang, gloss, island: -1 });
      uf.find(k); // register in the forest
      const gk = normalize(gloss);
      if (gk) {
        const arr = byGloss.get(gk) ?? [];
        arr.push(k);
        byGloss.set(gk, arr);
      }
    }
    return k;
  };

  for (const p of pairs) {
    const en = ensureNode("en", p.enPhrase, p.enGloss);
    const fr = ensureNode("fr", p.frPhrase, p.frGloss);
    phoneEdges.push({ kind: "phone", a: en, b: fr, weight: clamp01(p.phonetic), pairId: p.id });
    // A pair whose two sides also mean the same is itself a SENSE edge.
    if (p.semantic >= t.sem) {
      senseEdges.push({ kind: "sense", a: en, b: fr, weight: clamp01(p.semantic) });
      uf.union(en, fr);
    }
  }

  // SENSE edges from shared normalized glosses (synonym / translation links).
  for (const keys of byGloss.values()) {
    const first = keys[0];
    if (first === undefined) continue;
    for (let i = 1; i < keys.length; i++) {
      const other = keys[i];
      if (other === undefined) continue;
      senseEdges.push({ kind: "sense", a: first, b: other, weight: 1 });
      uf.union(first, other);
    }
  }

  // Assign stable island ids.
  const islandId = new Map<string, number>();
  let next = 0;
  for (const node of nodeMap.values()) {
    const root = uf.find(node.key);
    let id = islandId.get(root);
    if (id === undefined) {
      id = next++;
      islandId.set(root, id);
    }
    node.island = id;
  }

  const islandOf = (k: string): number => nodeMap.get(k)?.island ?? -1;
  const bridges = phoneEdges.filter((e) => islandOf(e.a) !== islandOf(e.b));
  const resonances = phoneEdges.filter((e) => islandOf(e.a) === islandOf(e.b));

  return {
    nodes: [...nodeMap.values()],
    edges: [...phoneEdges, ...senseEdges],
    islandCount: next,
    bridges,
    resonances,
  };
}
