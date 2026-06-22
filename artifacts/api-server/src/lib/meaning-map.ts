/**
 * Meaning Map — the semantic ⨉ homophonic graph.
 * ================================================
 *
 * The reservoir (see `reservoir-mining.ts`) is a *homophone pair bank*: rows of
 * EN↔FR phrases that SOUND alike. The Flit Lab generates sound-alikes on the
 * fly. Both answer "what sounds like X?" Neither answers the deeper question the
 * project is really chasing:
 *
 *     Is there a phrase in the other language that *sounds like* X **and**
 *     *means* X?  (a true cross-lingual homophone-translation — the gem)
 *
 * Those gems are rare, so you can't mine them by brute force. You find them by
 * *composing chains* over two kinds of edge:
 *
 *   • homophone edges  — cross-lingual SOUND similarity   (from phoneme-chain /
 *                        the reservoir).  "more" ~ "mort"
 *   • semantic edges   — same MEANING: translations across languages AND
 *                        synonyms within a language.       "more" = "plus",
 *                        "more" ≈ "extra"
 *
 * A node is a phrase in a language. Lay the two pair banks down as edges on the
 * same node set, layer synonyms on top, and the answer falls out as a graph
 * query: **convergence** — a target node reachable from an origin by *both* a
 * sound-only chain and a sense-only chain. The two worlds meet on the same word.
 *
 * "Whittle down the same meanings over time": every new edge can only create or
 * strengthen convergences, never remove them, and independent chains that land
 * on the same gem reinforce its confidence. The map sharpens monotonically as
 * the pair banks grow.
 *
 * This module is intentionally pure and dependency-free: no DB, no LLM, no
 * network. Scoring lives elsewhere; here we just hold the graph and walk it.
 * Feed it reservoir rows via {@link MeaningGraph.ingestHomophonePair} and
 * translation/synonym facts via {@link MeaningGraph.ingestSemanticPair}, then
 * call {@link MeaningGraph.findConvergences}. Runnable end-to-end demo lives in
 * `meaning-map.demo.ts` (`node --experimental-strip-types meaning-map.demo.ts`).
 */

export type Lang = string; // ISO-ish code: "en", "fr", "ko", …

export type EdgeKind = "homophone" | "semantic";

export interface MeaningNode {
  /** Stable identity: `${lang}::${normalizedText}`. */
  id: string;
  lang: Lang;
  text: string;
  /** Optional one-line English meaning, carried for display / debugging. */
  gloss?: string;
}

export interface Edge {
  from: string;
  to: string;
  kind: EdgeKind;
  /** Strength in [0,1]: phonetic similarity (homophone) or meaning confidence (semantic). */
  weight: number;
  /** Provenance, e.g. "reservoir:tierS", "translation", "synonym". */
  source?: string;
}

/** A scored walk from one node to another over edges of a single kind. */
export interface Path {
  /** Node ids, origin first, target last. */
  nodes: string[];
  /** Aggregate strength in [0,1] (product of weights × per-hop decay). */
  strength: number;
  hops: number;
}

/**
 * A gem: a target phrase in another language reachable from the origin by BOTH
 * a homophone chain and a semantic chain. The two paths are independent
 * witnesses that this phrase sounds like *and* means the origin.
 */
export interface Convergence {
  origin: MeaningNode;
  target: MeaningNode;
  /** Best sound-only path origin → target. */
  soundPath: Path;
  /** Best sense-only path origin → target. */
  sensePath: Path;
  /**
   * Headline score. The weakest of the two worlds (min) gates the gem — a
   * phrase that sounds perfect but means nothing, or vice-versa, is not a gem.
   * We report the geometric mean for ranking and `gate` (the min) so callers
   * can threshold honestly.
   */
  score: number;
  gate: number;
}

export interface ConvergenceOptions {
  /** Per-hop multiplicative penalty discouraging long chains. Default 0.85. */
  hopDecay?: number;
  /** Ignore edges weaker than this. Default 0.5. */
  minEdgeWeight?: number;
  /** Drop paths whose aggregate strength falls below this. Default 0.25. */
  minPathStrength?: number;
  /** Only surface convergences crossing a language boundary. Default true. */
  crossLingualOnly?: boolean;
  /** Cap on chain length (hops) explored per world. Default 4. */
  maxHops?: number;
}

const DEFAULTS: Required<ConvergenceOptions> = {
  hopDecay: 0.85,
  minEdgeWeight: 0.5,
  minPathStrength: 0.25,
  crossLingualOnly: true,
  maxHops: 4,
};

function normalize(text: string): string {
  return text.trim().toLowerCase().replace(/\s+/g, " ");
}

export function nodeId(lang: Lang, text: string): string {
  return `${lang}::${normalize(text)}`;
}

/** Tiny binary min-heap keyed by number, so Dijkstra stays O(E log V). */
class MinHeap<T> {
  private items: { key: number; value: T }[] = [];
  get size(): number {
    return this.items.length;
  }
  push(key: number, value: T): void {
    const a = this.items;
    a.push({ key, value });
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p]!.key <= a[i]!.key) break;
      [a[p], a[i]] = [a[i]!, a[p]!];
      i = p;
    }
  }
  pop(): { key: number; value: T } | undefined {
    const a = this.items;
    if (a.length === 0) return undefined;
    const top = a[0]!;
    const last = a.pop()!;
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        let s = i;
        if (l < a.length && a[l]!.key < a[s]!.key) s = l;
        if (r < a.length && a[r]!.key < a[s]!.key) s = r;
        if (s === i) break;
        [a[s], a[i]] = [a[i]!, a[s]!];
        i = s;
      }
    }
    return top;
  }
}

export class MeaningGraph {
  private nodes = new Map<string, MeaningNode>();
  /** Undirected adjacency: each edge is stored on both endpoints. */
  private adj = new Map<string, Edge[]>();

  get nodeCount(): number {
    return this.nodes.size;
  }
  get edgeCount(): number {
    let n = 0;
    for (const list of this.adj.values()) n += list.length;
    return n / 2; // each undirected edge counted twice
  }

  allNodes(): MeaningNode[] {
    return [...this.nodes.values()];
  }

  getNode(id: string): MeaningNode | undefined {
    return this.nodes.get(id);
  }

  addNode(lang: Lang, text: string, gloss?: string): MeaningNode {
    const id = nodeId(lang, text);
    const existing = this.nodes.get(id);
    if (existing) {
      // First non-empty gloss wins; don't clobber with a blank.
      if (!existing.gloss && gloss) existing.gloss = gloss;
      return existing;
    }
    const node: MeaningNode = { id, lang, text: text.trim(), gloss };
    this.nodes.set(id, node);
    this.adj.set(id, []);
    return node;
  }

  /**
   * Add an undirected edge. If an edge of the same kind already joins the two
   * nodes, keep the *stronger* weight (more evidence never weakens the map).
   */
  addEdge(aId: string, bId: string, kind: EdgeKind, weight: number, source?: string): void {
    if (aId === bId) return;
    const w = Math.max(0, Math.min(1, weight));
    for (const [from, to] of [[aId, bId], [bId, aId]] as const) {
      const list = this.adj.get(from);
      if (!list) continue;
      const found = list.find((e) => e.to === to && e.kind === kind);
      if (found) {
        if (w > found.weight) {
          found.weight = w;
          if (source) found.source = source;
        }
      } else {
        list.push({ from, to, kind, weight: w, source });
      }
    }
  }

  /** Convenience: a homophone (sound-alike) pair, e.g. one reservoir row. */
  ingestHomophonePair(
    a: { lang: Lang; text: string; gloss?: string },
    b: { lang: Lang; text: string; gloss?: string },
    similarity: number,
    source = "reservoir",
  ): void {
    const na = this.addNode(a.lang, a.text, a.gloss);
    const nb = this.addNode(b.lang, b.text, b.gloss);
    this.addEdge(na.id, nb.id, "homophone", similarity, source);
  }

  /**
   * Convenience: a same-meaning pair — a translation (across languages) or a
   * synonym (within a language). This is the layer the reservoir lacks.
   */
  ingestSemanticPair(
    a: { lang: Lang; text: string; gloss?: string },
    b: { lang: Lang; text: string; gloss?: string },
    confidence: number,
    source = "translation",
  ): void {
    const na = this.addNode(a.lang, a.text, a.gloss);
    const nb = this.addNode(b.lang, b.text, b.gloss);
    this.addEdge(na.id, nb.id, "semantic", confidence, source);
  }

  neighbors(id: string, kind: EdgeKind): Edge[] {
    return (this.adj.get(id) ?? []).filter((e) => e.kind === kind);
  }

  /**
   * Single-source best paths over edges of ONE kind, maximizing
   *   strength = Π(edge weights) × hopDecay^(hops-1)
   * via Dijkstra in -log space (all costs ≥ 0). Returns best Path to every node
   * reachable within the strength / hop / edge-weight budget.
   */
  bestPaths(originId: string, kind: EdgeKind, opts: ConvergenceOptions = {}): Map<string, Path> {
    const o = { ...DEFAULTS, ...opts };
    const decayCost = -Math.log(o.hopDecay);
    const maxCost = -Math.log(o.minPathStrength);

    // cost = Σ -ln(weight) + (hops-1)*decayCost ; minimize.
    const best = new Map<string, { cost: number; nodes: string[] }>();
    const heap = new MinHeap<{ id: string; nodes: string[] }>();
    best.set(originId, { cost: 0, nodes: [originId] });
    heap.push(0, { id: originId, nodes: [originId] });

    while (heap.size > 0) {
      const top = heap.pop()!;
      const cur = top.value;
      const curBest = best.get(cur.id);
      if (!curBest || top.key > curBest.cost) continue; // stale
      const hops = cur.nodes.length - 1;
      if (hops >= o.maxHops) continue;
      for (const e of this.neighbors(cur.id, kind)) {
        if (e.weight < o.minEdgeWeight) continue;
        const step = -Math.log(e.weight) + (hops > 0 ? decayCost : 0);
        const nextCost = top.key + step;
        if (nextCost > maxCost) continue;
        const prev = best.get(e.to);
        if (!prev || nextCost < prev.cost) {
          const nodes = [...cur.nodes, e.to];
          best.set(e.to, { cost: nextCost, nodes });
          heap.push(nextCost, { id: e.to, nodes });
        }
      }
    }

    const out = new Map<string, Path>();
    for (const [id, rec] of best) {
      if (id === originId) continue;
      out.set(id, { nodes: rec.nodes, strength: Math.exp(-rec.cost), hops: rec.nodes.length - 1 });
    }
    return out;
  }

  /**
   * Find gems for a single origin: targets reachable by BOTH a homophone chain
   * and a semantic chain. Ranked by geometric mean of the two path strengths,
   * gated by the weaker world.
   */
  convergencesFrom(originId: string, opts: ConvergenceOptions = {}): Convergence[] {
    const o = { ...DEFAULTS, ...opts };
    const origin = this.nodes.get(originId);
    if (!origin) return [];
    const sound = this.bestPaths(originId, "homophone", o);
    const sense = this.bestPaths(originId, "semantic", o);

    const out: Convergence[] = [];
    for (const [targetId, soundPath] of sound) {
      const sensePath = sense.get(targetId);
      if (!sensePath) continue;
      const target = this.nodes.get(targetId)!;
      if (o.crossLingualOnly && target.lang === origin.lang) continue;
      const gate = Math.min(soundPath.strength, sensePath.strength);
      const score = Math.sqrt(soundPath.strength * sensePath.strength);
      out.push({ origin, target, soundPath, sensePath, score, gate });
    }
    out.sort((a, b) => b.score - a.score);
    return out;
  }

  /** Sweep every node as an origin and return all gems, best first. */
  findConvergences(opts: ConvergenceOptions = {}): Convergence[] {
    const all: Convergence[] = [];
    const seen = new Set<string>();
    for (const id of this.nodes.keys()) {
      for (const c of this.convergencesFrom(id, opts)) {
        // De-dupe the symmetric (A→B / B→A) view; keep one per unordered pair.
        const key = [c.origin.id, c.target.id].sort().join("§");
        if (seen.has(key)) continue;
        seen.add(key);
        all.push(c);
      }
    }
    all.sort((a, b) => b.score - a.score);
    return all;
  }

  /**
   * Meaning clusters: connected components over strong semantic edges. Two
   * phrases in the same cluster are "the same meaning" as far as the map is
   * concerned — this is the structure that lets the map collapse synonyms and
   * translations into a single meaning over time. Returns a map nodeId → root.
   */
  meaningClusters(minSemanticWeight = 0.7): Map<string, string> {
    const parent = new Map<string, string>();
    const find = (x: string): string => {
      let r = x;
      while (parent.get(r) !== r) r = parent.get(r)!;
      // path compression
      let c = x;
      while (parent.get(c) !== r) {
        const n = parent.get(c)!;
        parent.set(c, r);
        c = n;
      }
      return r;
    };
    const union = (a: string, b: string): void => {
      const ra = find(a);
      const rb = find(b);
      if (ra !== rb) parent.set(ra, rb);
    };
    for (const id of this.nodes.keys()) parent.set(id, id);
    for (const list of this.adj.values()) {
      for (const e of list) {
        if (e.kind === "semantic" && e.weight >= minSemanticWeight) union(e.from, e.to);
      }
    }
    const out = new Map<string, string>();
    for (const id of this.nodes.keys()) out.set(id, find(id));
    return out;
  }
}
