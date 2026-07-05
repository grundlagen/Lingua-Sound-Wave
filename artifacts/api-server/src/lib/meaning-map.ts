/**
 * The Meaning Map — a resonance graph over meaning-chains and sound-chains.
 *
 * Two complementary implementations:
 *   PART I  — Resonance engine (union-find meaning clusters + sound-edge ranking)
 *   PART II — Convergence engine (MeaningMap class with Dijkstra closure)
 *
 * ─────────────────────────────────────────────────────────────────────────
 * PART I — The resonance hunt
 * ─────────────────────────────────────────────────────────────────────────
 * The reservoir and Flit Lab already mine cross-lingual *sound-alikes*
 * (à la Mots d'Heures: Gousses, Rames) — text in one language that, spoken
 * aloud, mimics text in the other. By design those matches sacrifice meaning
 * for sound: "un petit d'un petit" sounds like "Humpty Dumpty" but means
 * something else entirely.
 *
 * This module hunts the rarer treasure: pairs where *sound and meaning
 * coincide*. A French phrase that both means what an English phrase means
 * AND sounds like it. Cognates are the obvious floor of this set
 * ("table"≈"table", "nation"≈"nation"), but the interesting ones are hidden
 * behind synonym and paraphrase chains and have to be hunted for.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * The structure (the user's "pair banks, mapping synonyms on top")
 * ─────────────────────────────────────────────────────────────────────────
 * One universe of phrase-nodes spanning EN and FR. Two kinds of weighted,
 * undirected edges laid over the same nodes:
 *
 *   • MEANING edges  — phrases that mean the same thing. EN↔FR translations
 *     plus EN↔EN and FR↔FR synonyms. This is the "semantic meaning chain."
 *
 *   • SOUND edges    — phrases that sound alike. Primarily cross-lingual
 *     EN↔FR homophones (from the reservoir). This is the "homophone chain."
 *
 * Run union-find over the MEANING edges and you get *meaning clusters*: sets
 * of phrases, across both languages, that all denote one concept. Synonyms
 * and translations fatten each cluster.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * The resonance (where the chains connect)
 * ─────────────────────────────────────────────────────────────────────────
 * Take a SOUND edge a~b. Ask: how far apart, *in meaning*, are its endpoints?
 *
 *   • Same meaning cluster (0 hops)  → PERFECT resonance. Sound and meaning
 *     coincide. The grail. (Every cognate lands here; the prize is the
 *     non-obvious pairs the synonym chains drag into the same cluster.)
 *   • A few synonym/translation hops apart → a weaker, partial resonance —
 *     a chain that *almost* closes. These are the frontier.
 *   • Unreachable in meaning → pure sound-alike, no semantic resonance.
 *
 * Resonance score = soundWeight × meaningProximity, where meaningProximity
 * decays with the number of meaning hops and is bottlenecked by the weakest
 * link in the connecting chain (a chain is only as strong as its weakest
 * synonym).
 *
 * ─────────────────────────────────────────────────────────────────────────
 * Whittling down over time (the routine)
 * ─────────────────────────────────────────────────────────────────────────
 * Rank every sound edge by resonance. The strong band is the "map of
 * meaning" — it grows and stabilizes. The frontier (sound edges that are
 * close-but-not-closed in meaning) is the to-do list for the next routine:
 * find the one missing synonym or translation that would collapse the two
 * clusters into one and promote a near-resonance to a perfect one. Each
 * routine adds banks, recomputes clusters, and the map sharpens.
 *
 * This module is deliberately dependency-free and weight-agnostic: callers
 * supply edge weights from whatever judges they trust (phonemeChainScore for
 * sound, an LLM or embedding cosine for meaning). A crude offline phonetic
 * fallback is provided so the graph can be exercised without any model.
 */

export type Lang = "en" | "fr";

// ═══════════════════════════════════════════════════════════════════════════
// PART I — Resonance engine (union-find meaning clusters)
// ═══════════════════════════════════════════════════════════════════════════

export interface PhraseNode {
  /** Stable id: `${lang}:${normalized text}`. */
  id: string;
  text: string;
  lang: Lang;
}

export type EdgeKind = "meaning" | "sound";

export interface ResonanceEdge {
  kind: EdgeKind;
  a: string; // node id
  b: string; // node id
  weight: number; // [0,1]
  /** Optional provenance: "translation", "synonym", "cognate", "reservoir", … */
  source?: string;
}

/** A connecting meaning-chain between two nodes: the ordered ids and its bottleneck weight. */
export interface MeaningChain {
  path: string[]; // node ids, inclusive of both endpoints
  hops: number; // path.length - 1
  bottleneck: number; // min meaning-edge weight along the path (1 if hops === 0)
}

export interface Resonance {
  /** The sound edge that resonates. */
  sound: ResonanceEdge;
  a: PhraseNode;
  b: PhraseNode;
  /** Whether the endpoints already share a meaning cluster. */
  perfect: boolean;
  /** Meaning hops between the endpoints (0 = same cluster). null = unreachable. */
  meaningHops: number | null;
  /** The connecting meaning-chain, if any. */
  chain: MeaningChain | null;
  /** Composite resonance score in [0,1]. */
  score: number;
}

class UnionFind {
  private parent = new Map<string, string>();

  find(x: string): string {
    if (!this.parent.has(x)) this.parent.set(x, x);
    if (this.parent.get(x) !== x) {
      this.parent.set(x, this.find(this.parent.get(x)!));
    }
    return this.parent.get(x)!;
  }

  union(a: string, b: string): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent.set(rb, ra);
  }
}

export function buildMeaningClusters(
  nodes: PhraseNode[],
  edges: ResonanceEdge[],
): Map<string, PhraseNode[]> {
  const uf = new UnionFind();
  const nodeMap = new Map<string, PhraseNode>();
  for (const n of nodes) {
    nodeMap.set(n.id, n);
    uf.find(n.id);
  }
  for (const e of edges) {
    if (e.kind === "meaning") uf.union(e.a, e.b);
  }
  const clusters = new Map<string, PhraseNode[]>();
  for (const id of nodeMap.keys()) {
    const root = uf.find(id);
    if (!clusters.has(root)) clusters.set(root, []);
    clusters.get(root)!.push(nodeMap.get(id)!);
  }
  return clusters;
}

export function meaningProximity(
  aId: string,
  bId: string,
  clusters: Map<string, PhraseNode[]>,
  meaningEdges: ResonanceEdge[],
): { hops: number; bottleneck: number } | null {
  // Same cluster → perfect proximity
  for (const [, members] of clusters) {
    const aIn = members.some((n) => n.id === aId);
    const bIn = members.some((n) => n.id === bId);
    if (aIn && bIn) return { hops: 0, bottleneck: 1 };
  }

  // BFS over meaning edges for inter-cluster proximity
  const adj = new Map<string, Array<{ to: string; weight: number }>>();
  for (const e of meaningEdges) {
    if (e.kind !== "meaning") continue;
    if (!adj.has(e.a)) adj.set(e.a, []);
    if (!adj.has(e.b)) adj.set(e.b, []);
    adj.get(e.a)!.push({ to: e.b, weight: e.weight });
    adj.get(e.b)!.push({ to: e.a, weight: e.weight });
  }

  interface BfsState {
    id: string;
    hops: number;
    bottleneck: number;
  }
  const queue: BfsState[] = [{ id: aId, hops: 0, bottleneck: 1 }];
  const visited = new Set<string>([aId]);

  while (queue.length > 0) {
    const cur = queue.shift()!;
    if (cur.id === bId) return { hops: cur.hops, bottleneck: cur.bottleneck };
    for (const edge of adj.get(cur.id) ?? []) {
      if (visited.has(edge.to)) continue;
      visited.add(edge.to);
      queue.push({
        id: edge.to,
        hops: cur.hops + 1,
        bottleneck: Math.min(cur.bottleneck, edge.weight),
      });
    }
  }
  return null; // unreachable
}

export function rankResonances(
  nodes: PhraseNode[],
  soundEdges: ResonanceEdge[],
  meaningEdges: ResonanceEdge[],
): Resonance[] {
  const clusters = buildMeaningClusters(nodes, meaningEdges);
  const nodeMap = new Map<string, PhraseNode>();
  for (const n of nodes) nodeMap.set(n.id, n);

  const results: Resonance[] = [];
  for (const edge of soundEdges) {
    if (edge.kind !== "sound") continue;
    const a = nodeMap.get(edge.a);
    const b = nodeMap.get(edge.b);
    if (!a || !b) continue;

    const prox = meaningProximity(edge.a, edge.b, clusters, meaningEdges);
    const meaningHops = prox?.hops ?? null;
    const perfect = meaningHops === 0;
    const score = prox
      ? edge.weight * prox.bottleneck * Math.pow(0.8, prox.hops)
      : 0;

    results.push({
      sound: edge,
      a,
      b,
      perfect,
      meaningHops,
      chain: prox
        ? { path: [edge.a, edge.b], hops: prox.hops, bottleneck: prox.bottleneck }
        : null,
      score,
    });
  }
  return results.sort((a, b) => b.score - a.score);
}

export function crudePhoneticKey(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z]/g, "")
    .replace(/[aeiouy]+/g, "V")
    .replace(/(.)\1+/g, "$1");
}

export function crudePhoneticSimilarity(aText: string, bText: string): number {
  const a = crudePhoneticKey(aText);
  const b = crudePhoneticKey(bText);
  if (!a.length && !b.length) return 1;
  const d = levenshtein(a, b);
  return 1 - d / Math.max(a.length, b.length);
}

// ═══════════════════════════════════════════════════════════════════════════
// PART II — Convergence engine (MeaningMap with Dijkstra closure)
// ─────────────────────────────────────────────────────────────────────────
// meaning-map.ts — the Meaning Map engine.
//
// The grand goal (see MEANING-MAP.md): fuse two pair-banks into one graph and
// walk the connecting chains until the same meanings, approached from sound and
// from sense, *converge*.
//
//   • SEMANTIC edges preserve meaning, destroy sound  (translations, synonyms)
//   • HOMOPHONIC edges preserve sound, destroy meaning (sound-alikes)
//
// A node A in English and a node F in French are a CONVERGENCE when there is
// BOTH a meaning-only path A⇒F (so F means what A means) AND a sound-only path
// A≈F (so F sounds like A). The two paths are independent: that independence is
// exactly what makes the pair a real homophonic translation rather than a
// coincidence of one channel.
//
//   - cognate convergence : both paths length 1 (e.g. blue / bleu)
//   - transitive convergence: the meaning path runs through a synonym hop —
//     the feat the map exists to find.
//   - siren / false friend : a strong sound path with NO meaning path
//     (e.g. en:pain ≈ fr:pain "bread"). Kept and flagged, never silently dropped.
//
// "Whittling down the same meanings over time" is modelled literally: rounds of
// rising thresholds that prune the atlas toward its gold core, keeping one best
// representative per meaning cluster.
//
// Pure TypeScript, no workspace imports — runs under `node --experimental-strip-types`.
// The default phonetic scorer is a deliberately simple heuristic; the real
// `phoneme.ts` LLM chain scorer can be injected via `mineHomophonic(scorer)`.
// ═══════════════════════════════════════════════════════════════════════════

export type Layer = "semantic" | "homophonic";

export interface MapNode {
  id: string; // `${lang}:${text}`
  text: string;
  lang: Lang;
  gloss: string;
  concept?: string;
}

export interface MapEdge {
  to: string;
  layer: Layer;
  weight: number; // (0,1]
  reason: string;
}

export interface PathStep {
  from: string;
  to: string;
  layer: Layer;
  weight: number;
  reason: string;
}

export interface Reach {
  id: string;
  score: number; // product of edge weights along best path, with per-hop decay
  path: PathStep[];
}

export interface Convergence {
  from: string; // English node id
  to: string; // French node id
  kind: "cognate" | "transitive";
  meaning: number; // semantic-path score
  sound: number; // homophonic-path score
  combined: number; // geometric mean (echoes the repo's hybrid scoring)
  meaningPath: PathStep[];
  soundPath: PathStep[];
}

export interface Siren {
  from: string; // English node id
  to: string; // French node id
  sound: number;
  soundPath: PathStep[];
  note: string;
}

export interface ClosureOptions {
  maxDepth?: number;
  minWeight?: number; // prune edges below this
  hopDecay?: number; // multiplicative penalty per hop, in (0,1]
}

const nodeId = (lang: Lang, text: string): string => `${lang}:${text}`;

export function mapId(lang: Lang, text: string): string {
  return nodeId(lang, text);
}

export class MeaningMap {
  readonly nodes = new Map<string, MapNode>();
  private readonly adj = new Map<string, MapEdge[]>();

  addNode(node: Omit<MapNode, "id"> & { id?: string }): MapNode {
    const fullId = node.id ?? nodeId(node.lang, node.text);
    const existing = this.nodes.get(fullId);
    if (existing) return existing;
    const created: MapNode = { ...node, id: fullId };
    this.nodes.set(fullId, created);
    this.adj.set(fullId, []);
    return created;
  }

  /** Add a symmetric edge in both directions on the given layer. */
  addEdge(a: string, b: string, layer: Layer, weight: number, reason: string): void {
    if (a === b) return;
    if (!this.nodes.has(a) || !this.nodes.has(b)) return;
    this.pushEdge(a, { to: b, layer, weight, reason });
    this.pushEdge(b, { to: a, layer, weight, reason });
  }

  private pushEdge(from: string, edge: MapEdge): void {
    const list = this.adj.get(from);
    if (!list) return;
    // Keep the strongest edge per (to, layer) pair.
    const prior = list.find((e) => e.to === edge.to && e.layer === edge.layer);
    if (prior) {
      if (edge.weight > prior.weight) {
        prior.weight = edge.weight;
        prior.reason = edge.reason;
      }
      return;
    }
    list.push(edge);
  }

  edgesFrom(fromId: string): readonly MapEdge[] {
    return this.adj.get(fromId) ?? [];
  }

  countEdges(layer?: Layer): number {
    let n = 0;
    for (const list of this.adj.values()) {
      for (const e of list) if (!layer || e.layer === layer) n += 1;
    }
    return n / 2; // symmetric
  }

  /**
   * Best-first single-layer reachability: from `start`, follow only `layer`
   * edges, maximising the product of weights (with a per-hop decay so shorter,
   * cleaner chains win ties). Returns the best path to every reachable node.
   */
  closure(start: string, layer: Layer, opts: ClosureOptions = {}): Map<string, Reach> {
    const maxDepth = opts.maxDepth ?? 4;
    const minWeight = opts.minWeight ?? 0.2;
    const hopDecay = opts.hopDecay ?? 0.92;

    const best = new Map<string, Reach>();
    best.set(start, { id: start, score: 1, path: [] });

    // Simple Dijkstra-style frontier (graphs here are small).
    const frontier: Array<{ id: string; score: number; path: PathStep[] }> = [
      { id: start, score: 1, path: [] },
    ];

    while (frontier.length > 0) {
      // pop the highest-scoring node
      frontier.sort((a, b) => a.score - b.score);
      const cur = frontier.pop()!;
      const recorded = best.get(cur.id);
      if (recorded && cur.score < recorded.score - 1e-12) continue;
      if (cur.path.length >= maxDepth) continue;

      for (const edge of this.edgesFrom(cur.id)) {
        if (edge.layer !== layer) continue;
        if (edge.weight < minWeight) continue;
        const nextScore = cur.score * edge.weight * hopDecay;
        const prev = best.get(edge.to);
        if (prev && nextScore <= prev.score + 1e-12) continue;
        const step: PathStep = {
          from: cur.id,
          to: edge.to,
          layer,
          weight: edge.weight,
          reason: edge.reason,
        };
        const path = [...cur.path, step];
        best.set(edge.to, { id: edge.to, score: nextScore, path });
        frontier.push({ id: edge.to, score: nextScore, path });
      }
    }
    return best;
  }

  /**
   * For every English node, intersect its semantic closure and its homophonic
   * closure over French targets. Each French node in both is a convergence.
   */
  findConvergences(opts: ClosureOptions = {}): Convergence[] {
    const out: Convergence[] = [];
    for (const node of this.nodes.values()) {
      if (node.lang !== "en") continue;
      const sem = this.closure(node.id, "semantic", opts);
      const snd = this.closure(node.id, "homophonic", opts);
      for (const [targetId, meaning] of sem) {
        const target = this.nodes.get(targetId);
        if (!target || target.lang !== "fr") continue;
        const sound = snd.get(targetId);
        if (!sound) continue;
        const meaningHops = meaning.path.length;
        const soundHops = sound.path.length;
        const combined = Math.sqrt(meaning.score * sound.score);
        out.push({
          from: node.id,
          to: targetId,
          kind: meaningHops <= 1 && soundHops <= 1 ? "cognate" : "transitive",
          meaning: meaning.score,
          sound: sound.score,
          combined,
          meaningPath: meaning.path,
          soundPath: sound.path,
        });
      }
    }
    return out.sort((a, b) => b.combined - a.combined);
  }

  /**
   * Sirens: a strong sound path to a French node that the meaning channel never
   * reaches — the sound says "yes", the sense says "no". These are the traps a
   * naive homophonic translator falls into; we surface them on purpose.
   */
  findSirens(minSound = 0.6, opts: ClosureOptions = {}): Siren[] {
    const out: Siren[] = [];
    for (const node of this.nodes.values()) {
      if (node.lang !== "en") continue;
      const sem = this.closure(node.id, "semantic", opts);
      const snd = this.closure(node.id, "homophonic", opts);
      for (const [targetId, sound] of snd) {
        const target = this.nodes.get(targetId);
        if (!target || target.lang !== "fr") continue;
        if (sound.score < minSound) continue;
        if (sem.has(targetId)) continue; // it is a real convergence, not a siren
        out.push({
          from: node.id,
          to: targetId,
          sound: sound.score,
          soundPath: sound.path,
          note: `sounds like "${node.text}" but means "${target.gloss}"`,
        });
      }
    }
    return out.sort((a, b) => b.sound - a.sound);
  }

  /**
   * Inject homophonic edges by scoring every EN×FR node pair. This is the
   * "hunting": discovery of sound-alikes beyond the hand-curated floor.
   * `scorer` returns similarity in [0,1]; pairs at/above `threshold` get edges.
   */
  mineHomophonic(
    scorer: (a: MapNode, b: MapNode) => number,
    threshold = 0.6,
  ): number {
    const en: MapNode[] = [];
    const fr: MapNode[] = [];
    for (const n of this.nodes.values()) (n.lang === "en" ? en : fr).push(n);
    let added = 0;
    for (const a of en) {
      for (const b of fr) {
        const s = scorer(a, b);
        if (s >= threshold) {
          this.addEdge(a.id, b.id, "homophonic", round(s), `auto-phonetic ≈${s.toFixed(2)}`);
          added += 1;
        }
      }
    }
    return added;
  }
}

// ---------------------------------------------------------------------------
// Shared phonetic scorer — a small, dependency-free orthography→pseudo-sound
// mapper plus normalised Levenshtein. Good enough to mine obvious sound-alikes;
// replace with the real LLM `phonemeChainScore` for production quality.
// ---------------------------------------------------------------------------

function stripDiacritics(s: string): string {
  return s.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

/** Collapse a word to a coarse pronounced skeleton. Heuristic, not IPA-grade. */
export function pseudoPhonetic(text: string, lang: Lang): string {
  let s = stripDiacritics(text.toLowerCase()).replace(/[^a-z]/g, "");

  // shared digraphs
  s = s
    .replace(/ph/g, "f")
    .replace(/sch/g, "S")
    .replace(/sh/g, "S")
    .replace(/ch/g, "S")
    .replace(/th/g, "t")
    .replace(/ck/g, "k")
    .replace(/qu/g, "k")
    .replace(/gn/g, "n");

  if (lang === "fr") {
    s = s
      .replace(/eau/g, "o")
      .replace(/au/g, "o")
      .replace(/ou/g, "u")
      .replace(/oi/g, "wa")
      .replace(/ai|ei/g, "e")
      .replace(/eu|oeu/g, "o")
      .replace(/in|im|ain|ein/g, "5") // nasal-ish marker
      .replace(/on|om/g, "6")
      .replace(/an|am|en|em/g, "7")
      .replace(/h/g, "")
      .replace(/(ent)$/g, "") // silent verb ending
      .replace(/(er|ez|es|e)$/g, "") // weak final vowels
      .replace(/(s|t|d|x|z|p|g)$/g, ""); // common silent final consonants
  } else {
    s = s
      .replace(/kn/g, "n")
      .replace(/wr/g, "r")
      .replace(/gh/g, "")
      .replace(/x/g, "ks")
      .replace(/c(?=[eiy])/g, "s")
      .replace(/c/g, "k")
      .replace(/e$/g, ""); // silent final e
  }

  // map vowels to coarse classes; collapse doubles
  s = s
    .replace(/[aeiouy]+/g, (m) => m[0])
    .replace(/(.)\1+/g, "$1");
  return s;
}

function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (!m) return n;
  if (!n) return m;
  let prev = new Array<number>(n + 1);
  let cur = new Array<number>(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    cur[0] = i;
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, cur] = [cur, prev];
  }
  return prev[n]!;
}

function round(n: number, p = 4): number {
  return Number(n.toFixed(p));
}
