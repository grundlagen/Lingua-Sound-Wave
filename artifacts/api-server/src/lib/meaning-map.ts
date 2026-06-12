/**
 * The Meaning Map — a resonance graph over meaning-chains and sound-chains.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * The hunt
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

export interface PhraseNode {
  /** Stable id: `${lang}:${normalized text}`. */
  id: string;
  text: string;
  lang: Lang;
}

export type EdgeKind = "meaning" | "sound";

export interface Edge {
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
  sound: Edge;
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

export function nodeId(text: string, lang: Lang): string {
  return `${lang}:${normalize(text)}`;
}

export function normalize(text: string): string {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "") // strip diacritics for id stability
    .replace(/[^a-z0-9 ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// ───────────────────────── Union-Find (meaning clusters) ─────────────────

class UnionFind {
  private parent = new Map<string, string>();
  private rank = new Map<string, number>();

  add(id: string): void {
    if (!this.parent.has(id)) {
      this.parent.set(id, id);
      this.rank.set(id, 0);
    }
  }

  find(id: string): string {
    let root = id;
    while (this.parent.get(root) !== root) {
      root = this.parent.get(root)!;
    }
    // path compression
    let cur = id;
    while (this.parent.get(cur) !== root) {
      const next = this.parent.get(cur)!;
      this.parent.set(cur, root);
      cur = next;
    }
    return root;
  }

  union(a: string, b: string): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return;
    const rka = this.rank.get(ra)!;
    const rkb = this.rank.get(rb)!;
    if (rka < rkb) {
      this.parent.set(ra, rb);
    } else if (rka > rkb) {
      this.parent.set(rb, ra);
    } else {
      this.parent.set(rb, ra);
      this.rank.set(ra, rka + 1);
    }
  }
}

// ───────────────────────────── The graph ────────────────────────────────

export interface MeaningMapOptions {
  /** Meaning edges below this weight don't fuse clusters or count as hops. */
  meaningThreshold?: number;
  /** Sound edges below this weight aren't considered resonances at all. */
  soundThreshold?: number;
  /** Max meaning hops to search when measuring endpoint proximity. */
  maxMeaningHops?: number;
  /** Per-hop multiplicative decay applied to meaningProximity (0<d≤1). */
  hopDecay?: number;
}

const DEFAULTS: Required<MeaningMapOptions> = {
  meaningThreshold: 0.6,
  soundThreshold: 0.6,
  maxMeaningHops: 3,
  hopDecay: 0.6,
};

export class MeaningMap {
  private nodes = new Map<string, PhraseNode>();
  private meaningEdges: Edge[] = [];
  private soundEdges: Edge[] = [];
  /** adjacency for meaning graph: id -> [{to, weight}] (only edges ≥ threshold). */
  private meaningAdj = new Map<string, { to: string; weight: number }[]>();
  private uf = new UnionFind();
  private opts: Required<MeaningMapOptions>;
  private dirty = true;

  constructor(options: MeaningMapOptions = {}) {
    this.opts = { ...DEFAULTS, ...options };
  }

  addNode(text: string, lang: Lang): PhraseNode {
    const id = nodeId(text, lang);
    let node = this.nodes.get(id);
    if (!node) {
      node = { id, text, lang };
      this.nodes.set(id, node);
      this.uf.add(id);
      this.dirty = true;
    }
    return node;
  }

  /** Add a meaning edge (translation or synonym). Adds endpoints if needed. */
  addMeaningEdge(aText: string, aLang: Lang, bText: string, bLang: Lang, weight: number, source?: string): void {
    const a = this.addNode(aText, aLang);
    const b = this.addNode(bText, bLang);
    if (a.id === b.id) return;
    this.meaningEdges.push({ kind: "meaning", a: a.id, b: b.id, weight: clamp01(weight), source });
    this.dirty = true;
  }

  /** Add a sound edge (homophone / sound-alike). Usually cross-lingual. */
  addSoundEdge(aText: string, aLang: Lang, bText: string, bLang: Lang, weight: number, source?: string): void {
    const a = this.addNode(aText, aLang);
    const b = this.addNode(bText, bLang);
    if (a.id === b.id) return;
    this.soundEdges.push({ kind: "sound", a: a.id, b: b.id, weight: clamp01(weight), source });
    this.dirty = true;
  }

  private rebuild(): void {
    this.uf = new UnionFind();
    for (const id of this.nodes.keys()) this.uf.add(id);
    this.meaningAdj = new Map();
    for (const e of this.meaningEdges) {
      // The soft semantic graph keeps EVERY meaning edge so proximity search
      // can reach across not-yet-fused clusters (that reach is the frontier).
      // Only edges at or above threshold actually FUSE clusters in union-find;
      // a weak synonym hint links two phrases for the hunt without yet
      // declaring them the same concept.
      pushAdj(this.meaningAdj, e.a, e.b, e.weight);
      pushAdj(this.meaningAdj, e.b, e.a, e.weight);
      if (e.weight >= this.opts.meaningThreshold) {
        this.uf.union(e.a, e.b);
      }
    }
    this.dirty = false;
  }

  private ensure(): void {
    if (this.dirty) this.rebuild();
  }

  /** Meaning clusters as arrays of nodes, largest first. */
  clusters(): PhraseNode[][] {
    this.ensure();
    const byRoot = new Map<string, PhraseNode[]>();
    for (const node of this.nodes.values()) {
      const root = this.uf.find(node.id);
      const arr = byRoot.get(root) ?? [];
      arr.push(node);
      byRoot.set(root, arr);
    }
    return [...byRoot.values()].sort((a, b) => b.length - a.length);
  }

  /**
   * Shortest meaning-chain between two nodes, bounded by maxMeaningHops.
   * Returns hops 0 with bottleneck 1 if they are the same node.
   * Uses BFS for fewest hops, then reports the bottleneck of the found path.
   */
  meaningChain(aId: string, bId: string): MeaningChain | null {
    this.ensure();
    if (aId === bId) return { path: [aId], hops: 0, bottleneck: 1 };
    const maxHops = this.opts.maxMeaningHops;
    const prev = new Map<string, string>();
    const visited = new Set<string>([aId]);
    let frontier = [aId];
    let depth = 0;
    while (frontier.length && depth < maxHops) {
      const next: string[] = [];
      for (const cur of frontier) {
        for (const { to } of this.meaningAdj.get(cur) ?? []) {
          if (visited.has(to)) continue;
          visited.add(to);
          prev.set(to, cur);
          if (to === bId) {
            return this.reconstruct(prev, aId, bId);
          }
          next.push(to);
        }
      }
      frontier = next;
      depth++;
    }
    return null;
  }

  private reconstruct(prev: Map<string, string>, aId: string, bId: string): MeaningChain {
    const path: string[] = [bId];
    let cur = bId;
    while (cur !== aId) {
      cur = prev.get(cur)!;
      path.push(cur);
    }
    path.reverse();
    let bottleneck = 1;
    for (let i = 0; i + 1 < path.length; i++) {
      const w = this.edgeWeight(path[i]!, path[i + 1]!);
      bottleneck = Math.min(bottleneck, w);
    }
    return { path, hops: path.length - 1, bottleneck };
  }

  private edgeWeight(aId: string, bId: string): number {
    for (const { to, weight } of this.meaningAdj.get(aId) ?? []) {
      if (to === bId) return weight;
    }
    return 0;
  }

  /**
   * Score every sound edge by how close its endpoints sit in meaning space,
   * and return them sorted strongest-first. This is the map of meaning.
   */
  resonances(): Resonance[] {
    this.ensure();
    const out: Resonance[] = [];
    for (const sound of this.soundEdges) {
      if (sound.weight < this.opts.soundThreshold) continue;
      const a = this.nodes.get(sound.a)!;
      const b = this.nodes.get(sound.b)!;
      const chain = this.meaningChain(sound.a, sound.b);
      const meaningHops = chain ? chain.hops : null;
      const perfect = meaningHops === 0 || (chain !== null && this.uf.find(sound.a) === this.uf.find(sound.b));
      const proximity = this.meaningProximity(chain);
      const score = sound.weight * proximity;
      out.push({ sound, a, b, perfect, meaningHops, chain, score });
    }
    return out.sort((x, y) => y.score - x.score);
  }

  /**
   * The frontier: sound edges whose endpoints sound alike and are *near* in
   * meaning but not yet fused (1+ hops, or just over the search horizon).
   * These are the hunt targets for the next routine — find the missing
   * synonym/translation that would close the loop.
   */
  frontier(): Resonance[] {
    return this.resonances().filter((r) => !r.perfect && r.meaningHops !== null);
  }

  /** meaningProximity ∈ [0,1]: 1 for same cluster, decaying per hop, gated by bottleneck. */
  private meaningProximity(chain: MeaningChain | null): number {
    if (!chain) return 0;
    if (chain.hops === 0) return 1;
    return Math.pow(this.opts.hopDecay, chain.hops) * chain.bottleneck;
  }

  getNode(id: string): PhraseNode | undefined {
    return this.nodes.get(id);
  }

  stats(): { nodes: number; meaningEdges: number; soundEdges: number; clusters: number } {
    this.ensure();
    return {
      nodes: this.nodes.size,
      meaningEdges: this.meaningEdges.length,
      soundEdges: this.soundEdges.length,
      clusters: this.clusters().length,
    };
  }
}

// ─────────────────────── helpers ───────────────────────

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function pushAdj(map: Map<string, { to: string; weight: number }[]>, from: string, to: string, weight: number): void {
  const arr = map.get(from) ?? [];
  arr.push({ to, weight });
  map.set(from, arr);
}

// ─────────────── offline phonetic fallback (no model needed) ───────────────
//
// A crude, language-agnostic sound key + similarity so the graph can be
// exercised without the LLM G2P pipeline. NOT a replacement for
// phonemeChainScore — it collapses orthography toward a shared coarse
// phonetic skeleton (good enough to surface cognates and obvious
// sound-alikes in demos and tests). Swap in phonemeChainScore in production.

export function crudePhoneticKey(text: string): string {
  let s = normalize(text).replace(/ /g, "");
  // common EN/FR digraphs → single symbols
  s = s
    .replace(/eau|au|o/g, "o")
    .replace(/ph/g, "f")
    .replace(/qu|q|ck|c(?=[^eiy])|k/g, "k")
    .replace(/c(?=[eiy])|s+|z|ç/g, "s")
    .replace(/ch|sh|j|g(?=[eiy])/g, "x") // hush/affricate-ish bucket
    .replace(/tion|sion/g, "son")
    .replace(/ai|ei|ay|ey|e+/g, "e")
    .replace(/ou|w|u+/g, "u")
    .replace(/in|im|ain|ein|an|en|on|om|un/g, "n") // FR nasal bucket
    .replace(/y|i+/g, "i")
    .replace(/h/g, "");
  // drop doubled letters
  s = s.replace(/(.)\1+/g, "$1");
  // collapse vowels to skeleton positions but keep them (they carry the tune)
  return s;
}

/** Normalized similarity in [0,1] from edit distance over crude phonetic keys. */
export function crudePhoneticSimilarity(aText: string, bText: string): number {
  const a = crudePhoneticKey(aText);
  const b = crudePhoneticKey(bText);
  if (!a.length && !b.length) return 1;
  const d = levenshtein(a, b);
  return 1 - d / Math.max(a.length, b.length);
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
