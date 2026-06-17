/**
 * meaning-map.ts — the Meaning Map engine.
 *
 * The grand goal (see MEANING-MAP.md): fuse two pair-banks into one graph and
 * walk the connecting chains until the same meanings, approached from sound and
 * from sense, *converge*.
 *
 *   • SEMANTIC edges preserve meaning, destroy sound  (translations, synonyms)
 *   • HOMOPHONIC edges preserve sound, destroy meaning (sound-alikes)
 *
 * A node A in English and a node F in French are a CONVERGENCE when there is
 * BOTH a meaning-only path A⇒F (so F means what A means) AND a sound-only path
 * A≈F (so F sounds like A). The two paths are independent: that independence is
 * exactly what makes the pair a real homophonic translation rather than a
 * coincidence of one channel.
 *
 *   - cognate convergence : both paths length 1 (e.g. blue / bleu)
 *   - transitive convergence: the meaning path runs through a synonym hop —
 *     the feat the map exists to find.
 *   - siren / false friend : a strong sound path with NO meaning path
 *     (e.g. en:pain ≈ fr:pain "bread"). Kept and flagged, never silently dropped.
 *
 * "Whittling down the same meanings over time" is modelled literally: rounds of
 * rising thresholds that prune the atlas toward its gold core, keeping one best
 * representative per meaning cluster.
 *
 * Pure TypeScript, no workspace imports — runs under `node --experimental-strip-types`.
 * The default phonetic scorer is a deliberately simple heuristic; the real
 * `phoneme.ts` LLM chain scorer can be injected via `mineHomophonic(scorer)`.
 */

export type Lang = "en" | "fr";
export type Layer = "semantic" | "homophonic";

export interface MapNode {
  id: string; // `${lang}:${text}`
  text: string;
  lang: Lang;
  gloss: string;
  concept?: string;
}

export interface Edge {
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

export function id(lang: Lang, text: string): string {
  return nodeId(lang, text);
}

export class MeaningMap {
  readonly nodes = new Map<string, MapNode>();
  private readonly adj = new Map<string, Edge[]>();

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

  private pushEdge(from: string, edge: Edge): void {
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

  edgesFrom(fromId: string): readonly Edge[] {
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
// Default phonetic scorer — a small, dependency-free orthography→pseudo-sound
// mapper plus normalised Levenshtein. Good enough to mine obvious sound-alikes;
// replace with the real LLM `phonemeChainScore` for production quality.
// ---------------------------------------------------------------------------

function stripDiacritics(s: string): string {
  return s.normalize("NFD").replace(/[̀-ͯ]/g, "");
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
  if (m === 0) return n;
  if (n === 0) return m;
  const prev = new Array<number>(n + 1);
  const cur = new Array<number>(n + 1);
  for (let j = 0; j <= n; j += 1) prev[j] = j;
  for (let i = 1; i <= m; i += 1) {
    cur[0] = i;
    for (let j = 1; j <= n; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    for (let j = 0; j <= n; j += 1) prev[j] = cur[j];
  }
  return prev[n];
}

/** Default sound similarity in [0,1] between two map nodes. */
export function defaultPhoneticScore(a: MapNode, b: MapNode): number {
  const pa = pseudoPhonetic(a.text, a.lang);
  const pb = pseudoPhonetic(b.text, b.lang);
  if (!pa.length && !pb.length) return 1;
  const dist = levenshtein(pa, pb);
  const sim = 1 - dist / Math.max(pa.length, pb.length, 1);
  return Math.max(0, sim);
}

function round(x: number): number {
  return Math.round(x * 1000) / 1000;
}

// ---------------------------------------------------------------------------
// Whittling — rounds of rising thresholds, keeping one best pair per meaning
// cluster. Returns a trace so callers can show the atlas converging "over time".
// ---------------------------------------------------------------------------

export interface WhittleRound {
  threshold: number;
  survivors: Convergence[];
  dropped: number;
}

export function whittle(
  convergences: Convergence[],
  thresholds: number[] = [0.4, 0.55, 0.7, 0.85, 0.92],
): WhittleRound[] {
  const rounds: WhittleRound[] = [];
  let pool = [...convergences];
  for (const threshold of thresholds) {
    const passed = pool.filter((c) => c.combined >= threshold);
    // keep the single strongest convergence per English source (one meaning ⇒ one best mate)
    const bestPerSource = new Map<string, Convergence>();
    for (const c of passed) {
      const prev = bestPerSource.get(c.from);
      if (!prev || c.combined > prev.combined) bestPerSource.set(c.from, c);
    }
    const survivors = [...bestPerSource.values()].sort((a, b) => b.combined - a.combined);
    rounds.push({ threshold, survivors, dropped: pool.length - survivors.length });
    pool = survivors;
  }
  return rounds;
}
