import { Router, type IRouter, type Request } from "express";
import { db, homophoneReservoirTable } from "@workspace/db";
import { and, desc, eq, gte } from "drizzle-orm";
import { mapWithLimit } from "../lib/concurrency";
import { semanticSimilarity } from "../lib/semantic";
import {
  rankResonance,
  countQuadrants,
  buildMeaningMap,
  DEFAULT_THRESHOLDS,
  type ScoredPair,
  type ResonanceThresholds,
} from "../lib/meaning-graph";

/**
 * Meaning Map routes — the convergence layer that fuses the homophone reservoir
 * (sound) with semantic gloss similarity (sense). See MEANING_MAP.md.
 *
 * These are hand-written Express routes (like reservoir.ts) — they do not
 * depend on the generated OpenAPI client, so the frontend explorer gets a
 * generated hook for them only once the spec is extended (a documented next
 * step). The endpoints work and typecheck on their own today.
 */

const router: IRouter = Router();

// One LLM semantic call per pair; keep a few in flight without hammering the API.
const SEMANTIC_CONCURRENCY = 4;

function num(q: unknown): number | undefined {
  if (typeof q !== "string") return undefined;
  const n = Number(q);
  return Number.isFinite(n) ? n : undefined;
}

function clampLimit(q: unknown, def: number, max: number): number {
  const n = num(q);
  if (n === undefined) return def;
  return Math.max(1, Math.min(max, Math.floor(n)));
}

function thresholds(req: Request): ResonanceThresholds {
  const phon = num(req.query["phon"]);
  const sem = num(req.query["sem"]);
  return {
    phon: phon !== undefined ? Math.max(0, Math.min(1, phon)) : DEFAULT_THRESHOLDS.phon,
    sem: sem !== undefined ? Math.max(0, Math.min(1, sem)) : DEFAULT_THRESHOLDS.sem,
  };
}

/** Pull reservoir rows (optionally tier/minSim filtered), capped, strongest-sound first. */
async function loadPairs(opts: { tier?: string; minSim?: number; limit: number }) {
  const filters: ReturnType<typeof eq>[] = [];
  if (opts.tier === "S" || opts.tier === "A" || opts.tier === "B") {
    filters.push(eq(homophoneReservoirTable.tier, opts.tier));
  }
  if (typeof opts.minSim === "number" && Number.isFinite(opts.minSim)) {
    filters.push(gte(homophoneReservoirTable.similarity, opts.minSim));
  }
  const query = db
    .select()
    .from(homophoneReservoirTable)
    .orderBy(desc(homophoneReservoirTable.similarity), desc(homophoneReservoirTable.id))
    .limit(opts.limit);
  return filters.length > 0 ? await query.where(and(...filters)) : await query;
}

/** Score each row's gloss pair for meaning closeness (cached LLM, bounded concurrency). */
async function scorePairs(rows: Awaited<ReturnType<typeof loadPairs>>): Promise<ScoredPair[]> {
  const scored = await mapWithLimit(rows, SEMANTIC_CONCURRENCY, async (r) => {
    const verdict = await semanticSimilarity(r.enGloss, r.frGloss);
    return {
      id: r.id,
      enPhrase: r.enPhrase,
      frPhrase: r.frPhrase,
      enGloss: r.enGloss,
      frGloss: r.frGloss,
      phonetic: r.similarity,
      semantic: verdict.similarity,
    } satisfies ScoredPair;
  });
  // mapWithLimit stores a thrown error as the array element; keep only valid pairs.
  return scored.filter((s): s is ScoredPair => !!s && typeof (s as ScoredPair).id === "number");
}

/**
 * GET /meaning/resonance — the whittle-down.
 * Ranks reservoir pairs by resonance = sqrt(phonetic × semantic) and labels each
 * with its sound×sense quadrant. Query: tier, minSim, limit, minResonance, phon, sem.
 */
router.get("/meaning/resonance", async (req, res) => {
  const tier = typeof req.query["tier"] === "string" ? req.query["tier"] : undefined;
  const minSim = num(req.query["minSim"]);
  const limit = clampLimit(req.query["limit"], 60, 200);
  const minResonance = num(req.query["minResonance"]) ?? 0;
  const t = thresholds(req);

  const rows = await loadPairs({ tier, minSim, limit });
  const scored = await scorePairs(rows);
  const ranked = rankResonance(scored, t).filter((p) => p.resonance >= minResonance);

  res.json({
    thresholds: t,
    counts: countQuadrants(ranked),
    total: ranked.length,
    pairs: ranked.map((p) => ({
      id: p.id,
      enPhrase: p.enPhrase,
      frPhrase: p.frPhrase,
      enGloss: p.enGloss,
      frGloss: p.frGloss,
      phonetic: p.phonetic,
      semantic: p.semantic,
      resonance: p.resonance,
      quadrant: p.quadrant,
    })),
  });
});

/**
 * GET /meaning/map — the graph.
 * Builds nodes (phrases), phone edges (reservoir pairs), sense edges (shared
 * meaning), connected "meaning islands", and the phone bridges/resonances
 * between/within them. Query: tier, minSim, limit, phon, sem.
 */
router.get("/meaning/map", async (req, res) => {
  const tier = typeof req.query["tier"] === "string" ? req.query["tier"] : undefined;
  const minSim = num(req.query["minSim"]);
  const limit = clampLimit(req.query["limit"], 80, 300);
  const t = thresholds(req);

  const rows = await loadPairs({ tier, minSim, limit });
  const scored = await scorePairs(rows);
  const map = buildMeaningMap(scored, t);

  res.json({
    thresholds: t,
    islandCount: map.islandCount,
    nodeCount: map.nodes.length,
    bridgeCount: map.bridges.length,
    resonanceCount: map.resonances.length,
    nodes: map.nodes,
    edges: map.edges,
    bridges: map.bridges,
    resonances: map.resonances,
  });
});

export default router;
