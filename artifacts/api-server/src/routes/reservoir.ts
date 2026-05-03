import { Router, type IRouter } from "express";
import { db, homophoneReservoirTable, miningJobsTable } from "@workspace/db";
import { and, desc, eq, gte, ilike, or, sql } from "drizzle-orm";
import { startMiningJob, cancelMiningJob, currentJobId, SEED_CORPUS } from "../lib/reservoir-mining";

const router: IRouter = Router();

router.get("/reservoir/pairs", async (req, res) => {
  const tier = typeof req.query["tier"] === "string" ? req.query["tier"] : undefined;
  const minSim = typeof req.query["minSim"] === "string" ? Number(req.query["minSim"]) : undefined;
  const search = typeof req.query["search"] === "string" ? req.query["search"].trim() : undefined;
  const limitRaw = typeof req.query["limit"] === "string" ? Number(req.query["limit"]) : 100;
  const limit = Math.max(1, Math.min(500, Number.isFinite(limitRaw) ? limitRaw : 100));

  const filters: ReturnType<typeof eq>[] = [];
  if (tier && (tier === "S" || tier === "A" || tier === "B")) {
    filters.push(eq(homophoneReservoirTable.tier, tier));
  }
  if (typeof minSim === "number" && Number.isFinite(minSim)) {
    filters.push(gte(homophoneReservoirTable.similarity, minSim));
  }
  if (search) {
    const pattern = `%${search}%`;
    filters.push(
      or(ilike(homophoneReservoirTable.enPhrase, pattern), ilike(homophoneReservoirTable.frPhrase, pattern))!,
    );
  }

  const rowsQuery = db
    .select()
    .from(homophoneReservoirTable)
    .orderBy(desc(homophoneReservoirTable.similarity), desc(homophoneReservoirTable.id))
    .limit(limit);
  const rows = filters.length > 0 ? await rowsQuery.where(and(...filters)) : await rowsQuery;

  res.json(
    rows.map((r) => ({
      id: r.id,
      enPhrase: r.enPhrase,
      frPhrase: r.frPhrase,
      enGloss: r.enGloss,
      frGloss: r.frGloss,
      similarity: r.similarity,
      enCoherence: r.enCoherence,
      frCoherence: r.frCoherence,
      tier: r.tier,
      source: r.source,
      seed: r.seed ?? "",
      rationale: r.rationale ?? "",
      componentScores: Array.isArray(r.componentScores) ? r.componentScores : [],
      createdAt: r.createdAt.toISOString(),
    })),
  );
});

router.get("/reservoir/stats", async (_req, res) => {
  const counts = await db
    .select({
      tier: homophoneReservoirTable.tier,
      count: sql<number>`count(*)::int`,
    })
    .from(homophoneReservoirTable)
    .groupBy(homophoneReservoirTable.tier);

  const tierCounts: Record<string, number> = { S: 0, A: 0, B: 0 };
  let total = 0;
  for (const c of counts) {
    tierCounts[c.tier] = Number(c.count);
    total += Number(c.count);
  }

  const [avgRow] = await db
    .select({ avg: sql<number>`avg(${homophoneReservoirTable.similarity})::float` })
    .from(homophoneReservoirTable);

  res.json({
    total,
    target: 2500,
    tierCounts,
    averageSimilarity: avgRow?.avg ?? 0,
    seedCount: SEED_CORPUS.length,
  });
});

router.delete("/reservoir/pairs/:id", async (req, res) => {
  const id = Number(req.params["id"]);
  if (!Number.isInteger(id) || id <= 0) {
    res.status(400).json({ error: "Invalid id" });
    return;
  }
  const deleted = await db
    .delete(homophoneReservoirTable)
    .where(eq(homophoneReservoirTable.id, id))
    .returning({ id: homophoneReservoirTable.id });
  if (deleted.length === 0) {
    res.status(404).json({ error: "Not found", id });
    return;
  }
  res.json({ success: true, id });
});

router.post("/reservoir/mine/start", async (req, res) => {
  const maxSeedsRaw = (req.body as { maxSeeds?: unknown })?.maxSeeds;
  const maxSeeds = typeof maxSeedsRaw === "number" && Number.isFinite(maxSeedsRaw) && maxSeedsRaw > 0
    ? Math.min(SEED_CORPUS.length, Math.floor(maxSeedsRaw))
    : undefined;
  try {
    const jobId = await startMiningJob({ maxSeeds: maxSeeds ?? 8 });
    res.status(202).json({ jobId });
  } catch (err) {
    res.status(409).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

router.post("/reservoir/mine/cancel", async (_req, res) => {
  const id = currentJobId();
  if (id === null) {
    res.status(404).json({ error: "No active job" });
    return;
  }
  cancelMiningJob(id);
  res.json({ success: true, jobId: id });
});

router.get("/reservoir/mine/status", async (_req, res) => {
  const [latest] = await db
    .select()
    .from(miningJobsTable)
    .orderBy(desc(miningJobsTable.id))
    .limit(1);

  if (!latest) {
    res.json({ job: null, activeJobId: currentJobId() });
    return;
  }
  const tc = (latest.tierCounts as Record<string, number>) ?? { S: 0, A: 0, B: 0 };
  res.json({
    activeJobId: currentJobId(),
    job: {
      id: latest.id,
      status: latest.status,
      startedAt: latest.startedAt.toISOString(),
      finishedAt: latest.finishedAt ? latest.finishedAt.toISOString() : null,
      totalsAttempted: latest.totalsAttempted,
      totalsInserted: latest.totalsInserted,
      totalsSkipped: latest.totalsSkipped,
      totalsFailed: latest.totalsFailed,
      tierCounts: { S: tc["S"] ?? 0, A: tc["A"] ?? 0, B: tc["B"] ?? 0 },
      currentSeed: latest.currentSeed ?? "",
      lastError: latest.lastError ?? "",
    },
  });
});

export default router;
