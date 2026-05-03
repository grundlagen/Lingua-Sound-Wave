import { openai } from "@workspace/integrations-openai-ai-server";
import { db, homophoneReservoirTable, miningJobsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { synthesize } from "./tts";
import { getScoringMethod } from "./scoring";
import { mapWithLimit } from "./concurrency";
import { logger } from "./logger";
import { gradePair, type Tier } from "./tier-grader";
import { SEED_CORPUS, type Seed } from "./seed-corpus";

const MIN_SIM_FLOOR = 0.55;
const CANDIDATES_PER_SEED = 4;
const SEED_CONCURRENCY = 3;
const HYBRID_METHOD_ID = "hybrid-phoneme-audio";

interface RunningJob {
  id: number;
  cancelled: boolean;
}

let active: RunningJob | null = null;

export interface JobConfig {
  maxSeeds?: number;
  /** Skip seeds that have already produced at least one inserted row (resumability). */
  resume?: boolean;
}

/** Generate N homophonic candidates in the *other* language for a single seed. */
async function generateCandidatesForSeed(seed: Seed, count: number): Promise<string[]> {
  const sourceLangName = seed.lang === "en" ? "English" : "French";
  const targetLangName = seed.lang === "en" ? "French" : "English";

  const prompt = `You are a master of cross-lingual homophonic translation between English and French — the art of writing real text in one language whose spoken sound mimics text in the other language (à la Mots d'Heures: Gousses, Rames mimicking Mother Goose Rhymes).

Source ${sourceLangName} text:
"""${seed.text}"""

Produce ${count} candidate ${targetLangName} phrases, each:
  • Real, well-formed ${targetLangName} (only ${targetLangName} words / orthography; punctuation OK).
  • When spoken aloud by a native ${targetLangName} speaker at normal speed, sounds as much as possible like the source ${sourceLangName} text.
  • Should aim for plausibly meaningful ${targetLangName} text — surreal is fine, total nonsense is undesirable.
  • Match the syllable count and stress pattern of the source closely.
  • Each candidate should be a meaningfully different attempt (different word choices), not a tiny variation of the others.

Return strict JSON:
{ "candidates": ["...", "...", ...] }`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-5.4",
      messages: [
        { role: "system", content: "You are a meticulous bilingual phonetician. Return strict JSON only." },
        { role: "user", content: prompt },
      ],
      response_format: { type: "json_object" },
    });
    const txt = completion.choices[0]?.message?.content ?? "{}";
    const parsed = JSON.parse(txt) as { candidates?: unknown };
    if (!Array.isArray(parsed.candidates)) return [];
    return parsed.candidates
      .filter((c): c is string => typeof c === "string" && c.trim().length > 0)
      .map((c) => c.trim())
      .slice(0, count);
  } catch (err) {
    logger.warn({ err, seed: seed.text }, "mining: candidate generation failed");
    return [];
  }
}

/** Score a single (en, fr) pair with the hybrid method. */
async function scorePair(enText: string, frText: string) {
  const method = getScoringMethod(HYBRID_METHOD_ID);
  const [aEn, aFr] = await Promise.all([synthesize(enText), synthesize(frText)]);
  const r = await method.score(
    { ...aEn, text: enText, language: "en", languageName: "English" },
    { ...aFr, text: frText, language: "fr", languageName: "French" },
  );
  return r;
}

/** Process one seed: generate candidates, score, grade, insert survivors. Returns counts. */
async function processSeed(
  seed: Seed,
  _jobId: number,
): Promise<{ attempted: number; inserted: number; skipped: number; failed: number; tierCounts: Record<Tier, number> }> {
  const tierCounts: Record<Tier, number> = { S: 0, A: 0, B: 0 };
  let attempted = 0;
  let inserted = 0;
  let skipped = 0;
  let failed = 0;

  if (active?.cancelled) return { attempted, inserted, skipped, failed, tierCounts };

  const candidates = await generateCandidatesForSeed(seed, CANDIDATES_PER_SEED);
  for (const cand of candidates) {
    if (active?.cancelled) break;
    attempted++;

    const enText = seed.lang === "en" ? seed.text : cand;
    const frText = seed.lang === "en" ? cand : seed.text;

    try {
      const score = await scorePair(enText, frText);
      if (score.similarity < MIN_SIM_FLOOR) {
        skipped++;
        continue;
      }
      const grade = await gradePair(enText, frText, score.similarity);
      const components = score.components ?? [];

      try {
        const result = await db
          .insert(homophoneReservoirTable)
          .values({
            enPhrase: enText,
            frPhrase: frText,
            enGloss: grade.enGloss,
            frGloss: grade.frGloss,
            similarity: score.similarity,
            componentScores: components,
            enCoherence: grade.enCoherence,
            frCoherence: grade.frCoherence,
            tier: grade.tier,
            source: "mined",
            seed: seed.text,
            rationale: grade.rationale,
          })
          .onConflictDoNothing({ target: [homophoneReservoirTable.enPhrase, homophoneReservoirTable.frPhrase] })
          .returning({ id: homophoneReservoirTable.id });
        if (result.length > 0) {
          inserted++;
          tierCounts[grade.tier]++;
        } else {
          skipped++;
        }
      } catch (insErr) {
        failed++;
        logger.warn({ err: insErr, en: enText, fr: frText }, "mining: insert failed");
      }
    } catch (err) {
      failed++;
      logger.warn({ err, en: enText, fr: frText }, "mining: scoring/grading failed");
    }

  }

  return { attempted, inserted, skipped, failed, tierCounts };
}

/** Async runner — loop over the seed corpus, score and insert. */
async function runJobLoop(jobId: number, cfg: JobConfig): Promise<void> {
  const max = cfg.maxSeeds ?? SEED_CORPUS.length;
  const seeds = SEED_CORPUS.slice(0, max);

  let totalAttempted = 0;
  let totalInserted = 0;
  let totalSkipped = 0;
  let totalFailed = 0;
  const tierCounts: Record<Tier, number> = { S: 0, A: 0, B: 0 };

  await mapWithLimit(seeds, SEED_CONCURRENCY, async (seed) => {
    if (active?.cancelled) return;
    await db.update(miningJobsTable).set({ currentSeed: seed.text }).where(eq(miningJobsTable.id, jobId));
    try {
      const r = await processSeed(seed, jobId);
      totalAttempted += r.attempted;
      totalInserted += r.inserted;
      totalSkipped += r.skipped;
      totalFailed += r.failed;
      tierCounts.S += r.tierCounts.S;
      tierCounts.A += r.tierCounts.A;
      tierCounts.B += r.tierCounts.B;

      // Batched persistence after each seed completes.
      await db
        .update(miningJobsTable)
        .set({
          totalsAttempted: totalAttempted,
          totalsInserted: totalInserted,
          totalsSkipped: totalSkipped,
          totalsFailed: totalFailed,
          tierCounts,
        })
        .where(eq(miningJobsTable.id, jobId));
    } catch (err) {
      logger.error({ err, seed: seed.text }, "mining: seed crashed");
      totalFailed++;
    }
  });

  const finalStatus = active?.cancelled ? "cancelled" : "done";
  await db
    .update(miningJobsTable)
    .set({
      status: finalStatus,
      finishedAt: new Date(),
      totalsAttempted: totalAttempted,
      totalsInserted: totalInserted,
      totalsSkipped: totalSkipped,
      totalsFailed: totalFailed,
      tierCounts,
      currentSeed: null,
    })
    .where(eq(miningJobsTable.id, jobId));

  active = null;
  logger.info({ jobId, totalInserted, tierCounts, finalStatus }, "mining: job complete");
}

/** Public: start a mining job. Throws if one is already running. */
export async function startMiningJob(cfg: JobConfig = {}): Promise<number> {
  if (active) throw new Error("A mining job is already running");
  const [row] = await db
    .insert(miningJobsTable)
    .values({ status: "running", config: cfg })
    .returning({ id: miningJobsTable.id });
  if (!row) throw new Error("Failed to create mining job row");
  active = { id: row.id, cancelled: false };
  // Fire-and-forget; errors logged inside.
  runJobLoop(row.id, cfg).catch(async (err) => {
    logger.error({ err, jobId: row.id }, "mining: top-level loop error");
    await db
      .update(miningJobsTable)
      .set({ status: "failed", finishedAt: new Date(), lastError: err instanceof Error ? err.message : String(err) })
      .where(eq(miningJobsTable.id, row.id));
    active = null;
  });
  return row.id;
}

/** Public: signal the current job to stop after the current seed. */
export function cancelMiningJob(jobId: number): boolean {
  if (active && active.id === jobId) {
    active.cancelled = true;
    return true;
  }
  return false;
}

/** Public: id of the currently-running job, if any. */
export function currentJobId(): number | null {
  return active ? active.id : null;
}

export { SEED_CORPUS };
