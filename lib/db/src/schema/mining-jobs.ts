import { pgTable, serial, text, integer, jsonb, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const miningJobsTable = pgTable("mining_jobs", {
  id: serial("id").primaryKey(),
  status: text("status").notNull(),
  startedAt: timestamp("started_at").defaultNow().notNull(),
  finishedAt: timestamp("finished_at"),
  totalsAttempted: integer("totals_attempted").notNull().default(0),
  totalsInserted: integer("totals_inserted").notNull().default(0),
  totalsSkipped: integer("totals_skipped").notNull().default(0),
  totalsFailed: integer("totals_failed").notNull().default(0),
  tierCounts: jsonb("tier_counts").notNull().default({ S: 0, A: 0, B: 0 }),
  config: jsonb("config").notNull().default({}),
  lastError: text("last_error"),
  currentSeed: text("current_seed"),
});

export const insertMiningJobSchema = createInsertSchema(miningJobsTable).omit({
  id: true,
  startedAt: true,
});

export type InsertMiningJob = z.infer<typeof insertMiningJobSchema>;
export type MiningJobRow = typeof miningJobsTable.$inferSelect;
