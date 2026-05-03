import { pgTable, serial, text, real, integer, jsonb, timestamp, uniqueIndex, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const homophoneReservoirTable = pgTable(
  "homophone_reservoir",
  {
    id: serial("id").primaryKey(),
    enPhrase: text("en_phrase").notNull(),
    frPhrase: text("fr_phrase").notNull(),
    enGloss: text("en_gloss").notNull().default(""),
    frGloss: text("fr_gloss").notNull().default(""),
    similarity: real("similarity").notNull(),
    componentScores: jsonb("component_scores").notNull().default([]),
    enCoherence: integer("en_coherence").notNull().default(0),
    frCoherence: integer("fr_coherence").notNull().default(0),
    tier: text("tier").notNull(),
    source: text("source").notNull(),
    seed: text("seed"),
    rationale: text("rationale"),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (t) => ({
    uniquePair: uniqueIndex("homophone_reservoir_pair_uk").on(t.enPhrase, t.frPhrase),
    tierIdx: index("homophone_reservoir_tier_idx").on(t.tier),
    similarityIdx: index("homophone_reservoir_similarity_idx").on(t.similarity),
  }),
);

export const insertHomophoneReservoirSchema = createInsertSchema(homophoneReservoirTable).omit({
  id: true,
  createdAt: true,
});

export type InsertHomophoneReservoir = z.infer<typeof insertHomophoneReservoirSchema>;
export type HomophoneReservoirRow = typeof homophoneReservoirTable.$inferSelect;
