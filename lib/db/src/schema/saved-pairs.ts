import { pgTable, serial, text, real, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const savedPairsTable = pgTable("saved_pairs", {
  id: serial("id").primaryKey(),
  sourcePhrase: text("source_phrase").notNull(),
  sourceLanguage: text("source_language").notNull(),
  sourceMeaning: text("source_meaning").notNull(),
  matchPhrase: text("match_phrase").notNull(),
  matchLanguage: text("match_language").notNull(),
  matchMeaning: text("match_meaning").notNull(),
  similarity: real("similarity").notNull(),
  notes: text("notes"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertSavedPairSchema = createInsertSchema(savedPairsTable).omit({
  id: true,
  createdAt: true,
});

export type InsertSavedPair = z.infer<typeof insertSavedPairSchema>;
export type SavedPair = typeof savedPairsTable.$inferSelect;
