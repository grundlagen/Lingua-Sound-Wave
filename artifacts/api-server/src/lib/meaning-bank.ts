/**
 * meaning-bank.ts — the seed data for the Meaning Map.
 *
 * This is the human-curated core of the "map of meaning": two pair-banks plus a
 * synonym layer, expressed as plain data so the engine (meaning-map.ts) can run
 * with zero external services (no LLM, no DB, no network).
 *
 *   1. WORDS          — the vocabulary, each tagged with a language-neutral
 *                       `concept`. Concepts are how we *author* the semantic
 *                       pair-bank: words that share a concept are translations
 *                       (cross-language) or synonyms (same-language). The engine
 *                       never reads `concept` to decide a convergence — it only
 *                       uses the edges derived from it — so concept stays an
 *                       authoring convenience and a ground-truth label, not a
 *                       cheat.
 *
 *   2. SEMANTIC_LINKS — extra meaning edges between *different* concepts
 *                       (near-synonyms, hypernyms, poetic senses). This is the
 *                       "synonyms mapped on top" layer that lets meaning chains
 *                       run further than literal dictionary translation.
 *
 *   3. HOMOPHONIC_LINKS — hand-verified EN↔FR sound-alikes. The engine *also*
 *                       auto-mines homophonic edges with the phonetic scorer, so
 *                       this list is a high-confidence floor, not the ceiling.
 *
 * The data is deliberately small, EN↔FR, and skewed toward cases that make the
 * three interesting outcomes visible: cognate convergences (sound AND meaning),
 * transitive convergences (meaning reached only through a synonym chain), and
 * "sirens" / false friends (sound without meaning).
 */

export type Lang = "en" | "fr";

export interface BankWord {
  text: string;
  lang: Lang;
  /** Short human gloss, shown in output. */
  gloss: string;
  /** Language-neutral concept key. Shared concept ⇒ authored semantic edge. */
  concept: string;
}

/** Near-synonym / sense edge across two concepts. Weight in (0,1]. */
export interface SemanticLink {
  a: string; // "lang:text"
  b: string; // "lang:text"
  weight: number;
  reason: string;
}

/** Hand-verified cross-lingual sound-alike. Weight in (0,1]. */
export interface HomophonicLink {
  a: string; // "lang:text"
  b: string; // "lang:text"
  weight: number;
  reason: string;
}

// ---------------------------------------------------------------------------
// 1. Vocabulary — the two pair-banks live here, joined by `concept`.
// ---------------------------------------------------------------------------

export const WORDS: BankWord[] = [
  // --- SEA (cognate-ish, plus the archaic English poetic synonym "main") ---
  { text: "sea", lang: "en", gloss: "the sea", concept: "SEA" },
  { text: "ocean", lang: "en", gloss: "the ocean", concept: "SEA" },
  { text: "main", lang: "en", gloss: "the open sea (archaic, poetic)", concept: "SEA" },
  { text: "mer", lang: "fr", gloss: "the sea", concept: "SEA" },
  { text: "océan", lang: "fr", gloss: "the ocean", concept: "SEA" },

  // --- HAND (gives us the main/main false friend) ---
  { text: "hand", lang: "en", gloss: "the hand", concept: "HAND" },
  { text: "main", lang: "fr", gloss: "the hand", concept: "HAND" },

  // --- BREAD vs PAIN (the classic pain/pain false friend) ---
  { text: "bread", lang: "en", gloss: "bread", concept: "BREAD" },
  { text: "pain", lang: "fr", gloss: "bread", concept: "BREAD" },
  { text: "pain", lang: "en", gloss: "physical pain", concept: "SUFFERING" },
  { text: "ache", lang: "en", gloss: "a dull pain", concept: "SUFFERING" },
  { text: "douleur", lang: "fr", gloss: "pain, suffering", concept: "SUFFERING" },

  // --- CAT / DOG ---
  { text: "cat", lang: "en", gloss: "a cat", concept: "CAT" },
  { text: "chat", lang: "fr", gloss: "a cat", concept: "CAT" },
  { text: "dog", lang: "en", gloss: "a dog", concept: "DOG" },
  { text: "chien", lang: "fr", gloss: "a dog", concept: "DOG" },

  // --- Cognate convergences (sound AND meaning line up) ---
  { text: "blue", lang: "en", gloss: "the colour blue", concept: "BLUE" },
  { text: "bleu", lang: "fr", gloss: "the colour blue", concept: "BLUE" },
  { text: "rose", lang: "en", gloss: "a rose / pink", concept: "ROSE" },
  { text: "rose", lang: "fr", gloss: "a rose / pink", concept: "ROSE" },
  { text: "rich", lang: "en", gloss: "wealthy", concept: "RICH" },
  { text: "riche", lang: "fr", gloss: "wealthy", concept: "RICH" },
  { text: "table", lang: "en", gloss: "a table", concept: "TABLE" },
  { text: "table", lang: "fr", gloss: "a table", concept: "TABLE" },
  { text: "fruit", lang: "en", gloss: "fruit", concept: "FRUIT" },
  { text: "fruit", lang: "fr", gloss: "fruit", concept: "FRUIT" },
  { text: "orange", lang: "en", gloss: "orange (fruit/colour)", concept: "ORANGE" },
  { text: "orange", lang: "fr", gloss: "orange (fruit/colour)", concept: "ORANGE" },

  // --- JOY: the "content" double-act (transitive feat material) ---
  { text: "happy", lang: "en", gloss: "happy", concept: "JOY" },
  { text: "glad", lang: "en", gloss: "glad, pleased", concept: "JOY" },
  { text: "content", lang: "en", gloss: "content, satisfied", concept: "JOY" },
  { text: "content", lang: "fr", gloss: "content, happy", concept: "JOY" },
  { text: "heureux", lang: "fr", gloss: "happy", concept: "JOY" },

  // --- MERCY / THANKS (near-meaning, near-sound — historically linked) ---
  { text: "mercy", lang: "en", gloss: "mercy, clemency", concept: "MERCY" },
  { text: "merci", lang: "fr", gloss: "thanks", concept: "THANKS" },
  { text: "clémence", lang: "fr", gloss: "mercy, clemency", concept: "MERCY" },

  // --- A few more banks for sound-mining to chew on ---
  { text: "wine", lang: "en", gloss: "wine", concept: "WINE" },
  { text: "vin", lang: "fr", gloss: "wine", concept: "WINE" },
  { text: "fire", lang: "en", gloss: "fire", concept: "FIRE" },
  { text: "feu", lang: "fr", gloss: "fire", concept: "FIRE" },
  { text: "water", lang: "en", gloss: "water", concept: "WATER" },
  { text: "eau", lang: "fr", gloss: "water", concept: "WATER" },
  { text: "king", lang: "en", gloss: "a king", concept: "KING" },
  { text: "roi", lang: "fr", gloss: "a king", concept: "KING" },

  // --- IF (the "sea/si" siren) ---
  { text: "if", lang: "en", gloss: "if (conditional)", concept: "IF" },
  { text: "si", lang: "fr", gloss: "if / so", concept: "IF" },

  // --- GRAPE / RAISIN: a false friend rescued by a synonym chain.
  //     en:raisin SOUNDS like fr:raisin, but fr:raisin means "grape", not
  //     "dried grape". The meaning channel can still reach it — but only by
  //     hopping en:raisin → en:grape (a raisin IS a dried grape) → fr:raisin.
  //     That two-hop meaning path + one-hop sound path = a TRANSITIVE convergence.
  { text: "grape", lang: "en", gloss: "a grape", concept: "GRAPE" },
  { text: "raisin", lang: "fr", gloss: "a grape", concept: "GRAPE" },
  { text: "raisin", lang: "en", gloss: "a dried grape", concept: "DRIED_GRAPE" },
];

// ---------------------------------------------------------------------------
// 2. Synonym / sense layer — meaning edges across different concepts.
//    (Same-concept edges are derived automatically by the engine.)
// ---------------------------------------------------------------------------

export const SEMANTIC_LINKS: SemanticLink[] = [
  // mercy and thanks are historically the same word (L. mercedem); near-meaning.
  { a: "en:mercy", b: "fr:merci", weight: 0.45, reason: "etymological cousins; meanings have drifted" },
  // a raisin is a dried grape — the synonym hop that rescues the raisin/raisin pair.
  { a: "en:raisin", b: "en:grape", weight: 0.7, reason: "a raisin is a dried grape" },
  // pain (suffering) and bread are unrelated in meaning — NO link (that is the point).
  // poetic: the sea as "the deep"/"the main" already same concept; add ocean↔sea strength is automatic.
];

// ---------------------------------------------------------------------------
// 3. Curated homophonic floor — verified EN↔FR sound-alikes.
// ---------------------------------------------------------------------------

export const HOMOPHONIC_LINKS: HomophonicLink[] = [
  { a: "en:main", b: "fr:main", weight: 0.99, reason: "identical spelling, near-identical sound" },
  { a: "en:pain", b: "fr:pain", weight: 0.92, reason: "/peɪn/ ~ /pɛ̃/ — famous false friend" },
  { a: "en:bread", b: "fr:pain", weight: 0.10, reason: "(meaning match, sound mismatch — placeholder)" },
  { a: "en:sea", b: "fr:si", weight: 0.88, reason: "/siː/ ~ /si/" },
  { a: "en:mercy", b: "fr:merci", weight: 0.90, reason: "/ˈmɜːsi/ ~ /mɛʁsi/" },
  { a: "en:content", b: "fr:content", weight: 0.85, reason: "shared spelling, close vowels" },
  { a: "en:raisin", b: "fr:raisin", weight: 0.95, reason: "identical spelling, near-identical sound" },
];
