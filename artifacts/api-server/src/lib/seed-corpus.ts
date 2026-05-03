/**
 * Seed phrases used by the EN↔FR reservoir mining pipeline.
 *
 * The mining LLM is asked, for each seed, to invent a *real* sound-alike
 * candidate in the *other* language. Seeds are intentionally a mix of:
 *   - short single words (rich for nonsense-rhyme matches),
 *   - common idioms / clichés (often have famous homophonic translations),
 *   - sentence-length lines (model rosters, nursery rhymes, proverbs).
 *
 * The list is curated, deterministic, and small enough to ship as code so that
 * mining is reproducible. We rely on the LLM to multiply each seed into
 * several candidate cross-lingual matches, scored and graded individually.
 */

export interface Seed {
  text: string;
  /** "en" means seed is English, ask the LLM for FR matches. "fr" means seed is French, ask for EN matches. */
  lang: "en" | "fr";
  bucket: "word" | "phrase" | "sentence";
}

// ---- English seeds ----

const EN_WORDS = [
  "ocean", "river", "mountain", "garden", "feather", "thunder", "shadow",
  "silver", "winter", "harbor", "cellar", "summer", "ladder", "marble", "candle",
  "knee", "shoe", "heart", "moon", "fire", "salt", "bread", "wine", "book",
  "key", "tree", "stone", "cloud", "wave", "bee", "song", "rose", "ring",
  "dream", "ghost", "bone", "cheese", "milk", "water", "earth", "wind", "rain",
  "snow", "star", "field", "hill", "lake", "wood", "iron", "gold", "glass",
  "plate", "spoon", "knife", "table", "chair", "house", "door", "window",
  "letter", "paper", "pencil", "mirror", "ribbon", "garden", "honey", "apple",
];

const EN_PHRASES = [
  "to be or not", "open the door", "let it be", "in the rain", "by the sea",
  "out of the blue", "over the moon", "under the stars", "after the rain",
  "before the dawn", "near the river", "behind the door", "through the woods",
  "across the field", "above the clouds", "between the lines", "around the corner",
  "into the night", "without a sound", "with all my heart", "for a long time",
  "from the start", "to the end", "at the gate", "on the table", "in the kitchen",
  "the cat is sleeping", "the dog is running", "the sky is blue",
  "she opened the book", "he closed the door", "the moon is rising",
  "i love you so", "i miss you", "come to me", "tell me a story",
  "sing me a song", "play me a tune", "dance with me", "stay with me",
  "the bird flew away", "the boat sails on", "the bell rings out",
  "go to sleep", "wake up now", "look at the stars", "listen to the rain",
];

const EN_SENTENCES = [
  "Mary had a little lamb whose fleece was white as snow.",
  "The quick brown fox jumps over the lazy dog.",
  "Once upon a time in a faraway kingdom.",
  "All the king's horses and all the king's men.",
  "Twinkle twinkle little star, how I wonder what you are.",
  "A long time ago in a galaxy far away.",
  "She sells seashells by the seashore.",
  "The sun was setting over the quiet harbor.",
  "I wandered lonely as a cloud above the hills.",
  "Roses are red and violets are blue.",
];

// ---- French seeds ----

const FR_WORDS = [
  "chat", "chien", "soleil", "étoile", "fleur", "rivière", "montagne", "jardin",
  "fenêtre", "porte", "table", "chaise", "miroir", "bougie", "lettre",
  "souvenir", "papillon", "lumière", "ombre", "tonnerre", "argent", "hiver",
  "automne", "printemps", "été", "matin", "soir", "nuit", "jour", "ciel",
  "nuage", "pluie", "neige", "mer", "océan", "vague", "sable", "rocher",
  "feuille", "racine", "branche", "écorce", "fruit", "graine", "épine",
  "pain", "vin", "fromage", "miel", "sucre", "sel", "beurre", "lait",
  "couteau", "fourchette", "cuillère", "assiette", "verre", "bouteille",
];

const FR_PHRASES = [
  "il était une fois", "au bord de la mer", "sous la lune", "dans le jardin",
  "près du feu", "loin de chez moi", "tout est beau", "rien à dire",
  "viens chez moi", "ferme les yeux", "ouvre la porte", "écoute le vent",
  "regarde le ciel", "attends un peu", "donne-moi la main", "prends ma main",
  "le chat dort", "le chien aboie", "la pluie tombe", "le soleil brille",
  "la nuit tombe", "le jour se lève", "la mer est calme", "le vent se lève",
  "j'ai faim", "j'ai soif", "j'ai froid", "j'ai chaud", "je suis là",
  "je t'aime", "je te manque", "tu me manques", "on se voit", "à demain",
  "à bientôt", "bonne nuit", "bonjour mon ami", "au revoir mon amour",
  "encore une fois", "pas encore", "déjà fini", "tout doucement", "petit à petit",
  "ça va aller", "n'aie pas peur", "viens danser", "chante avec moi",
];

const FR_SENTENCES = [
  "Le petit chat noir dort sur le tapis du salon.",
  "Au clair de la lune mon ami Pierrot.",
  "Sous le pont Mirabeau coule la Seine.",
  "Il pleut il pleut bergère rentre tes blancs moutons.",
  "Frère Jacques frère Jacques dormez-vous dormez-vous.",
  "Une souris verte qui courait dans l'herbe.",
  "Alouette gentille alouette je te plumerai.",
  "À la claire fontaine m'en allant promener.",
  "Promenons-nous dans les bois pendant que le loup n'y est pas.",
  "Il était un petit navire qui n'avait jamais navigué.",
];

export const SEED_CORPUS: Seed[] = [
  ...EN_WORDS.map((text) => ({ text, lang: "en" as const, bucket: "word" as const })),
  ...EN_PHRASES.map((text) => ({ text, lang: "en" as const, bucket: "phrase" as const })),
  ...EN_SENTENCES.map((text) => ({ text, lang: "en" as const, bucket: "sentence" as const })),
  ...FR_WORDS.map((text) => ({ text, lang: "fr" as const, bucket: "word" as const })),
  ...FR_PHRASES.map((text) => ({ text, lang: "fr" as const, bucket: "phrase" as const })),
  ...FR_SENTENCES.map((text) => ({ text, lang: "fr" as const, bucket: "sentence" as const })),
];
