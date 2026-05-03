// Phoneme-level cross-lingual similarity.
//
// Pipeline:
//   1. G2P via LLM (cached per phrase+language) → canonical IPA + up to 3
//      pronunciation variants (fast-speech, schwa reduction, devoicing,
//      common allophonic / dialect substitutions). The variants are the
//      "synonym chains" the user asked for — multiple plausible phonetic
//      realizations of the same phrase.
//   2. Tokenize each IPA string (handles digraph affricates like tʃ dʒ ts,
//      strips stress marks, length marks, and tones).
//   3. Featural distance per phone pair: place / manner / voicing for
//      consonants; height / backness / rounding for vowels. Cross-class
//      (vowel↔consonant) gets a high cost.
//   4. Equivalence-class shortcuts encode common cross-lingual substitutions
//      (rhotic family, l-vocalization, TH-fronting, schwa↔reduced vowels,
//      voicing pairs, palatal sibilants, nasal place mismatches, etc.).
//   5. Needleman–Wunsch alignment over phones; cost normalized by alignment
//      length. Best alignment across all (source-variant × target-variant)
//      pairs wins — that is the "best chain."
//   6. Calibrated to similarity in [0, 1].

import { openai } from "@workspace/integrations-openai-ai-server";
import { logger } from "./logger";

// -------------------- IPA feature tables --------------------

interface ConsonantFeatures {
  type: "C";
  place: number;   // 0=bilabial 1=labiodental 2=dental 3=alveolar 4=postalveolar 5=retroflex 6=palatal 7=velar 8=uvular 9=glottal
  manner: number;  // 0=stop 1=fricative 2=affricate 3=nasal 4=lateral 5=approximant 6=trill 7=tap
  voiced: 0 | 1;
}
interface VowelFeatures {
  type: "V";
  height: number;  // 0=close 1=close-mid 2=mid 3=near-open 4=open
  back: number;    // 0=front 1=central 2=back
  round: 0 | 1;
}
type PhoneFeatures = ConsonantFeatures | VowelFeatures;

const C = (p: number, m: number, v: 0 | 1): ConsonantFeatures => ({ type: "C", place: p, manner: m, voiced: v });
const V = (h: number, b: number, r: 0 | 1): VowelFeatures => ({ type: "V", height: h, back: b, round: r });

const CONSONANTS: Record<string, ConsonantFeatures> = {
  // stops
  p: C(0, 0, 0), b: C(0, 0, 1),
  t: C(3, 0, 0), d: C(3, 0, 1),
  ʈ: C(5, 0, 0), ɖ: C(5, 0, 1),
  c: C(6, 0, 0), ɟ: C(6, 0, 1),
  k: C(7, 0, 0), g: C(7, 0, 1), ɡ: C(7, 0, 1),
  q: C(8, 0, 0), ɢ: C(8, 0, 1),
  ʔ: C(9, 0, 0),
  // fricatives
  f: C(1, 1, 0), v: C(1, 1, 1),
  θ: C(2, 1, 0), ð: C(2, 1, 1),
  s: C(3, 1, 0), z: C(3, 1, 1),
  ʃ: C(4, 1, 0), ʒ: C(4, 1, 1),
  ɕ: C(6, 1, 0), ʑ: C(6, 1, 1),
  ʂ: C(5, 1, 0), ʐ: C(5, 1, 1),
  ç: C(6, 1, 0), ʝ: C(6, 1, 1),
  x: C(7, 1, 0), ɣ: C(7, 1, 1),
  χ: C(8, 1, 0), ʁ: C(8, 1, 1),
  h: C(9, 1, 0), ɦ: C(9, 1, 1),
  // nasals
  m: C(0, 3, 1), ɱ: C(1, 3, 1),
  n: C(3, 3, 1), ɳ: C(5, 3, 1), ɲ: C(6, 3, 1), ŋ: C(7, 3, 1), ɴ: C(8, 3, 1),
  // lateral (incl. lateral fricatives)
  l: C(3, 4, 1), ʎ: C(6, 4, 1), ɫ: C(3, 4, 1), ɭ: C(5, 4, 1),
  ɬ: C(3, 1, 0), ɮ: C(3, 1, 1),
  // approximant / glide
  j: C(6, 5, 1), w: C(0, 5, 1), ɥ: C(6, 5, 1),
  ɹ: C(3, 5, 1), ɻ: C(5, 5, 1), ʋ: C(1, 5, 1), ɰ: C(7, 5, 1),
  // trill / tap
  r: C(3, 6, 1), ʀ: C(8, 6, 1),
  ɾ: C(3, 7, 1), ɽ: C(5, 7, 1),
};

// Digraph affricates and a few common compound symbols (greedy 2-char tokens).
const DIGRAPHS: Record<string, ConsonantFeatures> = {
  "tʃ": C(4, 2, 0), "dʒ": C(4, 2, 1),
  "ts": C(3, 2, 0), "dz": C(3, 2, 1),
  "tɕ": C(6, 2, 0), "dʑ": C(6, 2, 1),
  "tʂ": C(5, 2, 0), "dʐ": C(5, 2, 1),
  "pf": C(1, 2, 0), "bv": C(1, 2, 1),
  "kx": C(7, 2, 0),
};

const VOWELS: Record<string, VowelFeatures> = {
  // close
  i: V(0, 0, 0), y: V(0, 0, 1), ɨ: V(0, 1, 0), ʉ: V(0, 1, 1), ɯ: V(0, 2, 0), u: V(0, 2, 1),
  // near-close
  ɪ: V(1, 0, 0), ʏ: V(1, 0, 1), ʊ: V(1, 2, 1),
  // close-mid
  e: V(1, 0, 0), ø: V(1, 0, 1), ɘ: V(1, 1, 0), ɵ: V(1, 1, 1), ɤ: V(1, 2, 0), o: V(1, 2, 1),
  // mid (schwa)
  ə: V(2, 1, 0),
  // open-mid
  ɛ: V(2, 0, 0), œ: V(2, 0, 1), ɜ: V(2, 1, 0), ɞ: V(2, 1, 1), ʌ: V(2, 2, 0), ɔ: V(2, 2, 1),
  // near-open
  æ: V(3, 0, 0), ɐ: V(3, 1, 0),
  // open
  a: V(4, 0, 0), ɶ: V(4, 0, 1), ɑ: V(4, 2, 0), ɒ: V(4, 2, 1),
};

// -------------------- Equivalence classes (linguistic rules) --------------------
//
// These encode "phones that often substitute for each other across languages
// or in casual speech" — much more accurate than raw featural distance for
// the cases we care about.

const EQUIVALENCE: Record<string, number> = {};
function eq(a: string, b: string, cost: number) {
  const key = [a, b].sort().join("|");
  EQUIVALENCE[key] = cost;
}
function eqAll(group: string[], cost: number) {
  for (let i = 0; i < group.length; i++) for (let j = i + 1; j < group.length; j++) eq(group[i]!, group[j]!, cost);
}

eqAll(["r", "ɹ", "ʁ", "ʀ", "ɾ", "ɽ"], 0.10);          // rhotic family
eqAll(["l", "ɫ", "w"], 0.20);                          // l-vocalization
eqAll(["θ", "s", "f"], 0.25);                          // TH-fronting → /s/ or /f/
eqAll(["ð", "z", "d"], 0.25);                          // voiced TH → /z/ or /d/
eqAll(["ʃ", "ɕ", "ʂ", "s"], 0.18);                     // sibilant family
eqAll(["ʒ", "ʑ", "ʐ", "z"], 0.18);
eqAll(["m", "n", "ŋ", "ɲ", "ɴ"], 0.30);                // nasal place mismatches
eqAll(["ʈ", "t"], 0.10); eqAll(["ɖ", "d"], 0.10);      // retroflex ↔ alveolar
eqAll(["c", "k"], 0.15); eqAll(["ɟ", "g"], 0.15);      // palatal ↔ velar stop
eqAll(["q", "k"], 0.15); eqAll(["ɢ", "g"], 0.15);      // uvular ↔ velar stop
eqAll(["ç", "ʃ", "h"], 0.20);
eqAll(["x", "h", "χ"], 0.20);
eqAll(["v", "w"], 0.20); eqAll(["b", "v"], 0.20);
// Schwa reduction: any unstressed lax vowel collapses to schwa-territory.
eqAll(["ə", "ɐ", "ɜ", "ʌ", "ɪ", "ʊ", "ɛ"], 0.15);
// Tense/lax pairs.
eqAll(["i", "ɪ"], 0.10); eqAll(["e", "ɛ"], 0.10);
eqAll(["u", "ʊ"], 0.10); eqAll(["o", "ɔ"], 0.10);
eqAll(["a", "ɑ", "ɐ", "æ"], 0.15);
eqAll(["y", "i"], 0.20); eqAll(["ø", "e"], 0.20);
// Voicing pairs (for final-obstruent devoicing etc.).
eqAll(["p", "b"], 0.20); eqAll(["t", "d"], 0.20); eqAll(["k", "g"], 0.20);
eqAll(["s", "z"], 0.20); eqAll(["f", "v"], 0.20); eqAll(["ʃ", "ʒ"], 0.20);
// Glide ↔ corresponding vowel.
eq("j", "i", 0.20); eq("w", "u", 0.20);

// -------------------- Tokenizer --------------------

const SUFFIX_DIACRITICS = new Set(["ʰ", "ʲ", "ʷ", "ˠ", "ˤ", "̥", "̬", "̃", "̪", "̟", "̠", "̩"]);
// Strip: stress marks, length, tone letters/numbers, syllable separators, hyphens, slashes,
// and tie bars (combining double inverted breve below / above) so "t͡ʃ" → "tʃ".
const STRIP_MARKS = /[ˈˌːˑ˥˦˧˨˩˩˥|‖.\-/\u0361\u035C]/g;

export function tokenizeIPA(s: string): string[] {
  const cleaned = s.normalize("NFC")
    .replace(STRIP_MARKS, "")
    .replace(/[0-9]/g, "")  // tone digits in pinyin-ish IPA
    .replace(/\s+/g, " ")
    .trim();
  const out: string[] = [];
  let i = 0;
  while (i < cleaned.length) {
    const ch = cleaned[i]!;
    if (ch === " ") { i++; continue; }
    if (i + 1 < cleaned.length) {
      const di = cleaned.slice(i, i + 2);
      if (di in DIGRAPHS) { out.push(di); i += 2; continue; }
    }
    if (SUFFIX_DIACRITICS.has(ch)) { i++; continue; }
    if (ch in CONSONANTS || ch in VOWELS) { out.push(ch); i++; continue; }
    // unknown — skip silently
    i++;
  }
  return out;
}

function lookup(token: string): PhoneFeatures | null {
  if (token in DIGRAPHS) return DIGRAPHS[token]!;
  if (token in CONSONANTS) return CONSONANTS[token]!;
  if (token in VOWELS) return VOWELS[token]!;
  return null;
}

// -------------------- Substitution cost --------------------

const C_PLACE_W = 0.40, C_MANNER_W = 0.40, C_VOICE_W = 0.20;
const V_HEIGHT_W = 0.40, V_BACK_W = 0.40, V_ROUND_W = 0.20;
const CV_CROSS_COST = 1.0;
const GAP_COST = 0.5;

export function substitutionCost(a: string, b: string): number {
  if (a === b) return 0;
  const key = [a, b].sort().join("|");
  if (key in EQUIVALENCE) return EQUIVALENCE[key]!;
  const fa = lookup(a), fb = lookup(b);
  if (!fa || !fb) return 0.6;
  if (fa.type !== fb.type) return CV_CROSS_COST;
  if (fa.type === "C" && fb.type === "C") {
    return C_PLACE_W * Math.abs(fa.place - fb.place) / 9
      + C_MANNER_W * Math.abs(fa.manner - fb.manner) / 7
      + C_VOICE_W * Math.abs(fa.voiced - fb.voiced);
  }
  const va = fa as VowelFeatures, vb = fb as VowelFeatures;
  return V_HEIGHT_W * Math.abs(va.height - vb.height) / 4
    + V_BACK_W * Math.abs(va.back - vb.back) / 2
    + V_ROUND_W * Math.abs(va.round - vb.round);
}

// -------------------- Needleman–Wunsch alignment --------------------

export function alignCost(a: string[], b: string[]): { cost: number; aligned: number } {
  const n = a.length, m = b.length;
  if (n === 0 && m === 0) return { cost: 0, aligned: 0 };
  if (n === 0) return { cost: m * GAP_COST, aligned: m };
  if (m === 0) return { cost: n * GAP_COST, aligned: n };
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = 1; i <= n; i++) dp[i]![0] = i * GAP_COST;
  for (let j = 1; j <= m; j++) dp[0]![j] = j * GAP_COST;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const sub = dp[i - 1]![j - 1]! + substitutionCost(a[i - 1]!, b[j - 1]!);
      const del = dp[i - 1]![j]! + GAP_COST;
      const ins = dp[i]![j - 1]! + GAP_COST;
      dp[i]![j] = Math.min(sub, del, ins);
    }
  }
  return { cost: dp[n]![m]!, aligned: Math.max(n, m) };
}

// -------------------- LLM G2P (cached) --------------------

interface G2PResult {
  canonical: string[];
  variants: string[][];
  raw: { canonical: string; variants: string[] };
}
const G2P_CACHE = new Map<string, Promise<G2PResult>>();

export async function phraseToIPA(text: string, language: string, languageName: string): Promise<G2PResult> {
  const key = `${language}::${text}`;
  const cached = G2P_CACHE.get(key);
  if (cached) return cached;
  const p = (async (): Promise<G2PResult | null> => {
    const prompt = `You are a multilingual G2P (grapheme-to-phoneme) expert. Convert the phrase to broad IPA transcription as a native speaker would naturally pronounce it at normal speed. Then provide up to 3 alternate plausible IPA renderings reflecting common pronunciation variants — fast/casual speech, schwa reduction in unstressed syllables, regional/dialect substitutions, final-obstruent devoicing, common allophonic variation. Each variant must differ from the canonical form in a phonologically motivated way (not just a typo).

Phrase: """${text}"""
Language: ${languageName} (code ${language})

Return strict JSON:
{
  "canonical": "<broad IPA, IPA symbols only>",
  "variants": ["<variant1>", "<variant2>", "<variant3>"]
}

Rules:
- IPA symbols only. No romanizations, no pinyin, no jyutping, no Hepburn, no Latin letters used as orthography placeholders.
- Omit stress marks (ˈ ˌ) and tone marks (˥˦˧˨˩) — they will be stripped anyway.
- For tonal languages (Mandarin, Cantonese, Vietnamese, Thai, Yoruba), omit tones; render the segments only.
- Use spaces between word/syllable groups if helpful.
- For multi-word phrases, transcribe the whole phrase.
- Variants should be a small set of plausible alternatives, not nonsense.`;
    try {
      const completion = await openai.chat.completions.create({
        model: "gpt-5.4",
        messages: [
          { role: "system", content: "You are a meticulous multilingual phonetician. Return strict JSON only with IPA transcriptions." },
          { role: "user", content: prompt },
        ],
        response_format: { type: "json_object" },
      });
      const txt = completion.choices[0]?.message?.content ?? "{}";
      const parsed = JSON.parse(txt) as { canonical?: unknown; variants?: unknown };
      const canonical = typeof parsed.canonical === "string" ? parsed.canonical : "";
      const variantsRaw = Array.isArray(parsed.variants)
        ? parsed.variants.filter((v): v is string => typeof v === "string")
        : [];
      return {
        canonical: tokenizeIPA(canonical),
        variants: variantsRaw.map(tokenizeIPA).filter((v) => v.length > 0),
        raw: { canonical, variants: variantsRaw },
      };
    } catch (e) {
      logger.warn({ err: e instanceof Error ? e.message : String(e), text, language }, "phoneme: G2P failed");
      return null;
    }
  })();
  // Don't cache transient LLM failures — drop the entry so the next request retries.
  // Cap cache to bound memory in long-running processes.
  if (G2P_CACHE.size > 5000) G2P_CACHE.clear();
  G2P_CACHE.set(key, p as Promise<G2PResult>);
  const result = await p;
  if (result === null) {
    G2P_CACHE.delete(key);
    return { canonical: [], variants: [], raw: { canonical: "", variants: [] } };
  }
  return result;
}

// -------------------- Top-level scoring --------------------

export interface PhonemeScore {
  distance: number;
  similarity: number;
  bestAlignment: {
    sourceVariant: number;
    targetVariant: number;
    cost: number;
    aligned: number;
    sourceTokens: string[];
    targetTokens: string[];
  };
  sourceCanonical: string;
  targetCanonical: string;
  sourceVariantCount: number;
  targetVariantCount: number;
}

export async function phonemeChainScore(
  text1: string, lang1: string, langName1: string,
  text2: string, lang2: string, langName2: string,
): Promise<PhonemeScore> {
  const [g1, g2] = await Promise.all([
    phraseToIPA(text1, lang1, langName1),
    phraseToIPA(text2, lang2, langName2),
  ]);
  const aSet = [g1.canonical, ...g1.variants].filter((t) => t.length > 0);
  const bSet = [g2.canonical, ...g2.variants].filter((t) => t.length > 0);
  if (aSet.length === 0 || bSet.length === 0) {
    return {
      distance: 1.5,
      similarity: 0,
      bestAlignment: { sourceVariant: -1, targetVariant: -1, cost: 1.5, aligned: 0, sourceTokens: g1.canonical, targetTokens: g2.canonical },
      sourceCanonical: g1.raw.canonical,
      targetCanonical: g2.raw.canonical,
      sourceVariantCount: g1.variants.length,
      targetVariantCount: g2.variants.length,
    };
  }
  let best = {
    sourceVariant: 0, targetVariant: 0, cost: Infinity, aligned: 1,
    sourceTokens: aSet[0]!, targetTokens: bSet[0]!,
  };
  for (let i = 0; i < aSet.length; i++) {
    for (let j = 0; j < bSet.length; j++) {
      const r = alignCost(aSet[i]!, bSet[j]!);
      const norm = r.aligned > 0 ? r.cost / r.aligned : r.cost;
      if (norm < best.cost) {
        best = { sourceVariant: i, targetVariant: j, cost: norm, aligned: r.aligned, sourceTokens: aSet[i]!, targetTokens: bSet[j]! };
      }
    }
  }
  // Calibration anchors:
  //   identical IPA      cost ≈ 0           → sim ≈ 1.00
  //   true homophone     cost ≈ 0.10–0.20   → sim ≈ 0.45–0.67
  //   moderate neighbor  cost ≈ 0.30–0.45   → sim ≈ 0.16–0.30
  //   unrelated          cost ≈ 0.55–0.85   → sim ≈ 0.03–0.11
  const sim = Math.max(0, Math.min(1, Math.exp(-best.cost * 4)));
  return {
    distance: best.cost,
    similarity: sim,
    bestAlignment: best,
    sourceCanonical: g1.raw.canonical,
    targetCanonical: g2.raw.canonical,
    sourceVariantCount: g1.variants.length,
    targetVariantCount: g2.variants.length,
  };
}
