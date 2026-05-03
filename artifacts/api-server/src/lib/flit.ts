import { openai } from "@workspace/integrations-openai-ai-server";
import { synthesize } from "./tts";
import { getScoringMethod, type ComponentScore } from "./scoring";
import { mapWithLimit } from "./concurrency";
import { logger } from "./logger";

const HYBRID_METHOD_ID = "hybrid-phoneme-audio";
const SCORE_CONCURRENCY = 4;

export interface FlitInput {
  text: string;
  /** "en" or "fr" */
  language: "en" | "fr";
  /** How many paraphrases of the input to generate (N). */
  inputParaphrases?: number;
  /** How many target-side renderings per paraphrase (M). Total cross-product = N * M. */
  targetRenderings?: number;
  /** Top-K finals returned to the client (semantic-verified). */
  topK?: number;
}

export interface FlitCandidate {
  inputParaphrase: string;
  inputParaphraseGloss: string;
  targetText: string;
  targetGloss: string;
  similarity: number;
  componentScores: ComponentScore[];
  semanticOK: boolean;
  semanticNote: string;
}

export interface FlitResult {
  inputLanguage: "en" | "fr";
  targetLanguage: "en" | "fr";
  inputText: string;
  inputMeaning: string;
  inputParaphrases: { text: string; gloss: string }[];
  candidates: FlitCandidate[];
  best: FlitCandidate | null;
  attempted: number;
  elapsedMs: number;
}

const langName = (l: "en" | "fr") => (l === "en" ? "English" : "French");

async function paraphraseInput(text: string, lang: "en" | "fr", count: number): Promise<{ text: string; gloss: string }[]> {
  const name = langName(lang);
  const prompt = `Generate ${count} paraphrases of the following ${name} text. Each paraphrase must:
  • Preserve the meaning faithfully (same propositional content).
  • Vary the wording: use synonyms, restructure clauses, change word order.
  • Be natural, well-formed ${name}.
  • Have a noticeably different surface form from the original and from the other paraphrases.

Source ${name} text:
"""${text}"""

Return strict JSON: { "paraphrases": [ { "text": "...", "gloss": "literal English meaning, <= 12 words" }, ... ] }`;

  const completion = await openai.chat.completions.create({
    model: "gpt-5.4",
    messages: [
      { role: "system", content: "You are a precise multilingual writer. Return strict JSON only." },
      { role: "user", content: prompt },
    ],
    response_format: { type: "json_object" },
  });
  const txt = completion.choices[0]?.message?.content ?? "{}";
  try {
    const parsed = JSON.parse(txt) as { paraphrases?: unknown };
    if (!Array.isArray(parsed.paraphrases)) return [];
    return parsed.paraphrases
      .filter(
        (p): p is { text: string; gloss?: string } =>
          typeof p === "object" && p !== null && typeof (p as { text?: unknown }).text === "string",
      )
      .map((p) => ({
        text: (p.text as string).trim(),
        gloss: typeof p.gloss === "string" ? p.gloss.trim() : "",
      }))
      .filter((p) => p.text.length > 0)
      .slice(0, count);
  } catch {
    return [];
  }
}

async function generateTargetRenderings(
  inputParaphrase: string,
  inputLang: "en" | "fr",
  targetLang: "en" | "fr",
  count: number,
): Promise<{ text: string; gloss: string }[]> {
  const inName = langName(inputLang);
  const tName = langName(targetLang);
  const prompt = `You are a master of cross-lingual homophonic translation between English and French.

Input ${inName} text:
"""${inputParaphrase}"""

Goal: produce ${count} ${tName} candidate phrases that sound as much as possible like the input ${inName} text when spoken aloud (à la Mots d'Heures: Gousses, Rames). Each candidate should:
  • Be real, well-formed ${tName} (only ${tName} words; punctuation OK).
  • Match the input's syllable count and stress closely.
  • Aim for plausibly meaningful ${tName} text — surreal is fine; total nonsense is undesirable.
  • Differ meaningfully from the other candidates.

Return strict JSON: { "renderings": [ { "text": "...", "gloss": "literal English meaning, <= 12 words" }, ... ] }`;

  const completion = await openai.chat.completions.create({
    model: "gpt-5.4",
    messages: [
      { role: "system", content: "You are a meticulous bilingual phonetician. Return strict JSON only." },
      { role: "user", content: prompt },
    ],
    response_format: { type: "json_object" },
  });
  const txt = completion.choices[0]?.message?.content ?? "{}";
  try {
    const parsed = JSON.parse(txt) as { renderings?: unknown };
    if (!Array.isArray(parsed.renderings)) return [];
    return parsed.renderings
      .filter(
        (r): r is { text: string; gloss?: string } =>
          typeof r === "object" && r !== null && typeof (r as { text?: unknown }).text === "string",
      )
      .map((r) => ({
        text: (r.text as string).trim(),
        gloss: typeof r.gloss === "string" ? r.gloss.trim() : "",
      }))
      .filter((r) => r.text.length > 0)
      .slice(0, count);
  } catch {
    return [];
  }
}

async function describeMeaning(text: string, lang: "en" | "fr"): Promise<string> {
  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-5.4",
      messages: [
        {
          role: "system",
          content: "Translate or briefly explain the meaning of the user's phrase in plain English. Reply with the meaning only, under 12 words.",
        },
        { role: "user", content: `Phrase in ${langName(lang)}: "${text}"` },
      ],
    });
    return (completion.choices[0]?.message?.content ?? "").trim() || text;
  } catch {
    return text;
  }
}

interface SemanticVerdict {
  ok: boolean;
  note: string;
}

async function verifySemantic(
  originalInput: string,
  inputLang: "en" | "fr",
  candidate: { inputParaphrase: string; targetText: string },
): Promise<SemanticVerdict> {
  // We verify that the chosen target rendering's meaning still tracks the
  // original input meaning (not too far off after paraphrase + sound-match).
  const prompt = `Original ${langName(inputLang)} input:
"""${originalInput}"""

A candidate cross-lingual sound-alike was produced by:
  1. Paraphrasing the input → "${candidate.inputParaphrase}"
  2. Generating a sound-alike target rendering → "${candidate.targetText}"

Question: does the target rendering, taken at its literal meaning, still convey roughly the same idea / intent as the original input? It need not match perfectly; surreal flavor is acceptable when the gist is recognizable.

Return strict JSON: { "ok": true|false, "note": "<= 20 words explaining" }`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-5.4",
      messages: [
        { role: "system", content: "You are a strict bilingual semantic judge. Return strict JSON only." },
        { role: "user", content: prompt },
      ],
      response_format: { type: "json_object" },
    });
    const parsed = JSON.parse(completion.choices[0]?.message?.content ?? "{}") as { ok?: unknown; note?: unknown };
    return {
      ok: parsed.ok === true,
      note: typeof parsed.note === "string" ? parsed.note.trim() : "",
    };
  } catch {
    return { ok: false, note: "verification failed" };
  }
}

export async function runFlit(input: FlitInput): Promise<FlitResult> {
  const start = Date.now();
  const inputLang = input.language;
  const targetLang: "en" | "fr" = inputLang === "en" ? "fr" : "en";
  const N = input.inputParaphrases ?? 6;
  const M = input.targetRenderings ?? 4;
  const topK = input.topK ?? 5;
  const method = getScoringMethod(HYBRID_METHOD_ID);

  // 1. Paraphrase the input + describe its meaning, in parallel.
  const [paraphrases, inputMeaning] = await Promise.all([
    paraphraseInput(input.text, inputLang, N),
    describeMeaning(input.text, inputLang),
  ]);

  // Always include the original as the first paraphrase so we don't lose ground.
  const allInputs: { text: string; gloss: string }[] = [
    { text: input.text, gloss: inputMeaning },
    ...paraphrases.filter((p) => p.text.toLowerCase().trim() !== input.text.toLowerCase().trim()),
  ].slice(0, Math.max(1, N));

  logger.info({ inputLang, targetLang, N: allInputs.length, M, topK }, "flit: starting cross-product");

  // 2. For each input paraphrase, generate M target-language renderings (parallel).
  const renderingsPerInput = await Promise.all(
    allInputs.map((p) => generateTargetRenderings(p.text, inputLang, targetLang, M).catch(() => [])),
  );

  // 3. Build the cross-product: every (paraphrase, rendering) pair.
  interface Pair {
    inputParaphrase: { text: string; gloss: string };
    rendering: { text: string; gloss: string };
  }
  const pairs: Pair[] = [];
  for (let i = 0; i < allInputs.length; i++) {
    const para = allInputs[i]!;
    for (const rend of renderingsPerInput[i]!) {
      pairs.push({ inputParaphrase: para, rendering: rend });
    }
  }

  // 4. Score every pair acoustically (parallel, capped concurrency).
  const scored = await mapWithLimit(pairs, SCORE_CONCURRENCY, async (pair) => {
    try {
      const enText = inputLang === "en" ? pair.inputParaphrase.text : pair.rendering.text;
      const frText = inputLang === "en" ? pair.rendering.text : pair.inputParaphrase.text;
      const [aIn, aOut] = await Promise.all([synthesize(pair.inputParaphrase.text), synthesize(pair.rendering.text)]);
      const r = await method.score(
        { ...aIn, text: pair.inputParaphrase.text, language: inputLang, languageName: langName(inputLang) },
        { ...aOut, text: pair.rendering.text, language: targetLang, languageName: langName(targetLang) },
      );
      return {
        pair,
        similarity: r.similarity,
        components: r.components ?? [],
        en: enText,
        fr: frText,
      };
    } catch (err) {
      logger.warn({ err }, "flit: pair scoring failed");
      return null;
    }
  });
  const valid = scored.filter((s): s is NonNullable<typeof s> => s !== null && s.similarity > 0);
  valid.sort((a, b) => b.similarity - a.similarity);

  // 5. Take top-K, semantic-verify each.
  const finalists = valid.slice(0, topK);
  const verified: FlitCandidate[] = await Promise.all(
    finalists.map(async (f) => {
      const verdict = await verifySemantic(input.text, inputLang, {
        inputParaphrase: f.pair.inputParaphrase.text,
        targetText: f.pair.rendering.text,
      });
      return {
        inputParaphrase: f.pair.inputParaphrase.text,
        inputParaphraseGloss: f.pair.inputParaphrase.gloss,
        targetText: f.pair.rendering.text,
        targetGloss: f.pair.rendering.gloss,
        similarity: f.similarity,
        componentScores: f.components,
        semanticOK: verdict.ok,
        semanticNote: verdict.note,
      };
    }),
  );

  const best = verified.find((c) => c.semanticOK) ?? verified[0] ?? null;

  return {
    inputLanguage: inputLang,
    targetLanguage: targetLang,
    inputText: input.text,
    inputMeaning,
    inputParaphrases: allInputs,
    candidates: verified,
    best,
    attempted: pairs.length,
    elapsedMs: Date.now() - start,
  };
}
