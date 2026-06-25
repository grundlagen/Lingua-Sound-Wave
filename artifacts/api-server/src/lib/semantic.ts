import { openai } from "@workspace/integrations-openai-ai-server";

/**
 * Semantic similarity of two short English glosses — the MEANING half of the
 * meaning map.
 *
 * The homophone reservoir already tells us two phrases SOUND alike (phonetic
 * similarity) and ships a one-line English gloss of each side's meaning. This
 * module judges how close those two glosses are in MEANING, independent of how
 * the original phrases sound. Pairing the two signals is what lets us surface
 * "resonant" pairs — cross-lingual phrases that are simultaneously homophones
 * AND (near-)translations of one another.
 *
 * Mirrors the design of `tier-grader.ts`: gpt-5.4 via the OpenAI integration,
 * strict JSON, an in-process cache (meaning closeness is symmetric, so the
 * cache key is order-independent). LLM/parse failures degrade to an honest
 * "unknown" verdict rather than throwing.
 */

export type SemanticRelation = "same" | "related" | "unrelated";

export interface SemanticVerdict {
  /** 0..1 meaning closeness of the two glosses. */
  similarity: number;
  relation: SemanticRelation;
  /** <= 20 word justification, for auditability in the UI. */
  note: string;
}

const cache = new Map<string, SemanticVerdict>();

function key(a: string, b: string): string {
  const x = a.toLowerCase().trim();
  const y = b.toLowerCase().trim();
  // Order-independent: S(a, b) === S(b, a).
  return x <= y ? `${x}|||${y}` : `${y}|||${x}`;
}

export function clearSemanticCache(): void {
  cache.clear();
}

export async function semanticSimilarity(
  glossA: string,
  glossB: string,
): Promise<SemanticVerdict> {
  const a = glossA.trim();
  const b = glossB.trim();
  if (!a || !b) return { similarity: 0, relation: "unrelated", note: "missing gloss" };

  const k = key(a, b);
  const hit = cache.get(k);
  if (hit) return hit;

  const prompt = `You are a careful bilingual semanticist. Rate how close in MEANING these two short English glosses are, ignoring entirely how the original phrases sound.

Gloss A: "${a}"
Gloss B: "${b}"

Scale (0.0–1.0):
  1.0 = same meaning (synonyms, or direct translations of one another)
  0.7 = strongly related (same scene/topic, heavily overlapping sense)
  0.4 = loosely related (share a domain or connotation only)
  0.1 = unrelated
  0.0 = contradictory or nonsensical pairing

Return strict JSON:
{ "similarity": <number 0..1>, "relation": "same"|"related"|"unrelated", "note": "<= 20 words" }`;

  let parsed: Partial<SemanticVerdict> = {};
  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-5.4",
      messages: [
        { role: "system", content: "You are a meticulous bilingual semanticist. Return strict JSON only." },
        { role: "user", content: prompt },
      ],
      response_format: { type: "json_object" },
    });
    parsed = JSON.parse(completion.choices[0]?.message?.content ?? "{}") as Partial<SemanticVerdict>;
  } catch {
    parsed = {};
  }

  const similarity = Math.max(0, Math.min(1, Number(parsed.similarity ?? 0)));
  const relation: SemanticRelation =
    parsed.relation === "same" || parsed.relation === "related" || parsed.relation === "unrelated"
      ? parsed.relation
      : similarity >= 0.85
        ? "same"
        : similarity >= 0.4
          ? "related"
          : "unrelated";
  const note = typeof parsed.note === "string" ? parsed.note.trim() : "";

  const result: SemanticVerdict = { similarity, relation, note };
  cache.set(k, result);
  return result;
}
