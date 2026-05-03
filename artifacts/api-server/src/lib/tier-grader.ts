import { openai } from "@workspace/integrations-openai-ai-server";

export type Tier = "S" | "A" | "B";

export interface TierGrade {
  tier: Tier;
  enCoherence: number; // 0..3
  frCoherence: number; // 0..3
  enGloss: string;
  frGloss: string;
  rationale: string;
}

const cache = new Map<string, TierGrade>();

function key(en: string, fr: string): string {
  return `${en.toLowerCase().trim()}|||${fr.toLowerCase().trim()}`;
}

/**
 * Grade an EN↔FR sound-alike pair on (a) per-side semantic coherence (0..3)
 * and (b) overall tier (S/A/B) using the rubric:
 *   - S: both sides ≥2/3 coherence AND similarity ≥ 0.85
 *   - A: both sides ≥1/3 AND similarity ≥ 0.70, OR one side =3 + similarity ≥ 0.85
 *   - B: anything else above the floor
 *
 * Returns the LLM-supplied glosses too so callers don't have to make a second call.
 */
export async function gradePair(
  enPhrase: string,
  frPhrase: string,
  similarity: number,
): Promise<TierGrade> {
  const k = key(enPhrase, frPhrase);
  const hit = cache.get(k);
  if (hit) return hit;

  const prompt = `You are a strict bilingual editor judging a homophonic pair (English text and French text that are supposed to sound alike when spoken).

You will rate each side on COHERENCE only (does it parse as real, grammatical text with a recognizable meaning?), independent of whether the two sides mean the same thing.

Coherence scale (per side):
  3 = fully grammatical, makes immediate sense as a real sentence/phrase/word
  2 = grammatical but odd or surreal (Mots-d'Heures style is fine here)
  1 = recognizable words but ungrammatical or nonsense word salad
  0 = not real text in that language

Pair to judge:
  EN: "${enPhrase}"
  FR: "${frPhrase}"

Return strict JSON:
{
  "en_coherence": 0|1|2|3,
  "fr_coherence": 0|1|2|3,
  "en_gloss": "1-line literal English meaning of the EN side",
  "fr_gloss": "1-line literal English meaning of the FR side",
  "rationale": "<= 20 words"
}`;

  let parsed: { en_coherence?: number; fr_coherence?: number; en_gloss?: string; fr_gloss?: string; rationale?: string } = {};
  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-5.4",
      messages: [
        { role: "system", content: "You are a meticulous bilingual editor. Return strict JSON only." },
        { role: "user", content: prompt },
      ],
      response_format: { type: "json_object" },
    });
    parsed = JSON.parse(completion.choices[0]?.message?.content ?? "{}");
  } catch {
    parsed = {};
  }

  const enC = Math.max(0, Math.min(3, Math.round(Number(parsed.en_coherence ?? 0))));
  const frC = Math.max(0, Math.min(3, Math.round(Number(parsed.fr_coherence ?? 0))));
  const enGloss = typeof parsed.en_gloss === "string" ? parsed.en_gloss.trim() : "";
  const frGloss = typeof parsed.fr_gloss === "string" ? parsed.fr_gloss.trim() : "";
  const rationale = typeof parsed.rationale === "string" ? parsed.rationale.trim() : "";

  let tier: Tier = "B";
  if (enC >= 2 && frC >= 2 && similarity >= 0.85) tier = "S";
  else if ((enC >= 1 && frC >= 1 && similarity >= 0.7) || ((enC === 3 || frC === 3) && similarity >= 0.85)) tier = "A";

  const result: TierGrade = { tier, enCoherence: enC, frCoherence: frC, enGloss, frGloss, rationale };
  cache.set(k, result);
  return result;
}

export function clearTierCache(): void {
  cache.clear();
}
