import { Router, type IRouter } from "express";
import { openai } from "@workspace/integrations-openai-ai-server";
import { db, savedPairsTable } from "@workspace/db";
import { eq, desc } from "drizzle-orm";
import {
  DiscoverHomophonesBody as DiscoverRequest,
  DiscoverHomophonesResponse as DiscoverResponse,
  SynthesizeSpeechBody as TtsRequest,
  SynthesizeSpeechResponse as TtsResponse,
  ComparePhrasesBody as CompareRequest,
  ComparePhrasesResponse as CompareResponse,
  SavePairBody as SavePairRequest,
} from "@workspace/api-zod";
import { LANGUAGES, languageName } from "../lib/languages";
import { FEATURED_PAIRS } from "../lib/featured";
import { synthesize, toAudioPayload } from "../lib/tts";
import { dtwDistance, distanceToSimilarity } from "../lib/dsp";

const router: IRouter = Router();

router.get("/homophones/languages", (_req, res) => {
  res.json(LANGUAGES);
});

router.get("/homophones/featured", (_req, res) => {
  res.json(FEATURED_PAIRS);
});

router.get("/homophones/saved", async (_req, res) => {
  const rows = await db.select().from(savedPairsTable).orderBy(desc(savedPairsTable.createdAt));
  res.json(
    rows.map((r) => ({
      id: r.id,
      sourcePhrase: r.sourcePhrase,
      sourceLanguage: r.sourceLanguage,
      sourceMeaning: r.sourceMeaning,
      matchPhrase: r.matchPhrase,
      matchLanguage: r.matchLanguage,
      matchMeaning: r.matchMeaning,
      similarity: r.similarity,
      notes: r.notes ?? "",
      createdAt: r.createdAt.toISOString(),
    })),
  );
});

router.post("/homophones/saved", async (req, res) => {
  const body = SavePairRequest.parse(req.body);
  const [row] = await db
    .insert(savedPairsTable)
    .values({
      sourcePhrase: body.sourcePhrase,
      sourceLanguage: body.sourceLanguage,
      sourceMeaning: body.sourceMeaning,
      matchPhrase: body.matchPhrase,
      matchLanguage: body.matchLanguage,
      matchMeaning: body.matchMeaning,
      similarity: body.similarity,
      notes: body.notes ?? null,
    })
    .returning();
  res.status(201).json({
    id: row!.id,
    sourcePhrase: row!.sourcePhrase,
    sourceLanguage: row!.sourceLanguage,
    sourceMeaning: row!.sourceMeaning,
    matchPhrase: row!.matchPhrase,
    matchLanguage: row!.matchLanguage,
    matchMeaning: row!.matchMeaning,
    similarity: row!.similarity,
    notes: row!.notes ?? "",
    createdAt: row!.createdAt.toISOString(),
  });
});

router.delete("/homophones/saved/:id", async (req, res) => {
  const id = Number(req.params["id"]);
  if (!Number.isInteger(id) || id <= 0) {
    res.status(400).json({ error: "Invalid id" });
    return;
  }
  const deleted = await db
    .delete(savedPairsTable)
    .where(eq(savedPairsTable.id, id))
    .returning({ id: savedPairsTable.id });
  if (deleted.length === 0) {
    res.status(404).json({ error: "Saved pair not found", id });
    return;
  }
  res.json({ success: true, id });
});

router.post("/homophones/tts", async (req, res) => {
  const body = TtsRequest.parse(req.body);
  try {
    const audio = await synthesize(body.text);
    const payload = TtsResponse.parse({ audio: toAudioPayload(audio) });
    res.json(payload);
  } catch (err) {
    req.log.error({ err, text: body.text }, "tts: synthesis failed");
    res.status(502).json({ error: "Speech synthesis failed", detail: String(err instanceof Error ? err.message : err) });
  }
});

router.post("/homophones/compare", async (req, res) => {
  const body = CompareRequest.parse(req.body);
  try {
    const [a1, a2] = await Promise.all([synthesize(body.phrase1), synthesize(body.phrase2)]);
    const d = dtwDistance(a1.features.mfcc, a2.features.mfcc);
    const sim = distanceToSimilarity(d);
    const verdict =
      sim > 0.85
        ? "Near-identical acoustic match"
        : sim > 0.7
          ? "Strong acoustic similarity"
          : sim > 0.55
            ? "Moderate similarity"
            : "Weak acoustic match";
    const payload = CompareResponse.parse({
      phrase1: body.phrase1,
      phrase2: body.phrase2,
      audio1: toAudioPayload(a1),
      audio2: toAudioPayload(a2),
      similarity: sim,
      dtwDistance: d,
      verdict,
    });
    res.json(payload);
  } catch (err) {
    req.log.error({ err }, "compare: failed");
    res.status(502).json({ error: "Comparison failed", detail: String(err instanceof Error ? err.message : err) });
  }
});

interface CandidateSpec {
  phrase: string;
  language: string;
  meaning: string;
  notes: string;
}

async function generateCandidates(
  phrase: string,
  sourceLanguage: string,
  targetLanguages: string[],
  count: number,
): Promise<CandidateSpec[]> {
  const targetList = targetLanguages
    .map((c) => `${c} (${languageName(c)})`)
    .join(", ");
  const sourceName = languageName(sourceLanguage);
  const prompt = `You are a multilingual phonetics researcher. Find phrases (single words OR multi-word) from the listed target languages whose NATURAL SPOKEN PRONUNCIATION sounds as close as possible to the source phrase when both are spoken at normal speed by a native speaker.

Source phrase: "${phrase}"
Source language: ${sourceName} (${sourceLanguage})

Target languages to search: ${targetList}

CRITICAL RULES:
1. The candidate phrase must be REAL, meaningful text in its language (a real word, name, or sentence). No nonsense.
2. It MUST sound like the source phrase when spoken — not just look similar.
3. Multi-word phrases are STRONGLY ENCOURAGED — combining short words to mimic the source sound.
4. EXCLUDE direct cognates, loanwords, or words borrowed from / shared with the source language.
5. The candidate's meaning should be UNRELATED to the source phrase's meaning (true homophones, not translations).
6. Provide ${count} candidates spread across many languages.
7. Each candidate should genuinely sound like the source — be strict.

Return a JSON object: {"candidates": [{"phrase": "...", "language": "<ISO code from list>", "meaning": "literal English translation", "notes": "brief note on why it sounds like the source"}]}`;

  const completion = await openai.chat.completions.create({
    model: "gpt-5.4",
    messages: [
      { role: "system", content: "You are a meticulous multilingual phonetician. Return strict JSON only." },
      { role: "user", content: prompt },
    ],
    response_format: { type: "json_object" },
  });
  const txt = completion.choices[0]?.message?.content ?? "{}";
  try {
    const parsed = JSON.parse(txt);
    const arr = Array.isArray(parsed.candidates) ? parsed.candidates : [];
    return arr
      .filter((c: { phrase?: unknown; language?: unknown; meaning?: unknown }) => typeof c.phrase === "string" && typeof c.language === "string" && typeof c.meaning === "string")
      .map((c: { phrase: string; language: string; meaning: string; notes?: string }) => ({
        phrase: c.phrase,
        language: c.language,
        meaning: c.meaning,
        notes: c.notes ?? "",
      }));
  } catch {
    return [];
  }
}

async function describeMeaning(phrase: string, language: string): Promise<string> {
  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-5.4",
      messages: [
        {
          role: "system",
          content: "Translate or briefly explain the meaning of the user's phrase in plain English. Reply with the meaning only, under 12 words.",
        },
        { role: "user", content: `Phrase in ${languageName(language)}: "${phrase}"` },
      ],
    });
    return (completion.choices[0]?.message?.content ?? "").trim() || phrase;
  } catch {
    return phrase;
  }
}

async function mapWithLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let idx = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const i = idx++;
      if (i >= items.length) return;
      try {
        results[i] = await fn(items[i]!);
      } catch (err) {
        results[i] = err as R;
      }
    }
  });
  await Promise.all(workers);
  return results;
}

router.post("/homophones/discover", async (req, res) => {
  const body = DiscoverRequest.parse(req.body);
  const start = Date.now();
  const targets =
    body.targetLanguages && body.targetLanguages.length > 0
      ? body.targetLanguages
      : LANGUAGES.filter((l) => l.code !== body.sourceLanguage).map((l) => l.code);
  const wantCount = body.candidateCount ?? 32;
  const minSim = body.minSimilarity ?? 0.55;

  req.log.info({ phrase: body.phrase, targets: targets.length, wantCount }, "discover: generating candidates");

  // Step 1: candidates + source meaning + source TTS in parallel
  let candidatesRaw: CandidateSpec[];
  let sourceMeaning: string;
  let sourceAudio: Awaited<ReturnType<typeof synthesize>>;
  try {
    [candidatesRaw, sourceMeaning, sourceAudio] = await Promise.all([
      generateCandidates(body.phrase, body.sourceLanguage, targets, wantCount),
      describeMeaning(body.phrase, body.sourceLanguage),
      synthesize(body.phrase),
    ]);
  } catch (err) {
    req.log.error({ err }, "discover: candidate/source phase failed");
    res.status(502).json({
      error: "Failed to generate candidates or synthesize source phrase",
      detail: String(err instanceof Error ? err.message : err),
    });
    return;
  }

  // Validate candidate languages against the supported set
  const allowed = new Set(LANGUAGES.map((l) => l.code));
  const candidates = candidatesRaw.filter((c) => allowed.has(c.language));
  const droppedUnknown = candidatesRaw.length - candidates.length;
  if (droppedUnknown > 0) {
    req.log.warn({ droppedUnknown }, "discover: dropped candidates with unknown language codes");
  }

  req.log.info({ count: candidates.length }, "discover: candidates ready, synthesizing TTS");

  // Step 2: synthesize each candidate (concurrency limited)
  type Ok = { c: CandidateSpec; audio: Awaited<ReturnType<typeof synthesize>>; d: number; sim: number };
  type Err = { c: CandidateSpec; error: string };
  const synthResults = await mapWithLimit<CandidateSpec, Ok | Err>(candidates, 6, async (c) => {
    try {
      const audio = await synthesize(c.phrase);
      const d = dtwDistance(sourceAudio.features.mfcc, audio.features.mfcc);
      const sim = distanceToSimilarity(d);
      return { c, audio, d, sim };
    } catch (e) {
      return { c, error: e instanceof Error ? e.message : String(e) };
    }
  });

  const failures = synthResults.filter((r): r is Err => "error" in r);
  if (failures.length > 0) {
    req.log.warn({ failed: failures.length, sample: failures.slice(0, 3) }, "discover: some candidate syntheses failed");
  }

  const oks = synthResults.filter((r): r is Ok => "sim" in r);
  if (oks.length === 0 && candidates.length > 0) {
    res.status(502).json({
      error: "All candidate syntheses failed",
      detail: failures[0]?.error ?? "unknown error",
      candidatesAttempted: candidates.length,
    });
    return;
  }

  const matches = oks
    .filter((r) => r.sim >= minSim)
    .sort((a, b) => b.sim - a.sim)
    .map((r) => ({
      phrase: r.c.phrase,
      language: languageName(r.c.language),
      languageCode: r.c.language,
      meaning: r.c.meaning,
      notes: r.c.notes,
      similarity: r.sim,
      dtwDistance: r.d,
      audio: toAudioPayload(r.audio),
    }));

  const payload = DiscoverResponse.parse({
    sourcePhrase: body.phrase,
    sourceLanguage: body.sourceLanguage,
    sourceLanguageName: languageName(body.sourceLanguage),
    sourceMeaning,
    sourceAudio: toAudioPayload(sourceAudio),
    matches,
    candidatesEvaluated: candidates.length,
    candidatesFailed: failures.length,
    elapsedMs: Date.now() - start,
  });
  res.json(payload);
});

export default router;
