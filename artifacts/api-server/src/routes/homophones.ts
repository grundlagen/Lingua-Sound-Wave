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
  TranslatePassageBody as TranslateRequest,
  TranslatePassageResponse as TranslateResponse,
} from "@workspace/api-zod";
import { LANGUAGES, languageName } from "../lib/languages";
import { FEATURED_PAIRS } from "../lib/featured";
import { synthesize, toAudioPayload, type SynthesizedAudio } from "../lib/tts";
import { mapWithLimit } from "../lib/concurrency";
import { getScoringMethod, listScoringMethods, type ComponentScore as ComponentScoreT } from "../lib/scoring";

const router: IRouter = Router();

router.get("/homophones/languages", (_req, res) => {
  res.json(LANGUAGES);
});

router.get("/homophones/methods", (_req, res) => {
  res.json(listScoringMethods());
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
  const method = getScoringMethod(body.scoringMethod);
  try {
    const [a1, a2] = await Promise.all([synthesize(body.phrase1), synthesize(body.phrase2)]);
    const lang1 = body.language1 ?? "";
    const lang2 = body.language2 ?? "";
    const r = await method.score(
      { ...a1, text: body.phrase1, language: lang1, languageName: lang1 ? languageName(lang1) : "" },
      { ...a2, text: body.phrase2, language: lang2, languageName: lang2 ? languageName(lang2) : "" },
    );
    const sim = r.similarity;
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
      dtwDistance: r.distance,
      verdict,
      scoringMethod: method.id,
      scoringMethodLabel: method.label,
      ...(r.components ? { componentScores: r.components } : {}),
    });
    res.json(payload);
  } catch (err) {
    req.log.error({ err, method: method.id }, "compare: failed");
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

// ---- Passage chunking + homophonic translation ----

/** Split a passage into sentence-ish chunks, balancing readable units. */
function chunkPassage(text: string): string[] {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned) return [];
  // Split on common terminators (Latin + CJK + Spanish), keeping the punctuation.
  const re = /[^.!?。！？؟…]+[.!?。！？؟…]+|[^.!?。！？؟…]+$/g;
  const raw = cleaned.match(re) ?? [cleaned];
  const out: string[] = [];
  for (let s of raw) {
    s = s.trim();
    if (!s) continue;
    // If a sentence is very long, split on commas/semicolons to keep chunks ≤ ~120 chars.
    if (s.length <= 140) {
      out.push(s);
    } else {
      const parts = s.split(/(?<=[,;:、，；])\s+/);
      let buf = "";
      for (const p of parts) {
        if ((buf + " " + p).trim().length > 140 && buf) {
          out.push(buf.trim());
          buf = p;
        } else {
          buf = buf ? buf + " " + p : p;
        }
      }
      if (buf.trim()) out.push(buf.trim());
    }
  }
  return out;
}

interface ChunkLLMResult {
  semantic: string;
  homophones: { phrase: string; gloss: string }[];
}

async function translateChunkLLM(
  chunk: string,
  sourceLanguage: string,
  targetLanguage: string,
  numCandidates: number,
): Promise<ChunkLLMResult> {
  const sourceName = languageName(sourceLanguage);
  const targetName = languageName(targetLanguage);
  const prompt = `You are a master of literary translation AND of homophonic translation (the art of writing real text in one language whose spoken sound mimics text in another, regardless of meaning — like the French nonsense-French of "Mots d'Heures: Gousses, Rames" mimicking English "Mother Goose Rhymes").

Source chunk (in ${sourceName}, code ${sourceLanguage}):
"""${chunk}"""

Target language: ${targetName} (code ${targetLanguage})

Produce two things:

1. "semantic": a faithful, natural meaning-preserving translation of the source chunk into ${targetName}.

2. "homophones": ${numCandidates} candidate HOMOPHONIC renderings — each is real, well-formed text in ${targetName} (using only ${targetName} words / orthography) whose NATURAL SPOKEN PRONUNCIATION by a native ${targetName} speaker sounds as much as possible like the source chunk when spoken aloud. The homophonic rendering's literal meaning will usually be unrelated nonsense — that is expected and desirable. It must NOT be a translation; it must SOUND like the source.

Rules for homophonic candidates:
- Use only real ${targetName} words. No invented words. Punctuation OK.
- Match the syllable count and stress pattern of the source as closely as possible.
- Multi-word combinations are encouraged.
- Each candidate should be a different attempt (different word choices), not a tiny variation.
- Provide a literal English "gloss" of the homophonic candidate's meaning.

Return strict JSON:
{
  "semantic": "...",
  "homophones": [
    {"phrase": "...", "gloss": "literal English meaning"},
    ...
  ]
}`;

  const completion = await openai.chat.completions.create({
    model: "gpt-5.4",
    messages: [
      { role: "system", content: "You are a meticulous translator and homophonic-translation expert. Return strict JSON only." },
      { role: "user", content: prompt },
    ],
    response_format: { type: "json_object" },
  });
  const txt = completion.choices[0]?.message?.content ?? "{}";
  const parsed = JSON.parse(txt) as { semantic?: unknown; homophones?: unknown };
  const semantic = typeof parsed.semantic === "string" ? parsed.semantic.trim() : "";
  const arr = Array.isArray(parsed.homophones) ? parsed.homophones : [];
  const homophones = arr
    .filter((h): h is { phrase: string; gloss?: string } =>
      typeof h === "object" && h !== null && typeof (h as { phrase?: unknown }).phrase === "string",
    )
    .map((h) => ({
      phrase: h.phrase.trim(),
      gloss: typeof h.gloss === "string" ? h.gloss.trim() : "",
    }))
    .filter((h) => h.phrase.length > 0)
    .slice(0, numCandidates);
  return { semantic, homophones };
}

router.post("/homophones/translate", async (req, res) => {
  const body = TranslateRequest.parse(req.body);
  const start = Date.now();
  const numCandidates = body.candidatesPerChunk ?? 2;
  const maxChunks = body.maxChunks ?? 30;
  const method = getScoringMethod(body.scoringMethod);
  const allChunks = chunkPassage(body.passage);
  const chunks = allChunks.slice(0, maxChunks);
  const dropped = allChunks.length - chunks.length;
  if (chunks.length === 0) {
    res.status(400).json({ error: "Passage is empty after normalization" });
    return;
  }
  if (dropped > 0) {
    req.log.warn({ dropped, total: allChunks.length, kept: chunks.length }, "translate: chunks dropped past maxChunks");
  }

  req.log.info(
    { chunks: chunks.length, candidatesPerChunk: numCandidates, method: method.id },
    "translate: starting passage translation",
  );

  const results = await mapWithLimit(chunks, 4, async (chunk, i) => {
    try {
      const llm = await translateChunkLLM(chunk, body.sourceLanguage, body.targetLanguage, numCandidates);
      let sourceAudio: SynthesizedAudio;
      let homophoneAudios: { spec: { phrase: string; gloss: string }; audio: SynthesizedAudio | null; error: string | null }[];
      try {
        sourceAudio = await synthesize(chunk);
      } catch (e) {
        return {
          index: i,
          sourceText: chunk,
          semanticTranslation: llm.semantic || "(translation failed)",
          homophonic: "",
          homophonicGloss: "",
          similarity: 0,
          dtwDistance: 0,
          alternatives: [],
          error: `source TTS failed: ${e instanceof Error ? e.message : String(e)}`,
        };
      }

      homophoneAudios = await Promise.all(
        llm.homophones.map(async (spec) => {
          try {
            const a = await synthesize(spec.phrase);
            return { spec, audio: a, error: null };
          } catch (e) {
            return { spec, audio: null, error: e instanceof Error ? e.message : String(e) };
          }
        }),
      );

      const scoredRaw = await Promise.all(
        homophoneAudios
          .filter((h): h is { spec: { phrase: string; gloss: string }; audio: SynthesizedAudio; error: null } => h.audio !== null)
          .map(async (h) => {
            const r = await method.score(
              { ...sourceAudio, text: chunk, language: body.sourceLanguage, languageName: languageName(body.sourceLanguage) },
              { ...h.audio, text: h.spec.phrase, language: body.targetLanguage, languageName: languageName(body.targetLanguage) },
            );
            return { spec: h.spec, audio: h.audio, d: r.distance, sim: r.similarity, components: r.components };
          }),
      );
      const scored = scoredRaw.sort((a, b) => b.sim - a.sim);

      if (scored.length === 0) {
        return {
          index: i,
          sourceText: chunk,
          semanticTranslation: llm.semantic || "",
          homophonic: "",
          homophonicGloss: "",
          similarity: 0,
          dtwDistance: 0,
          sourceAudio: toAudioPayload(sourceAudio),
          alternatives: [],
          error: "All homophonic candidate syntheses failed",
        };
      }

      const best = scored[0]!;
      const alternatives = scored.slice(1).map((s) => ({
        phrase: s.spec.phrase,
        gloss: s.spec.gloss,
        similarity: s.sim,
      }));

      return {
        index: i,
        sourceText: chunk,
        semanticTranslation: llm.semantic || "",
        homophonic: best.spec.phrase,
        homophonicGloss: best.spec.gloss,
        similarity: best.sim,
        dtwDistance: best.d,
        sourceAudio: toAudioPayload(sourceAudio),
        homophonicAudio: toAudioPayload(best.audio),
        alternatives,
        ...(best.components ? { componentScores: best.components } : {}),
      };
    } catch (e) {
      req.log.warn({ err: e, chunk }, "translate: chunk failed entirely");
      return {
        index: i,
        sourceText: chunk,
        semanticTranslation: "",
        homophonic: "",
        homophonicGloss: "",
        similarity: 0,
        dtwDistance: 0,
        alternatives: [],
        error: e instanceof Error ? e.message : String(e),
      };
    }
  });

  const failed = results.filter((r) => "error" in r && r.error).length;
  const sims = results.filter((r) => !r.error && r.similarity > 0).map((r) => r.similarity);
  const avg = sims.length > 0 ? sims.reduce((a, b) => a + b, 0) / sims.length : 0;

  const payload = TranslateResponse.parse({
    sourceLanguage: body.sourceLanguage,
    sourceLanguageName: languageName(body.sourceLanguage),
    targetLanguage: body.targetLanguage,
    targetLanguageName: languageName(body.targetLanguage),
    chunks: results,
    chunksAttempted: chunks.length,
    chunksFailed: failed,
    chunksDropped: dropped,
    averageSimilarity: avg,
    elapsedMs: Date.now() - start,
    scoringMethod: method.id,
    scoringMethodLabel: method.label,
  });
  res.json(payload);
});

router.post("/homophones/discover", async (req, res) => {
  const body = DiscoverRequest.parse(req.body);
  const start = Date.now();
  const method = getScoringMethod(body.scoringMethod);
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
  type Ok = { c: CandidateSpec; audio: Awaited<ReturnType<typeof synthesize>>; d: number; sim: number; components?: ComponentScoreT[] };
  type Err = { c: CandidateSpec; error: string };
  const synthResults = await mapWithLimit<CandidateSpec, Ok | Err>(candidates, 6, async (c) => {
    try {
      const audio = await synthesize(c.phrase);
      const r = await method.score(
        { ...sourceAudio, text: body.phrase, language: body.sourceLanguage, languageName: languageName(body.sourceLanguage) },
        { ...audio, text: c.phrase, language: c.language, languageName: languageName(c.language) },
      );
      return { c, audio, d: r.distance, sim: r.similarity, components: r.components };
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
      ...(r.components ? { componentScores: r.components } : {}),
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
    scoringMethod: method.id,
    scoringMethodLabel: method.label,
  });
  res.json(payload);
});

export default router;
