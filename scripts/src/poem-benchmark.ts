/**
 * Long-poem cross-lingual scoring benchmark.
 *
 * For each pair of (source phrase, target phrase) we:
 *   1. Hit our /api/homophones/compare endpoint with the default
 *      hybrid-phoneme-audio judge to get our similarity (0-1).
 *   2. Ask an LLM oracle (gpt-5.4) to *independently* rate phonetic
 *      similarity 0-100 of the same pair, with NO knowledge of our score.
 *   3. Compare. We expect:
 *        - Semantic translations (same meaning, different sounds) -> LOW for both
 *        - Cross-lingual homophone hits (same sound, different meaning) -> HIGH for both
 *      and a positive Pearson correlation between our score and the oracle.
 *
 * The "long poem" angle: we pull lines and full stanzas from famous poems
 * that exist in multiple languages so we can verify the scorer works on
 * realistic long inputs, not just toy phrases.
 */

import { openai } from "@workspace/integrations-openai-ai-server";
import { writeFileSync } from "node:fs";

const API = process.env.HOMOPHONE_API ?? "http://localhost:80/api";

interface Pair {
  group: string;
  label: string;
  /** "translation" = same meaning across langs (should score LOW phonetically) */
  /** "homophone"   = known cross-lingual sound match (should score HIGH) */
  /** "control"     = totally unrelated random pair (sanity floor) */
  kind: "translation" | "homophone" | "control";
  a: { text: string; lang: string };
  b: { text: string; lang: string };
}

// ---------- Stimuli: long poems with parallel translations ----------
//
// Each "translation" pair: same poem, different languages -> meaning preserved,
// sound NOT preserved. Our scorer should give a low number; the LLM oracle
// should agree.

const PAIRS: Pair[] = [
  // ============ Li Bai: Quiet Night Thoughts (静夜思) ============
  {
    group: "Li Bai · Quiet Night Thoughts",
    label: "zh ↔ en (Witter Bynner)",
    kind: "translation",
    a: { text: "床前明月光，疑是地上霜。举头望明月，低头思故乡。", lang: "zh" },
    b: {
      text:
        "So bright a gleam on the foot of my bed. Could there have been a frost already? Lifting myself to look, I found that it was moonlight. Sinking back again, I thought suddenly of home.",
      lang: "en",
    },
  },
  {
    group: "Li Bai · Quiet Night Thoughts",
    label: "zh ↔ ja (yomikudashi)",
    kind: "translation",
    a: { text: "床前明月光，疑是地上霜。举头望明月，低头思故乡。", lang: "zh" },
    b: {
      text: "牀前月光を看る、疑うらくは是れ地上の霜かと。頭を挙げて山月を望み、頭を低れて故郷を思う。",
      lang: "ja",
    },
  },
  {
    group: "Li Bai · Quiet Night Thoughts",
    label: "en ↔ ja (both translations)",
    kind: "translation",
    a: {
      text: "Lifting myself to look, I found that it was moonlight. Sinking back again, I thought suddenly of home.",
      lang: "en",
    },
    b: { text: "頭を挙げて山月を望み、頭を低れて故郷を思う。", lang: "ja" },
  },

  // ============ Basho: Old Pond Haiku (古池や) ============
  {
    group: "Basho · Old Pond",
    label: "ja ↔ en",
    kind: "translation",
    a: { text: "古池や蛙飛び込む水の音", lang: "ja" },
    b: { text: "An old silent pond. A frog jumps into the pond — splash! Silence again.", lang: "en" },
  },
  {
    group: "Basho · Old Pond",
    label: "ja ↔ es",
    kind: "translation",
    a: { text: "古池や蛙飛び込む水の音", lang: "ja" },
    b: { text: "Un viejo estanque; se zambulle una rana, ruido del agua.", lang: "es" },
  },

  // ============ Pablo Neruda: Sonnet XVII (opening) ============
  {
    group: "Neruda · Sonnet XVII",
    label: "es ↔ en (Stephen Tapscott)",
    kind: "translation",
    a: {
      text: "No te amo como si fueras rosa de sal, topacio o flecha de claveles que propagan el fuego.",
      lang: "es",
    },
    b: {
      text: "I do not love you as if you were salt-rose, or topaz, or the arrow of carnations the fire shoots off.",
      lang: "en",
    },
  },
  {
    group: "Neruda · Sonnet XVII",
    label: "es ↔ fr",
    kind: "translation",
    a: {
      text: "No te amo como si fueras rosa de sal, topacio o flecha de claveles que propagan el fuego.",
      lang: "es",
    },
    b: {
      text:
        "Je ne t'aime pas comme si tu étais rose de sel, topaze ou flèche d'œillets qui propagent le feu.",
      lang: "fr",
    },
  },

  // ============ Verlaine: Chanson d'automne (opening stanza) ============
  {
    group: "Verlaine · Chanson d'automne",
    label: "fr ↔ en",
    kind: "translation",
    a: {
      text: "Les sanglots longs des violons de l'automne blessent mon cœur d'une langueur monotone.",
      lang: "fr",
    },
    b: {
      text: "The long sobs of the violins of autumn wound my heart with a monotonous languor.",
      lang: "en",
    },
  },
  {
    group: "Verlaine · Chanson d'automne",
    label: "fr ↔ de",
    kind: "translation",
    a: {
      text: "Les sanglots longs des violons de l'automne blessent mon cœur d'une langueur monotone.",
      lang: "fr",
    },
    b: {
      text: "Die langen Schluchzer der Violinen des Herbstes verwunden mein Herz mit eintöniger Mattigkeit.",
      lang: "de",
    },
  },

  // ============ Goethe: Wanderers Nachtlied II ============
  {
    group: "Goethe · Wanderers Nachtlied II",
    label: "de ↔ en (Longfellow)",
    kind: "translation",
    a: {
      text:
        "Über allen Gipfeln ist Ruh, in allen Wipfeln spürest du kaum einen Hauch; die Vögelein schweigen im Walde.",
      lang: "de",
    },
    b: {
      text:
        "O'er all the hilltops is quiet now, in all the treetops hearest thou hardly a breath; the birds are asleep in the trees.",
      lang: "en",
    },
  },

  // ============ Positive controls (known cross-lingual homophones) ============
  // These should score HIGH on both our judge and the oracle.
  {
    group: "Known homophones (positive control)",
    label: "en 'knee how' ↔ zh 你好",
    kind: "homophone",
    a: { text: "knee how", lang: "en" },
    b: { text: "你好", lang: "zh" },
  },
  {
    group: "Known homophones (positive control)",
    label: "en 'show me' ↔ ko 쇼미",
    kind: "homophone",
    a: { text: "show me", lang: "en" },
    b: { text: "쇼미", lang: "ko" },
  },
  {
    group: "Known homophones (positive control)",
    label: "en 'I love you' ↔ ko 아이 럽 유",
    kind: "homophone",
    a: { text: "I love you", lang: "en" },
    b: { text: "아이 럽 유", lang: "ko" },
  },

  // ============ Negative controls (truly unrelated random pairs) ============
  {
    group: "Unrelated controls (negative)",
    label: "en haiku ↔ es Neruda",
    kind: "control",
    a: { text: "An old silent pond. A frog jumps into the pond — splash!", lang: "en" },
    b: { text: "No te amo como si fueras rosa de sal o topacio.", lang: "es" },
  },
  {
    group: "Unrelated controls (negative)",
    label: "de Goethe ↔ ja haiku",
    kind: "control",
    a: { text: "Über allen Gipfeln ist Ruh.", lang: "de" },
    b: { text: "古池や蛙飛び込む水の音", lang: "ja" },
  },
];

// ---------- Our scorer ----------

interface CompareResult {
  similarity: number;
  componentScores?: { id: string; similarity: number }[];
  error?: string;
}

async function ourScore(pair: Pair): Promise<CompareResult> {
  const res = await fetch(`${API}/homophones/compare`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      phrase1: pair.a.text,
      language1: pair.a.lang,
      phrase2: pair.b.text,
      language2: pair.b.lang,
      scoringMethod: "hybrid-phoneme-audio",
    }),
  });
  if (!res.ok) {
    return { similarity: NaN, error: `HTTP ${res.status}: ${await res.text()}` };
  }
  const j = (await res.json()) as CompareResult;
  return j;
}

// ---------- Independent oracle ----------
//
// Ask an LLM to rate phonetic similarity 0-100 between two phrases. The model
// is told nothing about our scorer — it gets only the texts and language tags.
// We use strict JSON output with a brief reasoning trace for auditability.

interface OracleVerdict {
  similarity_0_100: number;
  reasoning: string;
}

async function oracleScore(pair: Pair): Promise<OracleVerdict> {
  const prompt = `You are a phonetician. Rate ONLY the phonetic / sound-similarity of these two phrases when read aloud, ignoring meaning entirely.

Phrase A (${pair.a.lang}): ${pair.a.text}
Phrase B (${pair.b.lang}): ${pair.b.text}

Return strict JSON:
{
  "similarity_0_100": <integer 0-100, where 0 = completely different sounds, 100 = essentially homophonous>,
  "reasoning": "<one sentence comparing the broad IPA shapes>"
}

Calibration:
- Two random different-language sentences with no shared sounds: 5-15
- Two phrases that share a few syllables by chance: 20-40
- Phrases that loosely echo each other's prosody but differ in segments: 40-60
- Recognizable cross-lingual homophone (e.g. English "knee how" vs Mandarin 你好): 80-95
- Identical pronunciation: 95-100`;

  const completion = await openai.chat.completions.create({
    model: "gpt-5.4",
    messages: [
      { role: "system", content: "You are a meticulous multilingual phonetician. Return strict JSON only." },
      { role: "user", content: prompt },
    ],
    response_format: { type: "json_object" },
  });
  const txt = completion.choices[0]?.message?.content ?? "{}";
  const parsed = JSON.parse(txt) as Partial<OracleVerdict>;
  return {
    similarity_0_100: typeof parsed.similarity_0_100 === "number" ? parsed.similarity_0_100 : NaN,
    reasoning: parsed.reasoning ?? "",
  };
}

// ---------- Stats ----------

function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0,
    dx = 0,
    dy = 0;
  for (let i = 0; i < n; i++) {
    const ax = xs[i]! - mx;
    const ay = ys[i]! - my;
    num += ax * ay;
    dx += ax * ax;
    dy += ay * ay;
  }
  return num / Math.sqrt(dx * dy);
}

function spearman(xs: number[], ys: number[]): number {
  const rank = (arr: number[]) => {
    const sorted = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const out = new Array(arr.length).fill(0);
    sorted.forEach((s, k) => (out[s.i] = k + 1));
    return out;
  };
  return pearson(rank(xs), rank(ys));
}

// ---------- Run ----------

interface Row {
  pair: Pair;
  our: number;
  ourComponents: { id: string; similarity: number }[];
  oracle: number;
  oracleReason: string;
  ourErr?: string;
  delta: number; // (our*100) - oracle  (positive = we score higher than oracle)
}

async function main() {
  console.log(`# Long-poem cross-lingual scoring benchmark\n`);
  console.log(`Pairs: ${PAIRS.length}. Endpoint: ${API}/homophones/compare (hybrid-phoneme-audio).`);
  console.log(`Independent oracle: gpt-5.4 phonetic-similarity rater (blind to our score).\n`);

  const rows: Row[] = [];
  // Sequential to keep server load + LLM bills polite. The /compare endpoint
  // already parallelizes its two sub-judges internally.
  for (const pair of PAIRS) {
    process.stdout.write(`· ${pair.group} | ${pair.label} ... `);
    const [our, oracle] = await Promise.all([ourScore(pair), oracleScore(pair).catch((e) => ({ similarity_0_100: NaN, reasoning: `oracle err: ${e}` }))]);
    const ourPct = our.similarity * 100;
    const row: Row = {
      pair,
      our: our.similarity,
      ourComponents: our.componentScores ?? [],
      oracle: oracle.similarity_0_100,
      oracleReason: oracle.reasoning,
      ourErr: our.error,
      delta: ourPct - oracle.similarity_0_100,
    };
    rows.push(row);
    console.log(
      our.error
        ? `ERR ${our.error}`
        : `our=${ourPct.toFixed(1)}% oracle=${oracle.similarity_0_100}% Δ=${row.delta.toFixed(1)}pt`,
    );
  }

  // ---------- Report ----------
  const valid = rows.filter((r) => !r.ourErr && Number.isFinite(r.our) && Number.isFinite(r.oracle));
  const ours = valid.map((r) => r.our * 100);
  const oracles = valid.map((r) => r.oracle);
  const r_pearson = pearson(ours, oracles);
  const r_spearman = spearman(ours, oracles);

  const byKind = (k: Pair["kind"]) => valid.filter((r) => r.pair.kind === k);
  const meanOf = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : NaN);

  const groupStats = (k: Pair["kind"]) => {
    const sub = byKind(k);
    return {
      n: sub.length,
      ourMean: meanOf(sub.map((r) => r.our * 100)),
      oracleMean: meanOf(sub.map((r) => r.oracle)),
    };
  };

  let md = `# Long-poem cross-lingual scoring benchmark\n\n`;
  md += `**Setup.** ${PAIRS.length} pairs through \`POST /api/homophones/compare\` with the default \`hybrid-phoneme-audio\` judge. ` +
    `Independent oracle: gpt-5.4 prompted to rate phonetic similarity 0-100 of each pair *blind* to our score, with no knowledge of our internal IPA, audio, or DTW machinery.\n\n`;
  md += `**Hypothesis.** If our scorer is well-calibrated, then across this mixed set of (a) parallel poem translations [should score LOW — meaning preserved, sound not], (b) known cross-lingual homophones [should score HIGH], and (c) totally unrelated control pairs [should score LOW], our score and the oracle should agree directionally. Pearson + Spearman correlation captures that.\n\n`;

  md += `## Headline numbers\n\n`;
  md += `| | n | our mean (hybrid) | oracle mean | sanity |\n`;
  md += `|---|---:|---:|---:|---|\n`;
  for (const k of ["translation", "homophone", "control"] as const) {
    const s = groupStats(k);
    const expected = k === "homophone" ? "should be HIGH" : "should be LOW";
    md += `| **${k}** | ${s.n} | ${s.ourMean.toFixed(1)}% | ${s.oracleMean.toFixed(1)}% | ${expected} |\n`;
  }
  md += `\n**Pearson r** (our% vs oracle%): \`${r_pearson.toFixed(3)}\`\n`;
  md += `**Spearman ρ** (rank correlation): \`${r_spearman.toFixed(3)}\`\n\n`;
  md += `Both should be strongly positive (~0.7+). A near-zero or negative correlation would mean our score and a blind expert disagree on what counts as "sounds alike".\n\n`;

  md += `## Per-pair results\n\n`;
  md += `| Group | Pair | Kind | Our combined | Phoneme | Acoustic | Oracle | Δ (ours − oracle) | Oracle reasoning |\n`;
  md += `|---|---|---|---:|---:|---:|---:|---:|---|\n`;
  for (const r of rows) {
    const ph = r.ourComponents.find((c) => c.id === "phoneme-chain");
    const ac = r.ourComponents.find((c) => c.id === "wav2vec2-dtw");
    md += `| ${r.pair.group} | ${r.pair.label} | ${r.pair.kind} | ${r.ourErr ? "ERR" : (r.our * 100).toFixed(1) + "%"} | ${ph ? (ph.similarity * 100).toFixed(0) + "%" : "—"} | ${ac ? (ac.similarity * 100).toFixed(0) + "%" : "—"} | ${Number.isFinite(r.oracle) ? r.oracle + "%" : "—"} | ${Number.isFinite(r.delta) ? (r.delta >= 0 ? "+" : "") + r.delta.toFixed(1) + "pt" : "—"} | ${r.oracleReason.replace(/\|/g, "/").slice(0, 180)} |\n`;
  }

  md += `\n## Interpretation rubric\n\n`;
  md += `- **Translations score low on both.** Confirms the scorer is not fooled by semantic equivalence — it measures sound, not meaning.\n`;
  md += `- **Homophones score high on both.** Confirms positive recall on the intended use case.\n`;
  md += `- **Controls score low on both.** Confirms no spurious-positive baseline noise.\n`;
  md += `- **Per-pair Δ small.** Confirms calibration (our scale ≈ a phonetician's intuitive scale).\n`;
  md += `- **High Pearson + Spearman.** Confirms ranking and magnitude both agree with an independent expert.\n`;

  const outPath = new URL("../poem-benchmark-report.md", import.meta.url);
  writeFileSync(outPath, md);
  console.log(`\nReport written to ${outPath.pathname}`);
  console.log(`\n=== Summary ===`);
  console.log(`Pearson r:  ${r_pearson.toFixed(3)}`);
  console.log(`Spearman ρ: ${r_spearman.toFixed(3)}`);
  for (const k of ["translation", "homophone", "control"] as const) {
    const s = groupStats(k);
    console.log(`${k.padEnd(12)} n=${s.n}  our=${s.ourMean.toFixed(1)}%  oracle=${s.oracleMean.toFixed(1)}%`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
