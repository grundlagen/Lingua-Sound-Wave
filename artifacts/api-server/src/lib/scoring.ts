import { dtwDistance, distanceToSimilarity, type AcousticFeatures } from "./dsp";
import { computeWav2VecFrames, meanPoolEmbedding, getWav2VecStatus } from "./wav2vec";
import { phonemeChainScore } from "./phoneme";

export interface ScoreInput {
  samples: Float32Array;
  sampleRate: number;
  features: AcousticFeatures;
  // Optional textual context — populated by callers that have it on hand.
  // Phoneme-based methods require these; audio-only methods ignore them.
  text?: string;
  language?: string;
  languageName?: string;
}

export interface ScoreResult {
  distance: number;
  similarity: number;
  method: string;
}

export interface ScoringMethodInfo {
  id: string;
  label: string;
  description: string;
  status: "ready" | "lazy" | "error";
  statusDetail?: string;
}

export interface ScoringMethod extends ScoringMethodInfo {
  score(src: ScoreInput, cand: ScoreInput): Promise<ScoreResult>;
}

// ---- Method 1: classic MFCC + DTW (calibrated against TTS anchors) ----

const MFCC_DTW: ScoringMethod = {
  id: "mfcc-dtw",
  label: "MFCC + DTW (classical)",
  description:
    "Mel-frequency cepstral coefficients with dynamic time warping on cosine distance. CPU-only, instant, calibrated against TTS anchor pairs. Default judge.",
  status: "ready",
  async score(src, cand) {
    const d = dtwDistance(src.features.mfcc, cand.features.mfcc);
    return { distance: d, similarity: distanceToSimilarity(d), method: this.id };
  },
};

// ---- Method 2: wav2vec2 mean-pool cosine (neural embeddings) ----

const WAV2VEC2_MEAN_COS: ScoringMethod = {
  id: "wav2vec2-mean-cos",
  label: "wav2vec2 mean-pool cosine (coarse neural baseline)",
  description:
    "Self-supervised wav2vec2-base hidden states mean-pooled over time, compared by cosine similarity. Captures broad acoustic style (speaker, channel, prosody) more than fine-grained phonetic alignment — when the same TTS voice synthesizes both clips, raw cosines stay in a narrow ~0.95–0.99 band, so we stretch them aggressively. Useful as a coarse baseline next to the more rigorous DTW methods. First call downloads ~95MB; subsequent calls ~0.5–2s per pair.",
  status: "lazy",
  async score(src, cand) {
    const [aFrames, bFrames] = await Promise.all([
      computeWav2VecFrames(src.samples, src.sampleRate),
      computeWav2VecFrames(cand.samples, cand.sampleRate),
    ]);
    const a = meanPoolEmbedding(aFrames);
    const b = meanPoolEmbedding(bFrames);
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
    // a and b are L2-normalized → dot is cosine similarity in [-1, 1].
    const cos = Math.max(-1, Math.min(1, dot));
    const distance = 1 - cos; // [0, 2]
    // Empirical TTS-vs-TTS cosines (single voice) cluster in [0.94, 0.995].
    // Stretch that band into [0, 1] so distinctions are visible.
    const sim = Math.max(0, Math.min(1, (cos - 0.94) / 0.055));
    return { distance, similarity: sim, method: this.id };
  },
};

// ---- Method 3: wav2vec2 frame DTW (most rigorous neural method) ----

const WAV2VEC2_DTW: ScoringMethod = {
  id: "wav2vec2-dtw",
  label: "wav2vec2 frame DTW (neural, rigorous)",
  description:
    "Frame-level wav2vec2 embeddings (~50 frames/sec) aligned via DTW on cosine distance. Most rigorous judge — handles speaking-rate variation while preserving phonetic detail. Slowest method (~1–4s per pair after model load).",
  status: "lazy",
  async score(src, cand) {
    const [aFrames, bFrames] = await Promise.all([
      computeWav2VecFrames(src.samples, src.sampleRate),
      computeWav2VecFrames(cand.samples, cand.sampleRate),
    ]);
    const d = dtwDistance(aFrames, bFrames);
    // Empirical wav2vec frame DTW distances on TTS:
    //   identical pair       d ≈ 0.02–0.05
    //   strong homophone     d ≈ 0.10–0.20
    //   unrelated phrases    d ≈ 0.35–0.55
    const adjusted = Math.max(0, d - 0.04);
    const sim = Math.max(0, Math.min(1, Math.exp(-adjusted * 5.5)));
    return { distance: d, similarity: sim, method: this.id };
  },
};

// NOTE: An earlier "allophone-chain" method (small ~50ms MFCC units → free
// nearest-neighbor chain with gap penalty) failed to discriminate because
// single-voice TTS gave every source unit a cheap voice-floor match in the
// candidate's inventory, so the gap threshold rarely fired. That motivated
// moving to symbolic IPA space (PHONEME_CHAIN below), where there is no
// voice-floor noise.

// ---- Method 4: phoneme chain (G2P → IPA → featural alignment + variants) ----

const PHONEME_CHAIN: ScoringMethod = {
  id: "phoneme-chain",
  label: "Phoneme chain (IPA + featural alignment + variant chaining)",
  description:
    "Linguistically-informed cross-lingual matcher. Skips audio entirely: an LLM converts each phrase to IPA along with up to 3 alternate pronunciation variants (fast-speech, schwa reduction, devoicing, dialect substitutions). IPA strings are aligned with weighted Needleman–Wunsch using a phonetic feature distance (place/manner/voicing for consonants; height/backness/rounding for vowels) plus equivalence-class shortcuts (rhotic family, l-vocalization, TH-fronting, sibilant family, voicing pairs, schwa↔reduced vowels, nasal place mismatches). Picks the best alignment across all source-variant × target-variant chains. First call per phrase ≈1–2s LLM; cached afterwards.",
  status: "ready",
  async score(src, cand) {
    if (!src.text || !cand.text || !src.language || !cand.language) {
      return { distance: 1.5, similarity: 0, method: this.id };
    }
    const r = await phonemeChainScore(
      src.text, src.language, src.languageName ?? src.language,
      cand.text, cand.language, cand.languageName ?? cand.language,
    );
    return { distance: r.distance, similarity: r.similarity, method: this.id };
  },
};

// ---- Method 5: hybrid (phoneme-chain × wav2vec2-dtw) ----
//
// Geometric mean of the two strongest discriminators. Rationale: their failure
// modes are largely independent — phoneme-chain can be fooled by bad LLM G2P;
// wav2vec2-dtw can be fooled by TTS voice-floor on prosodically similar but
// phonetically unrelated phrases (e.g. P5 on the 8-pair benchmark). Requiring
// BOTH to agree pulls down each method's idiosyncratic false positives while
// preserving genuine homophones (where both naturally agree).
//
// We use the geometric mean rather than a min so a single very-low score
// doesn't dominate when the other method is mildly uncertain. Distance is
// reported as the average of the two normalized distances for display.

const HYBRID_PHONEME_AUDIO: ScoringMethod = {
  id: "hybrid-phoneme-audio",
  label: "Hybrid (phoneme chain × wav2vec2 DTW)",
  description:
    "Geometric mean of the symbolic phoneme-chain matcher and the wav2vec2-dtw acoustic matcher. Each method's idiosyncratic false positives get pulled down by the other (e.g. wav2vec2-dtw's TTS-voice-floor false positives on prosodically similar but phonetically unrelated phrases are corrected by phoneme-chain's symbolic 'no'). Requires both LLM G2P and the wav2vec2 model — slowest method, most reliable.",
  status: "lazy",
  async score(src, cand) {
    const [pho, aco] = await Promise.all([
      PHONEME_CHAIN.score(src, cand),
      WAV2VEC2_DTW.score(src, cand),
    ]);
    const sim = Math.sqrt(Math.max(0, pho.similarity) * Math.max(0, aco.similarity));
    const distance = (pho.distance + aco.distance) / 2;
    return { distance, similarity: sim, method: this.id };
  },
};

// ---- Method 6: hybrid (phoneme-chain × MFCC-DTW) ----
//
// MFCC-DTW has a wider relative spread than wav2vec2-dtw on TTS (negatives
// cluster ~20-30%, positives ~50-90%) because it doesn't inherit the
// neural-model voice-floor that wav2vec2 picks up from same-voice synthesis.
// Combining it with phoneme-chain (geometric mean) confirms phonetic matches
// acoustically without dragging negatives upward.
//
// Empirically (8-pair benchmark) this hybrid matches or beats both inputs.
// Cheaper than hybrid-phoneme-audio: no neural model required.

const HYBRID_PHONEME_MFCC: ScoringMethod = {
  id: "hybrid-phoneme-mfcc",
  label: "Hybrid (phoneme chain × MFCC DTW)",
  description:
    "Geometric mean of the symbolic phoneme-chain matcher and the classical MFCC + DTW matcher. MFCC-DTW has a wider negative/positive spread on TTS than wav2vec2-dtw (no neural voice-floor inflation), so it provides cleaner acoustic confirmation without dragging unrelated pairs upward. Cheaper than the wav2vec2 hybrid — no neural model needed.",
  status: "ready",
  async score(src, cand) {
    const [pho, aco] = await Promise.all([
      PHONEME_CHAIN.score(src, cand),
      MFCC_DTW.score(src, cand),
    ]);
    const sim = Math.sqrt(Math.max(0, pho.similarity) * Math.max(0, aco.similarity));
    const distance = (pho.distance + aco.distance) / 2;
    return { distance, similarity: sim, method: this.id };
  },
};

const ALL_METHODS: ScoringMethod[] = [MFCC_DTW, WAV2VEC2_MEAN_COS, WAV2VEC2_DTW, PHONEME_CHAIN, HYBRID_PHONEME_AUDIO, HYBRID_PHONEME_MFCC];

export const DEFAULT_METHOD_ID = "mfcc-dtw";

export function getScoringMethod(id: string | undefined | null): ScoringMethod {
  if (!id) return MFCC_DTW;
  return ALL_METHODS.find((m) => m.id === id) ?? MFCC_DTW;
}

export function listScoringMethods(): ScoringMethodInfo[] {
  const w2v = getWav2VecStatus();
  return ALL_METHODS.map((m) => {
    if (m.id === "mfcc-dtw" || m.id === "phoneme-chain" || m.id === "hybrid-phoneme-mfcc") {
      return { id: m.id, label: m.label, description: m.description, status: "ready" as const };
    }
    if (m.id === "hybrid-phoneme-audio") {
      // Inherit the wav2vec2 status since the hybrid needs the model.
      if (w2v.lastError) return { id: m.id, label: m.label, description: m.description, status: "error" as const, statusDetail: w2v.lastError };
      if (w2v.ready) return { id: m.id, label: m.label, description: m.description, status: "ready" as const, statusDetail: "phoneme G2P + wav2vec2 ready" };
      return { id: m.id, label: m.label, description: m.description, status: "lazy" as const, statusDetail: w2v.loading ? "downloading wav2vec2 model…" : "downloads wav2vec2 (~95MB) on first use" };
    }
    if (w2v.lastError) {
      return {
        id: m.id,
        label: m.label,
        description: m.description,
        status: "error" as const,
        statusDetail: w2v.lastError,
      };
    }
    if (w2v.ready) {
      const ms = w2v.loadDurationMs;
      return {
        id: m.id,
        label: m.label,
        description: m.description,
        status: "ready" as const,
        statusDetail: ms ? `model loaded in ${(ms / 1000).toFixed(1)}s` : "model loaded",
      };
    }
    return {
      id: m.id,
      label: m.label,
      description: m.description,
      status: "lazy" as const,
      statusDetail: w2v.loading ? "downloading model…" : "downloads ~95MB on first use",
    };
  });
}
