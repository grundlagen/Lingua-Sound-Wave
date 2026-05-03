import { dtwDistance, distanceToSimilarity, type AcousticFeatures } from "./dsp";
import { computeWav2VecFrames, meanPoolEmbedding, getWav2VecStatus } from "./wav2vec";

export interface ScoreInput {
  samples: Float32Array;
  sampleRate: number;
  features: AcousticFeatures;
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

// NOTE: An "allophone-chain" side-project method (small ~50ms MFCC units →
// nearest-neighbor chain with explicit gap handling) was prototyped and
// removed after testing showed it failed to discriminate. The free chain with
// reuse let unrelated TTS clips pull cheap voice-floor matches across the
// candidate's inventory, so the gap threshold rarely fired and negatives and
// positives collapsed into the same ~50–60% band (mean(neg)=54.7%,
// mean(pos)=58.9%, only +4.2 pts of separation; H1 "knee how"↔你好 scored
// 44.6%, lower than several translations). To revive: enforce no-reuse and
// monotonic ordering, or move to G2P+IPA phoneme units.

const ALL_METHODS: ScoringMethod[] = [MFCC_DTW, WAV2VEC2_MEAN_COS, WAV2VEC2_DTW];

export const DEFAULT_METHOD_ID = "mfcc-dtw";

export function getScoringMethod(id: string | undefined | null): ScoringMethod {
  if (!id) return MFCC_DTW;
  return ALL_METHODS.find((m) => m.id === id) ?? MFCC_DTW;
}

export function listScoringMethods(): ScoringMethodInfo[] {
  const w2v = getWav2VecStatus();
  return ALL_METHODS.map((m) => {
    if (m.id === "mfcc-dtw") {
      return { id: m.id, label: m.label, description: m.description, status: "ready" as const };
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
