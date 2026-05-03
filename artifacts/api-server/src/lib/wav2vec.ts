import { resample } from "./dsp";
import { logger } from "./logger";

const MODEL_ID = "Xenova/wav2vec2-base-960h";
const TARGET_RATE = 16000;

let loadPromise: Promise<{ processor: unknown; model: unknown }> | null = null;
let ready = false;
let lastError: string | null = null;
let loadStartedAt: number | null = null;
let loadFinishedAt: number | null = null;

export function getWav2VecStatus() {
  return {
    modelId: MODEL_ID,
    ready,
    loading: loadPromise !== null && !ready,
    lastError,
    loadStartedAt,
    loadFinishedAt,
    loadDurationMs: loadStartedAt && loadFinishedAt ? loadFinishedAt - loadStartedAt : null,
  };
}

async function loadModel(): Promise<{ processor: unknown; model: unknown }> {
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    loadStartedAt = Date.now();
    try {
      logger.info({ modelId: MODEL_ID }, "wav2vec: starting model download/load");
      const tx = await import("@huggingface/transformers");
      tx.env.allowLocalModels = false;
      tx.env.useBrowserCache = false;
      const processor = await tx.AutoProcessor.from_pretrained(MODEL_ID);
      const model = await tx.AutoModel.from_pretrained(MODEL_ID, { dtype: "fp32" });
      loadFinishedAt = Date.now();
      ready = true;
      lastError = null;
      logger.info(
        { modelId: MODEL_ID, ms: loadFinishedAt - (loadStartedAt ?? loadFinishedAt) },
        "wav2vec: model ready",
      );
      return { processor, model };
    } catch (e) {
      lastError = e instanceof Error ? e.message : String(e);
      logger.error({ err: e, modelId: MODEL_ID }, "wav2vec: model load failed");
      loadPromise = null;
      throw e;
    }
  })();
  return loadPromise;
}

/** Kick off loading without awaiting — useful at startup. */
export function preloadWav2Vec(): void {
  loadModel().catch(() => {
    /* errors already logged */
  });
}

interface ModelOutput {
  last_hidden_state?: { data: Float32Array; dims: number[] };
  logits?: { data: Float32Array; dims: number[] };
  [key: string]: unknown;
}

// Per-process frame cache keyed by the source samples buffer. Because Discover
// and Translate score one source against many candidates, the same source
// Float32Array is passed to computeWav2VecFrames N times — caching by reference
// avoids N-1 redundant ~hundred-ms forward passes per request.
const frameCache = new WeakMap<Float32Array, Promise<Float32Array[]>>();

/** Run wav2vec2 on a (resampled to 16kHz) audio buffer and return frame-level embedding vectors.
 *  Results are memoized by the input Float32Array reference. */
export async function computeWav2VecFrames(samples: Float32Array, sampleRate: number): Promise<Float32Array[]> {
  const cached = frameCache.get(samples);
  if (cached) return cached;
  const promise = (async () => {
    const audio = sampleRate === TARGET_RATE ? samples : resample(samples, sampleRate, TARGET_RATE);
    if (audio.length < 400) {
      const pad = new Float32Array(400);
      pad.set(audio);
      return runForward(pad);
    }
    return runForward(audio);
  })();
  frameCache.set(samples, promise);
  // If the forward pass throws, evict so the next call can retry.
  promise.catch(() => frameCache.delete(samples));
  return promise;
}

async function runForward(audio: Float32Array): Promise<Float32Array[]> {
  const { processor, model } = await loadModel();
  // The processor accepts a Float32Array of mono samples at sampling_rate.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const inputs = await (processor as any)(audio, { sampling_rate: TARGET_RATE });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const out = (await (model as any)(inputs)) as ModelOutput;
  const tensor = out.last_hidden_state ?? out.logits;
  if (!tensor) {
    throw new Error(`wav2vec model returned no usable tensor (keys: ${Object.keys(out).join(", ")})`);
  }
  const dims = tensor.dims; // [batch=1, T, D]
  if (dims.length !== 3 || dims[0] !== 1) {
    throw new Error(`unexpected wav2vec tensor dims: [${dims.join(",")}]`);
  }
  const T = dims[1]!;
  const D = dims[2]!;
  const data = tensor.data;
  const frames: Float32Array[] = new Array(T);
  for (let i = 0; i < T; i++) {
    frames[i] = data.slice(i * D, (i + 1) * D);
  }
  return frames;
}

/** Mean-pool frame embeddings over time and L2-normalize. */
export function meanPoolEmbedding(frames: Float32Array[]): Float32Array {
  if (frames.length === 0) return new Float32Array(0);
  const D = frames[0]!.length;
  const out = new Float32Array(D);
  for (const f of frames) for (let i = 0; i < D; i++) out[i]! += f[i]!;
  let norm = 0;
  for (let i = 0; i < D; i++) {
    out[i]! /= frames.length;
    norm += out[i]! * out[i]!;
  }
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < D; i++) out[i]! /= norm;
  return out;
}
