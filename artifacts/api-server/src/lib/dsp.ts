// eslint-disable-next-line @typescript-eslint/no-explicit-any
const FFT: any = require("fft.js");

// All TTS audio is requested as PCM16 mono at 24kHz from gpt-audio.
export const TTS_SAMPLE_RATE = 24000;

/** Decode 16-bit signed little-endian PCM bytes to Float32 samples in [-1, 1]. */
export function pcm16ToFloat32(buf: Buffer): Float32Array {
  const out = new Float32Array(Math.floor(buf.length / 2));
  for (let i = 0; i < out.length; i++) {
    const lo = buf[i * 2]!;
    const hi = buf[i * 2 + 1]!;
    let v = (hi << 8) | lo;
    if (v & 0x8000) v -= 0x10000;
    out[i] = v / 32768;
  }
  return out;
}

/** Encode Float32 samples to a WAV file buffer at the given sample rate. */
export function float32ToWav(samples: Float32Array, sampleRate: number): Buffer {
  const numSamples = samples.length;
  const byteRate = sampleRate * 2;
  const buffer = Buffer.alloc(44 + numSamples * 2);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + numSamples * 2, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(1, 22); // mono
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(numSamples * 2, 40);
  for (let i = 0; i < numSamples; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]!));
    s = s < 0 ? s * 0x8000 : s * 0x7fff;
    buffer.writeInt16LE(s | 0, 44 + i * 2);
  }
  return buffer;
}

/** Trim leading/trailing silence based on RMS energy. */
export function trimSilence(samples: Float32Array, sampleRate: number, threshold = 0.005): Float32Array {
  const win = Math.max(1, Math.floor(sampleRate * 0.02));
  const frames: number[] = [];
  for (let i = 0; i < samples.length; i += win) {
    let sum = 0;
    const end = Math.min(samples.length, i + win);
    for (let j = i; j < end; j++) sum += samples[j]! * samples[j]!;
    frames.push(Math.sqrt(sum / (end - i)));
  }
  let start = 0;
  while (start < frames.length && frames[start]! < threshold) start++;
  let stop = frames.length - 1;
  while (stop > start && frames[stop]! < threshold) stop--;
  const a = start * win;
  const b = Math.min(samples.length, (stop + 1) * win);
  if (b <= a) return samples;
  return samples.slice(a, b);
}

/** Downsample to N peaks for waveform visualization, normalized to [-1, 1]. */
export function downsampleWaveform(samples: Float32Array, n: number): number[] {
  const peaks = new Array<number>(n).fill(0);
  if (samples.length === 0) return peaks;
  const step = samples.length / n;
  let maxAbs = 1e-9;
  for (let i = 0; i < n; i++) {
    const start = Math.floor(i * step);
    const end = Math.max(start + 1, Math.floor((i + 1) * step));
    let max = 0;
    for (let j = start; j < end && j < samples.length; j++) {
      const a = Math.abs(samples[j]!);
      if (a > max) max = a;
    }
    peaks[i] = max;
    if (max > maxAbs) maxAbs = max;
  }
  for (let i = 0; i < n; i++) peaks[i] = peaks[i]! / maxAbs;
  return peaks;
}

// ---------- MFCC ----------

const FRAME_MS = 25;
const HOP_MS = 10;
const N_MELS = 40;
const N_MFCC = 13;
// FFT_SIZE must be >= frameLen. At 24kHz, 25ms = 600 samples → use 1024.
const FFT_SIZE = 1024;

function hzToMel(f: number): number {
  return 2595 * Math.log10(1 + f / 700);
}
function melToHz(m: number): number {
  return 700 * (Math.pow(10, m / 2595) - 1);
}

function buildMelFilterbank(nMels: number, fftSize: number, sampleRate: number): Float32Array[] {
  const fmin = 0;
  const fmax = sampleRate / 2;
  const melMin = hzToMel(fmin);
  const melMax = hzToMel(fmax);
  const melPoints = new Array(nMels + 2).fill(0).map((_, i) => melMin + ((melMax - melMin) * i) / (nMels + 1));
  const hzPoints = melPoints.map(melToHz);
  const bin = hzPoints.map((hz) => Math.floor(((fftSize + 1) * hz) / sampleRate));
  const filters: Float32Array[] = [];
  const halfFft = fftSize / 2 + 1;
  for (let m = 1; m <= nMels; m++) {
    const f = new Float32Array(halfFft);
    const left = bin[m - 1]!;
    const center = bin[m]!;
    const right = bin[m + 1]!;
    for (let k = left; k < center; k++) {
      if (center === left) continue;
      f[k] = (k - left) / (center - left);
    }
    for (let k = center; k < right; k++) {
      if (right === center) continue;
      f[k] = (right - k) / (right - center);
    }
    filters.push(f);
  }
  return filters;
}

function hammingWindow(n: number): Float32Array {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

function dctII(input: Float32Array, nOut: number): Float32Array {
  const N = input.length;
  const out = new Float32Array(nOut);
  for (let k = 0; k < nOut; k++) {
    let s = 0;
    for (let n = 0; n < N; n++) {
      s += input[n]! * Math.cos((Math.PI / N) * (n + 0.5) * k);
    }
    out[k] = s;
  }
  return out;
}

export interface AcousticFeatures {
  mfcc: Float32Array[]; // frames of 13 MFCC + delta + delta2 (39)
  melSpectrogram: number[][]; // frames of N_MELS log-energies (for viz)
  melMin: number;
  melMax: number;
  durationMs: number;
}

export function extractFeatures(samples: Float32Array, sampleRate: number): AcousticFeatures {
  // Pre-emphasis
  const pre = new Float32Array(samples.length);
  pre[0] = samples[0] ?? 0;
  for (let i = 1; i < samples.length; i++) pre[i] = samples[i]! - 0.97 * samples[i - 1]!;

  const frameLen = Math.floor((FRAME_MS / 1000) * sampleRate);
  const hop = Math.floor((HOP_MS / 1000) * sampleRate);
  const window = hammingWindow(frameLen);
  const fft = new FFT(FFT_SIZE);
  const fbank = buildMelFilterbank(N_MELS, FFT_SIZE, sampleRate);
  const halfFft = FFT_SIZE / 2 + 1;

  const mfccFrames: Float32Array[] = [];
  const mels: number[][] = [];
  let melMin = Infinity;
  let melMax = -Infinity;

  const fftIn = new Float32Array(FFT_SIZE);
  const fftOut = fft.createComplexArray();
  const fftInComplex = fft.createComplexArray();

  for (let start = 0; start + frameLen <= pre.length; start += hop) {
    fftIn.fill(0);
    for (let i = 0; i < frameLen; i++) fftIn[i] = pre[start + i]! * window[i]!;
    fft.toComplexArray(fftIn, fftInComplex);
    fft.transform(fftOut, fftInComplex);
    const power = new Float32Array(halfFft);
    for (let k = 0; k < halfFft; k++) {
      const re = fftOut[2 * k]!;
      const im = fftOut[2 * k + 1]!;
      power[k] = (re * re + im * im) / FFT_SIZE;
    }
    const melE = new Float32Array(N_MELS);
    const logMel = new Array<number>(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      const f = fbank[m]!;
      for (let k = 0; k < halfFft; k++) s += power[k]! * f[k]!;
      melE[m] = s;
      const lg = Math.log(s + 1e-10);
      logMel[m] = lg;
      if (lg < melMin) melMin = lg;
      if (lg > melMax) melMax = lg;
    }
    mels.push(logMel);
    const logE = new Float32Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) logE[m] = Math.log(melE[m]! + 1e-10);
    const mfcc = dctII(logE, N_MFCC);
    // CMN: subtract mean later
    mfccFrames.push(mfcc);
  }

  // Cepstral mean normalization
  if (mfccFrames.length > 0) {
    const mean = new Float32Array(N_MFCC);
    for (const f of mfccFrames) for (let i = 0; i < N_MFCC; i++) mean[i]! += f[i]!;
    for (let i = 0; i < N_MFCC; i++) mean[i]! /= mfccFrames.length;
    for (const f of mfccFrames) for (let i = 0; i < N_MFCC; i++) f[i]! -= mean[i]!;
  }

  // Append deltas
  const delta = computeDelta(mfccFrames);
  const delta2 = computeDelta(delta);
  const combined: Float32Array[] = mfccFrames.map((f, i) => {
    const c = new Float32Array(N_MFCC * 3);
    c.set(f, 0);
    c.set(delta[i]!, N_MFCC);
    c.set(delta2[i]!, N_MFCC * 2);
    return c;
  });

  return {
    mfcc: combined,
    melSpectrogram: mels,
    melMin: isFinite(melMin) ? melMin : -10,
    melMax: isFinite(melMax) ? melMax : 0,
    durationMs: (samples.length / sampleRate) * 1000,
  };
}

function computeDelta(frames: Float32Array[]): Float32Array[] {
  const N = 2;
  const denom = 2 * (1 * 1 + 2 * 2);
  const out: Float32Array[] = frames.map(() => new Float32Array(N_MFCC));
  for (let t = 0; t < frames.length; t++) {
    for (let i = 0; i < N_MFCC; i++) {
      let s = 0;
      for (let n = 1; n <= N; n++) {
        const tp = Math.min(frames.length - 1, t + n);
        const tn = Math.max(0, t - n);
        s += n * (frames[tp]![i]! - frames[tn]![i]!);
      }
      out[t]![i] = s / denom;
    }
  }
  return out;
}

// ---------- DTW ----------

function cosineDistance(a: Float32Array, b: Float32Array): number {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  if (denom === 0) return 1;
  const cos = dot / denom;
  return 1 - cos; // [0, 2]
}

/** Unconstrained DTW. Returns the average per-step cosine distance along the optimal warping path (∈ [0, 2]). */
export function dtwDistance(a: Float32Array[], b: Float32Array[]): number {
  if (a.length === 0 || b.length === 0) return 2;
  const n = a.length;
  const m = b.length;
  const INF = Number.POSITIVE_INFINITY;
  // Track cumulative cost AND the path length used to reach each cell so we
  // can normalize by actual path length, not by the (n+m) upper bound.
  const prevCost = new Float64Array(m + 1).fill(INF);
  const currCost = new Float64Array(m + 1).fill(INF);
  const prevLen = new Int32Array(m + 1);
  const currLen = new Int32Array(m + 1);
  prevCost[0] = 0;
  for (let i = 1; i <= n; i++) {
    currCost[0] = INF;
    currLen[0] = 0;
    for (let j = 1; j <= m; j++) {
      const cost = cosineDistance(a[i - 1]!, b[j - 1]!);
      const cDel = prevCost[j]!; // (i-1, j) — vertical
      const cIns = currCost[j - 1]!; // (i, j-1) — horizontal
      const cDia = prevCost[j - 1]!; // (i-1, j-1) — diagonal
      let best = cDia;
      let bestLen = prevLen[j - 1]!;
      if (cDel < best) { best = cDel; bestLen = prevLen[j]!; }
      if (cIns < best) { best = cIns; bestLen = currLen[j - 1]!; }
      currCost[j] = cost + best;
      currLen[j] = bestLen + 1;
    }
    prevCost.set(currCost);
    prevLen.set(currLen);
  }
  const total = prevCost[m]!;
  const len = prevLen[m]!;
  if (!isFinite(total) || len <= 0) return 2;
  return total / len;
}

/** Convert raw DTW distance (avg per-step cosine distance, [0,2]) to a [0,1] similarity score. */
export function distanceToSimilarity(d: number): number {
  // Calibrated against TTS anchor pairs:
  //   "hello"/"hello"           d≈0.07 → sim≈0.89  (TTS noise floor)
  //   "knee how"/"你好"          d≈0.07 → sim≈0.89  (true cross-lingual homophone)
  //   "so"/"saw"                d≈0.05 → sim≈0.95
  //   "hello"/"halo"            d≈0.45 → sim≈0.20  (loosely similar)
  //   "hello"/"banana republic" d≈0.52 → sim≈0.15  (unrelated)
  // The 0.04 floor accounts for irreducible TTS reproducibility noise.
  const adjusted = Math.max(0, d - 0.04);
  const sim = Math.exp(-adjusted * 4.0);
  return Math.max(0, Math.min(1, sim));
}
