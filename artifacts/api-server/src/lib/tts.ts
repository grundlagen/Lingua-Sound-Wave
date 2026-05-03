import { textToSpeech } from "@workspace/integrations-openai-ai-server/audio";
import {
  pcm16ToFloat32,
  trimSilence,
  downsampleWaveform,
  float32ToWav,
  extractFeatures,
  TTS_SAMPLE_RATE,
  type AcousticFeatures,
} from "./dsp";

export interface SynthesizedAudio {
  wavBase64: string;
  sampleRate: number;
  durationMs: number;
  waveform: number[];
  melSpectrogram: number[][];
  melMin: number;
  melMax: number;
  features: AcousticFeatures;
  samples: Float32Array;
}

/** Synthesize a phrase via OpenAI gpt-audio (PCM16 @ 24kHz) and extract features. */
export async function synthesize(text: string, voice: "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer" = "alloy"): Promise<SynthesizedAudio> {
  const pcm = await textToSpeech(text, voice, "pcm16");
  let samples = pcm16ToFloat32(pcm);
  samples = trimSilence(samples, TTS_SAMPLE_RATE);
  const waveform = downsampleWaveform(samples, 200);
  const features = extractFeatures(samples, TTS_SAMPLE_RATE);
  const wav = float32ToWav(samples, TTS_SAMPLE_RATE);
  return {
    wavBase64: wav.toString("base64"),
    sampleRate: TTS_SAMPLE_RATE,
    durationMs: features.durationMs,
    waveform,
    melSpectrogram: features.melSpectrogram,
    melMin: features.melMin,
    melMax: features.melMax,
    features,
    samples,
  };
}

/** Strip the heavy internal fields before returning to client. */
export function toAudioPayload(a: SynthesizedAudio) {
  return {
    wavBase64: a.wavBase64,
    sampleRate: a.sampleRate,
    durationMs: a.durationMs,
    waveform: a.waveform,
    melSpectrogram: a.melSpectrogram,
    melMin: a.melMin,
    melMax: a.melMax,
  };
}
