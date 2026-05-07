// Core linguistic to audio mapping
// Keeps simple and extensible

export interface Phoneme {
  symbol: string;
  frequency: number; // Hz base
  duration: number; // ms
  amplitude: number;
}

export const PHONEME_MAP: Record<string, Phoneme> = {
  'a': { symbol: 'a', frequency: 440, duration: 200, amplitude: 0.8 },
  'e': { symbol: 'e', frequency: 523.25, duration: 180, amplitude: 0.7 },
  'i': { symbol: 'i', frequency: 659.25, duration: 150, amplitude: 0.9 },
  'o': { symbol: 'o', frequency: 392, duration: 220, amplitude: 0.75 },
  'u': { symbol: 'u', frequency: 349.23, duration: 250, amplitude: 0.65 },
  // Add more vowels, consonants, diphthongs...
  ' ': { symbol: 'space', frequency: 0, duration: 50, amplitude: 0 }, // pause
};

export function textToPhonemes(text: string): Phoneme[] {
  const lower = text.toLowerCase();
  const phonemes: Phoneme[] = [];
  for (const char of lower) {
    if (PHONEME_MAP[char]) {
      phonemes.push(PHONEME_MAP[char]);
    } else if (/[a-z]/.test(char)) {
      // Fallback for unknown letters - creative mapping
      phonemes.push({ symbol: char, frequency: 300 + (char.charCodeAt(0) % 400), duration: 180, amplitude: 0.6 });
    }
  }
  return phonemes;
}

export function calculateWaveParameters(phonemes: Phoneme[]) {
  // Aggregate for overall sound wave
  const totalDuration = phonemes.reduce((sum, p) => sum + p.duration, 0);
  const avgFreq = phonemes.reduce((sum, p) => sum + p.frequency, 0) / phonemes.length || 440;
  return { totalDuration, avgFreq, phonemeCount: phonemes.length };
}
