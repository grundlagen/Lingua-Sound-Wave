const cache = new Map<string, string>();
const MAX_ENTRIES = 64;

function base64ToBlobUrl(b64: string, mime = "audio/wav"): string {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: mime });
  return URL.createObjectURL(blob);
}

export function audioUrlFor(b64: string): string {
  const existing = cache.get(b64);
  if (existing !== undefined) {
    // refresh LRU order
    cache.delete(b64);
    cache.set(b64, existing);
    return existing;
  }
  const url = base64ToBlobUrl(b64);
  cache.set(b64, url);
  // Evict oldest entries past the cap, revoking their object URLs.
  while (cache.size > MAX_ENTRIES) {
    const oldestKey = cache.keys().next().value as string | undefined;
    if (!oldestKey) break;
    const oldUrl = cache.get(oldestKey);
    cache.delete(oldestKey);
    if (oldUrl) URL.revokeObjectURL(oldUrl);
  }
  return url;
}
