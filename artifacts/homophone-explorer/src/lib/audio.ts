export function base64ToBlobUrl(b64: string, mime = "audio/wav"): string {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: mime });
  return URL.createObjectURL(blob);
}

const cache = new Map<string, string>();

export function audioUrlFor(b64: string): string {
  let url = cache.get(b64);
  if (!url) {
    url = base64ToBlobUrl(b64);
    cache.set(b64, url);
  }
  return url;
}
