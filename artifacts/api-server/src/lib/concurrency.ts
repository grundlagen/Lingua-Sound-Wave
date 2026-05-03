/** Run async work over an array with a max concurrency. Errors are captured per-item, not thrown. */
export async function mapWithLimit<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let idx = 0;
  const workerCount = Math.max(1, Math.min(limit, items.length));
  const workers = Array.from({ length: workerCount }, async () => {
    while (true) {
      const i = idx++;
      if (i >= items.length) return;
      try {
        results[i] = await fn(items[i]!, i);
      } catch (err) {
        results[i] = err as R;
      }
    }
  });
  await Promise.all(workers);
  return results;
}
