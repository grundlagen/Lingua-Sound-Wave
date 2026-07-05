"""Legacy vs calibrated composer ranking, one load, per-line progress.

Usage: python -u compare_bench.py [n_lines]
"""
import sys
import time

import beauty_compose as BC

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
CAL = BC._CAL
assert CAL is not None, "channel-calibration.json missing"

t0 = time.time()
D = BC.load_all()
print(f"loaded in {time.time() - t0:.0f}s", flush=True)

lines = []
for i, line in enumerate(open("corpus-carves.tsv", encoding="utf-8")):
    if i == 0:
        continue
    lines.append(line.split("\t")[0])
    if len(lines) >= N:
        break

res = {"legacy": [0, []], "calibrated": [0, []]}
for k, en in enumerate(lines):
    for mode in ("legacy", "calibrated"):
        BC._CAL = None if mode == "legacy" else CAL
        t = time.time()
        s, m = BC.translate(en, D, show=False)
        ok = s >= 0.55 and m >= 0.45
        res[mode][0] += ok
        res[mode][1].append((s, m))
        print(f"[{k + 1}/{N}] {mode:10s} s={s:.2f} m={m:.2f} "
              f"{'PASS' if ok else 'fail'} ({time.time() - t:.0f}s)  {en[:50]}",
              flush=True)

for mode, (band, sm) in res.items():
    ms = sum(x for x, _ in sm) / len(sm)
    mm = sum(x for _, x in sm) / len(sm)
    print(f"{mode:10s}: {band}/{N} = {band / N:.0%} joint  "
          f"(mean sound {ms:.2f}, mean meaning {mm:.2f})")
