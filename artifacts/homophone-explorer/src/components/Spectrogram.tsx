import { useEffect, useRef } from "react";

interface Props {
  frames: number[][];
  melMin: number;
  melMax: number;
  height?: number;
  className?: string;
}

function viridis(t: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [
    [0.0, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1.0, [253, 231, 37]],
  ];
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i]!;
    const [t0, c0] = stops[i - 1]!;
    if (x <= t1) {
      const u = (x - t0) / (t1 - t0);
      return [
        c0[0] + (c1[0] - c0[0]) * u,
        c0[1] + (c1[1] - c0[1]) * u,
        c0[2] + (c1[2] - c0[2]) * u,
      ];
    }
  }
  return stops[stops.length - 1]![1];
}

export function Spectrogram({ frames, melMin, melMax, height = 96, className }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || frames.length === 0) return;
    const bins = frames[0]!.length;
    const nFrames = frames.length;
    canvas.width = nFrames;
    canvas.height = bins;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(nFrames, bins);
    const range = Math.max(1e-6, melMax - melMin);
    for (let x = 0; x < nFrames; x++) {
      const col = frames[x]!;
      for (let y = 0; y < bins; y++) {
        const v = (col[y]! - melMin) / range;
        const [r, g, b] = viridis(v);
        const py = bins - 1 - y;
        const idx = (py * nFrames + x) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [frames, melMin, melMax]);

  return (
    <canvas
      ref={ref}
      className={className}
      style={{ width: "100%", height, imageRendering: "pixelated", borderRadius: 6 }}
      data-testid="spectrogram-canvas"
    />
  );
}
