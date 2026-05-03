import { useEffect, useRef } from "react";

interface Props {
  data: number[];
  height?: number;
  color?: string;
  className?: string;
}

export function Waveform({ data, height = 56, color = "#6366f1", className }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width * dpr));
    const h = Math.floor(height * dpr);
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, w, h);
    if (data.length === 0) return;
    const mid = h / 2;
    const step = w / data.length;
    ctx.fillStyle = color;
    for (let i = 0; i < data.length; i++) {
      const v = Math.max(0, Math.min(1, Math.abs(data[i]!)));
      const barH = Math.max(1, v * (h * 0.95));
      const x = Math.floor(i * step);
      const wBar = Math.max(1, Math.floor(step) - 1);
      ctx.fillRect(x, mid - barH / 2, wBar, barH);
    }
  }, [data, height, color]);

  return (
    <canvas
      ref={ref}
      className={className}
      style={{ width: "100%", height }}
      data-testid="waveform-canvas"
    />
  );
}
