interface Props {
  value: number;
  showLabel?: boolean;
}

function colorFor(v: number): string {
  if (v >= 0.85) return "#10b981";
  if (v >= 0.7) return "#22c55e";
  if (v >= 0.55) return "#eab308";
  return "#94a3b8";
}

function verdictFor(v: number): string {
  if (v >= 0.85) return "Near-identical";
  if (v >= 0.7) return "Strong match";
  if (v >= 0.55) return "Moderate match";
  return "Weak match";
}

export function SimilarityMeter({ value, showLabel = true }: Props) {
  const pct = Math.round(value * 100);
  const color = colorFor(value);
  return (
    <div className="space-y-1" data-testid="similarity-meter">
      <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      {showLabel ? (
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium" style={{ color }}>
            {verdictFor(value)}
          </span>
          <span className="tabular-nums text-muted-foreground">{pct}% acoustic similarity</span>
        </div>
      ) : null}
    </div>
  );
}
