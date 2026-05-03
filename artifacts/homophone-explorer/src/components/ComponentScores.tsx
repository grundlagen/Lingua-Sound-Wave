import type { ComponentScore } from "@workspace/api-client-react";

interface Props {
  components: ComponentScore[];
  size?: "sm" | "md";
  testId?: string;
}

function colorForSim(sim: number): string {
  if (sim >= 0.7) return "bg-emerald-500/15 text-emerald-700 border-emerald-500/30";
  if (sim >= 0.5) return "bg-amber-500/15 text-amber-700 border-amber-500/30";
  return "bg-slate-500/10 text-slate-600 border-slate-500/30";
}

export function ComponentScores({ components, size = "md", testId }: Props) {
  if (!components || components.length === 0) return null;
  const sm = size === "sm";
  return (
    <div
      className={`flex flex-wrap gap-2 ${sm ? "text-[11px]" : "text-xs"}`}
      data-testid={testId ?? "component-scores"}
    >
      {components.map((c) => (
        <div
          key={c.id}
          className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 ${colorForSim(c.similarity)}`}
          data-testid={`component-${c.id}`}
        >
          <span className="opacity-80">{c.label}</span>
          <span className="font-semibold tabular-nums">{(c.similarity * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
