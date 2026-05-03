interface Props {
  tier: "S" | "A" | "B" | string;
  size?: "sm" | "md";
}

const STYLE: Record<string, string> = {
  S: "bg-gradient-to-br from-amber-400 to-amber-600 text-amber-50 border-amber-700",
  A: "bg-gradient-to-br from-emerald-500 to-emerald-700 text-emerald-50 border-emerald-800",
  B: "bg-gradient-to-br from-slate-400 to-slate-600 text-slate-50 border-slate-700",
};

export function TierBadge({ tier, size = "md" }: Props) {
  const cls = STYLE[tier] ?? STYLE["B"]!;
  const dim = size === "sm" ? "h-6 w-6 text-[11px]" : "h-7 w-7 text-xs";
  return (
    <div
      className={`inline-grid place-items-center rounded-md border font-bold tabular-nums ${dim} ${cls}`}
      title={`Tier ${tier}`}
      data-testid={`tier-badge-${tier}`}
    >
      {tier}
    </div>
  );
}
