import { useGetScoringMethods, type ScoringMethodInfo } from "@workspace/api-client-react";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface Props {
  value: string;
  onChange: (id: string) => void;
  testId?: string;
}

const STATUS_BADGE: Record<ScoringMethodInfo["status"], { label: string; cls: string }> = {
  ready: { label: "ready", cls: "bg-emerald-500/10 text-emerald-600 border-emerald-500/30" },
  lazy: { label: "loads on demand", cls: "bg-amber-500/10 text-amber-700 border-amber-500/30" },
  error: { label: "unavailable", cls: "bg-destructive/10 text-destructive border-destructive/30" },
};

export function MethodSelector({ value, onChange, testId }: Props) {
  const { data: methods = [] } = useGetScoringMethods();
  const selected = methods.find((m) => m.id === value);
  return (
    <div className="space-y-2">
      <Label>Acoustic judge (scoring method)</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger data-testid={testId ?? "select-method"}>
          <SelectValue placeholder="Choose a scoring method" />
        </SelectTrigger>
        <SelectContent className="max-w-[480px]">
          {methods.map((m) => {
            const badge = STATUS_BADGE[m.status];
            return (
              <SelectItem key={m.id} value={m.id} data-testid={`method-opt-${m.id}`}>
                <div className="flex items-center gap-2">
                  <span className="font-medium">{m.label}</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded border ${badge.cls}`}>{badge.label}</span>
                </div>
              </SelectItem>
            );
          })}
        </SelectContent>
      </Select>
      {selected ? (
        <p className="text-xs text-muted-foreground leading-relaxed" data-testid="method-description">
          {selected.description}
          {selected.statusDetail ? (
            <span className="block mt-0.5 italic opacity-70">{selected.statusDetail}</span>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}
