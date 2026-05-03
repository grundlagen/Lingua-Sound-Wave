import { useState } from "react";
import { Loader2, Pickaxe, StopCircle, Database, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  useListReservoirPairs,
  useGetReservoirStats,
  useGetReservoirMiningStatus,
  useStartReservoirMining,
  useCancelReservoirMining,
  getListReservoirPairsQueryKey,
  getGetReservoirStatsQueryKey,
  getGetReservoirMiningStatusQueryKey,
} from "@workspace/api-client-react";
import { useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { ReservoirPairCard } from "@/components/ReservoirPairCard";
import { TierBadge } from "@/components/TierBadge";

export function ReservoirPage() {
  const [tier, setTier] = useState<string>("all");
  const [minSim, setMinSim] = useState<number[]>([0.55]);
  const [search, setSearch] = useState("");
  const [maxSeeds, setMaxSeeds] = useState<number[]>([8]);

  const qc = useQueryClient();
  const { toast } = useToast();

  const params: { tier?: "S" | "A" | "B"; minSim?: number; search?: string; limit?: number } = { limit: 200 };
  if (tier === "S" || tier === "A" || tier === "B") params.tier = tier;
  if (minSim[0] && minSim[0] > 0) params.minSim = minSim[0];
  if (search.trim()) params.search = search.trim();

  const { data: pairs = [], isLoading } = useListReservoirPairs(params);
  const { data: stats } = useGetReservoirStats();
  const { data: status } = useGetReservoirMiningStatus({
    query: { queryKey: getGetReservoirMiningStatusQueryKey(), refetchInterval: 2000 },
  });

  const start = useStartReservoirMining({
    mutation: {
      onSuccess: () => {
        toast({ title: "Mining started", description: "Generating EN↔FR pairs in the background…" });
        qc.invalidateQueries({ queryKey: getGetReservoirMiningStatusQueryKey() });
      },
      onError: (err) => {
        toast({ title: "Could not start", description: String(err), variant: "destructive" });
      },
    },
  });
  const cancel = useCancelReservoirMining({
    mutation: {
      onSuccess: () => {
        qc.invalidateQueries({ queryKey: getGetReservoirMiningStatusQueryKey() });
        toast({ title: "Cancellation requested" });
      },
    },
  });

  const isRunning = status?.activeJobId !== null && status?.activeJobId !== undefined;
  const job = status?.job;
  const total = stats?.total ?? 0;
  const target = stats?.target ?? 2500;
  const progress = Math.min(100, (total / target) * 100);

  // Auto-refresh table while mining is running
  if (isRunning) {
    qc.invalidateQueries({ queryKey: getListReservoirPairsQueryKey(), refetchType: "none" });
    qc.invalidateQueries({ queryKey: getGetReservoirStatsQueryKey(), refetchType: "none" });
  }

  return (
    <div className="space-y-6">
      <Card className="p-6 space-y-4">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-500 grid place-items-center">
              <Database className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-bold">EN ↔ FR Reservoir</h2>
              <p className="text-sm text-muted-foreground">
                Persistent corpus of mined English↔French sound-alike pairs, graded S / A / B.
              </p>
            </div>
          </div>
          <div className="flex gap-2 items-center">
            <TierBadge tier="S" size="sm" />
            <span className="text-xs tabular-nums" data-testid="stat-tier-S">{stats?.tierCounts?.S ?? 0}</span>
            <TierBadge tier="A" size="sm" />
            <span className="text-xs tabular-nums" data-testid="stat-tier-A">{stats?.tierCounts?.A ?? 0}</span>
            <TierBadge tier="B" size="sm" />
            <span className="text-xs tabular-nums" data-testid="stat-tier-B">{stats?.tierCounts?.B ?? 0}</span>
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Reservoir progress toward target</span>
            <span className="tabular-nums font-medium" data-testid="stat-total">
              {total} / {target} pairs
            </span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>
      </Card>

      <Card className="p-4 space-y-4">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <Pickaxe className="h-4 w-4 text-muted-foreground" />
            <span className="font-medium text-sm">Mining controls</span>
          </div>
          <div className="flex items-center gap-2">
            {isRunning ? (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => cancel.mutate()}
                disabled={cancel.isPending}
                data-testid="btn-cancel-mining"
              >
                <StopCircle className="h-4 w-4 mr-1" /> Cancel
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={() => start.mutate({ data: { maxSeeds: maxSeeds[0]! } })}
                disabled={start.isPending}
                data-testid="btn-start-mining"
              >
                {start.isPending ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Pickaxe className="h-4 w-4 mr-1" />}
                Mine more pairs
              </Button>
            )}
          </div>
        </div>
        <div className="grid sm:grid-cols-[1fr,auto] gap-4 items-end">
          <div className="space-y-2">
            <Label className="text-xs">Seeds per run: <span className="font-mono">{maxSeeds[0]}</span></Label>
            <Slider value={maxSeeds} min={2} max={40} step={1} onValueChange={setMaxSeeds} disabled={isRunning} />
            <p className="text-[11px] text-muted-foreground">
              Each seed generates ~4 candidates. Already-seen pairs are skipped (idempotent).
            </p>
          </div>
        </div>
        {job ? (
          <div className="rounded-lg border bg-muted/30 p-3 space-y-2 text-xs">
            <div className="flex items-center gap-2 flex-wrap">
              <Badge variant={job.status === "running" ? "default" : "outline"}>{job.status}</Badge>
              {isRunning ? (
                <span className="flex items-center gap-1 text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  current seed: <span className="font-mono">{job.currentSeed || "…"}</span>
                </span>
              ) : (
                <span className="text-muted-foreground">last job #{job.id}</span>
              )}
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 tabular-nums">
              <Stat label="Attempted" value={job.totalsAttempted} />
              <Stat label="Inserted" value={job.totalsInserted} />
              <Stat label="Skipped" value={job.totalsSkipped} />
              <Stat label="Failed" value={job.totalsFailed} />
            </div>
            <div className="flex items-center gap-3 text-xs flex-wrap">
              <span className="text-muted-foreground">tiers this run:</span>
              <span className="flex items-center gap-1"><TierBadge tier="S" size="sm" /> {job.tierCounts.S}</span>
              <span className="flex items-center gap-1"><TierBadge tier="A" size="sm" /> {job.tierCounts.A}</span>
              <span className="flex items-center gap-1"><TierBadge tier="B" size="sm" /> {job.tierCounts.B}</span>
            </div>
            {job.lastError ? (
              <p className="text-destructive text-xs">{job.lastError}</p>
            ) : null}
          </div>
        ) : null}
      </Card>

      <Card className="p-4 space-y-4">
        <div className="grid sm:grid-cols-[1fr,160px,1fr] gap-3 items-end">
          <div className="space-y-2">
            <Label className="text-xs">Search</Label>
            <div className="relative">
              <Search className="h-4 w-4 absolute left-2 top-2.5 text-muted-foreground" />
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="EN or FR substring…"
                className="pl-8"
                data-testid="input-search"
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label className="text-xs">Tier</Label>
            <Select value={tier} onValueChange={setTier}>
              <SelectTrigger data-testid="select-tier">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All tiers</SelectItem>
                <SelectItem value="S">S only</SelectItem>
                <SelectItem value="A">A only</SelectItem>
                <SelectItem value="B">B only</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label className="text-xs">
              Min similarity: <span className="font-mono">{Math.round((minSim[0] ?? 0) * 100)}%</span>
            </Label>
            <Slider value={minSim} min={0.5} max={1} step={0.01} onValueChange={setMinSim} />
          </div>
        </div>
      </Card>

      <div className="space-y-3">
        {isLoading ? (
          <div className="text-center py-12 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin mx-auto" />
          </div>
        ) : pairs.length === 0 ? (
          <Card className="p-12 text-center text-muted-foreground">
            <Database className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="font-medium">No pairs match these filters yet.</p>
            <p className="text-sm mt-1">Click <strong>Mine more pairs</strong> above to start populating the reservoir.</p>
          </Card>
        ) : (
          pairs.map((p) => <ReservoirPairCard key={p.id} pair={p} />)
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded border bg-card p-2">
      <div className="text-[10px] uppercase text-muted-foreground tracking-wider">{label}</div>
      <div className="font-semibold">{value}</div>
    </div>
  );
}
