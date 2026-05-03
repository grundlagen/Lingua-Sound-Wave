import { useState } from "react";
import { Trash2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { TierBadge } from "./TierBadge";
import { ComponentScores } from "./ComponentScores";
import { SimilarityMeter } from "./SimilarityMeter";
import {
  useDeleteReservoirPair,
  getListReservoirPairsQueryKey,
  getGetReservoirStatsQueryKey,
  type ReservoirPair,
} from "@workspace/api-client-react";
import { useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";

interface Props {
  pair: ReservoirPair;
  onDeleted?: () => void;
}

function CoherenceDots({ value, label }: { value: number; label: string }) {
  return (
    <div className="flex items-center gap-1 text-[11px] text-muted-foreground" title={`${label} coherence ${value}/3`}>
      <span className="font-mono uppercase">{label}</span>
      <div className="flex gap-0.5">
        {[1, 2, 3].map((i) => (
          <span
            key={i}
            className={`h-1.5 w-1.5 rounded-full ${i <= value ? "bg-emerald-500" : "bg-muted"}`}
          />
        ))}
      </div>
    </div>
  );
}

export function ReservoirPairCard({ pair, onDeleted }: Props) {
  const [confirming, setConfirming] = useState(false);
  const qc = useQueryClient();
  const { toast } = useToast();
  const del = useDeleteReservoirPair({
    mutation: {
      onSuccess: () => {
        qc.invalidateQueries({ queryKey: getListReservoirPairsQueryKey() });
        qc.invalidateQueries({ queryKey: getGetReservoirStatsQueryKey() });
        toast({ title: "Removed", description: `${pair.enPhrase} ↔ ${pair.frPhrase}` });
        onDeleted?.();
      },
      onError: (err) => {
        toast({ title: "Delete failed", description: String(err), variant: "destructive" });
      },
    },
  });

  return (
    <div
      className="rounded-xl border bg-card p-4 space-y-3 hover-elevate"
      data-testid={`reservoir-pair-${pair.id}`}
    >
      <div className="flex items-start gap-3">
        <TierBadge tier={pair.tier} />
        <div className="flex-1 min-w-0 space-y-2">
          <div className="grid sm:grid-cols-2 gap-3">
            <div>
              <Badge variant="secondary" className="text-[10px] mb-1">EN</Badge>
              <div className="font-semibold leading-tight" data-testid={`en-phrase-${pair.id}`}>
                {pair.enPhrase}
              </div>
              {pair.enGloss ? <p className="text-xs text-muted-foreground mt-0.5 italic">"{pair.enGloss}"</p> : null}
            </div>
            <div>
              <Badge variant="secondary" className="text-[10px] mb-1">FR</Badge>
              <div className="font-semibold leading-tight" data-testid={`fr-phrase-${pair.id}`}>
                {pair.frPhrase}
              </div>
              {pair.frGloss ? <p className="text-xs text-muted-foreground mt-0.5 italic">"{pair.frGloss}"</p> : null}
            </div>
          </div>
          <SimilarityMeter value={pair.similarity} />
          {pair.componentScores && pair.componentScores.length > 0 ? (
            <ComponentScores
              components={pair.componentScores.map((c) => ({ id: c.id, label: c.label, similarity: c.similarity, distance: c.distance }))}
              size="sm"
            />
          ) : null}
          <div className="flex items-center gap-3 flex-wrap">
            <CoherenceDots value={pair.enCoherence} label="EN" />
            <CoherenceDots value={pair.frCoherence} label="FR" />
            {pair.seed ? (
              <span className="text-[11px] text-muted-foreground">
                seed: <span className="font-mono">{pair.seed}</span>
              </span>
            ) : null}
            <span className="text-[11px] text-muted-foreground capitalize">{pair.source}</span>
          </div>
          {pair.rationale ? (
            <p className="text-[11px] text-muted-foreground border-l-2 border-muted pl-2">
              {pair.rationale}
            </p>
          ) : null}
        </div>
        <div>
          {confirming ? (
            <div className="flex gap-1">
              <Button
                size="sm"
                variant="destructive"
                onClick={() => del.mutate({ id: pair.id })}
                disabled={del.isPending}
                data-testid={`confirm-delete-${pair.id}`}
              >
                {del.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : "Delete"}
              </Button>
              <Button size="sm" variant="outline" onClick={() => setConfirming(false)}>
                Cancel
              </Button>
            </div>
          ) : (
            <Button
              size="icon"
              variant="ghost"
              onClick={() => setConfirming(true)}
              aria-label="Remove from reservoir"
              data-testid={`delete-${pair.id}`}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
