import { Trash2, BookmarkCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useGetSavedPairs, useDeleteSavedPair, getGetSavedPairsQueryKey } from "@workspace/api-client-react";
import { useQueryClient } from "@tanstack/react-query";
import { SimilarityMeter } from "@/components/SimilarityMeter";

export function SavedPage() {
  const { data: pairs = [], isLoading } = useGetSavedPairs();
  const qc = useQueryClient();
  const del = useDeleteSavedPair({
    mutation: {
      onSuccess: () => qc.invalidateQueries({ queryKey: getGetSavedPairsQueryKey() }),
    },
  });

  if (isLoading) return <Card className="p-6 text-sm text-muted-foreground">Loading…</Card>;

  if (pairs.length === 0)
    return (
      <Card className="p-10 text-center space-y-2" data-testid="empty-saved">
        <BookmarkCheck className="mx-auto h-8 w-8 text-muted-foreground" />
        <div className="font-medium">No saved pairs yet</div>
        <div className="text-sm text-muted-foreground">
          Bookmark interesting matches from the Discover tab and they'll show up here.
        </div>
      </Card>
    );

  return (
    <div className="grid gap-3" data-testid="saved-list">
      {pairs.map((p) => (
        <Card key={p.id} className="p-4 space-y-3" data-testid={`saved-${p.id}`}>
          <div className="flex items-start justify-between gap-3">
            <div className="grid gap-2 sm:grid-cols-2 flex-1">
              <div>
                <Badge variant="outline" className="text-xs">{p.sourceLanguage}</Badge>
                <div className="font-semibold mt-1">{p.sourcePhrase}</div>
                <div className="text-xs text-muted-foreground">"{p.sourceMeaning}"</div>
              </div>
              <div>
                <Badge variant="secondary" className="text-xs">{p.matchLanguage}</Badge>
                <div className="font-semibold mt-1">{p.matchPhrase}</div>
                <div className="text-xs text-muted-foreground">"{p.matchMeaning}"</div>
              </div>
            </div>
            <Button
              size="icon"
              variant="ghost"
              onClick={() => del.mutate({ id: p.id })}
              disabled={del.isPending}
              aria-label={`Delete saved pair ${p.sourcePhrase} and ${p.matchPhrase}`}
              data-testid={`delete-${p.id}`}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
          <SimilarityMeter value={p.similarity} />
          {p.notes ? <p className="text-xs text-muted-foreground italic">{p.notes}</p> : null}
        </Card>
      ))}
    </div>
  );
}
