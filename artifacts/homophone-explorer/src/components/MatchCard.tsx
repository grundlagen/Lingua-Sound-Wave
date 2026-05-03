import { useState } from "react";
import { Bookmark, BookmarkCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AudioCard } from "./AudioCard";
import { SimilarityMeter } from "./SimilarityMeter";
import type { AcousticMatch, AudioPayload } from "@workspace/api-client-react";
import { useSavePair, getGetSavedPairsQueryKey } from "@workspace/api-client-react";
import { useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";

interface Props {
  source: { phrase: string; language: string; languageName: string; meaning: string; audio: AudioPayload };
  match: AcousticMatch;
  onSaved?: () => void;
}

export function MatchCard({ source, match, onSaved }: Props) {
  const [saved, setSaved] = useState(false);
  const { toast } = useToast();
  const qc = useQueryClient();
  const save = useSavePair({
    mutation: {
      onSuccess: () => qc.invalidateQueries({ queryKey: getGetSavedPairsQueryKey() }),
    },
  });

  const onSave = async () => {
    try {
      await save.mutateAsync({
        data: {
          sourcePhrase: source.phrase,
          sourceLanguage: source.language,
          sourceMeaning: source.meaning,
          matchPhrase: match.phrase,
          matchLanguage: match.languageCode,
          matchMeaning: match.meaning,
          similarity: match.similarity,
          notes: match.notes,
        },
      });
      setSaved(true);
      toast({ title: "Saved", description: `${source.phrase} ↔ ${match.phrase}` });
      onSaved?.();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      toast({ title: "Save failed", description: msg, variant: "destructive" });
    }
  };

  return (
    <div className="rounded-xl border bg-card p-4 space-y-3 hover-elevate" data-testid={`match-card-${match.languageCode}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="secondary" className="text-xs">
              {match.language}
            </Badge>
            <h3 className="text-lg font-semibold truncate" data-testid={`match-phrase-${match.languageCode}`}>
              {match.phrase}
            </h3>
          </div>
          <p className="text-sm text-muted-foreground mt-1">"{match.meaning}"</p>
          {match.notes ? (
            <p className="text-xs text-muted-foreground mt-1 italic">{match.notes}</p>
          ) : null}
        </div>
        <Button
          size="icon"
          variant="outline"
          onClick={onSave}
          disabled={saved || save.isPending}
          aria-label={saved ? `Saved ${match.phrase}` : `Save ${match.phrase}`}
          data-testid={`save-${match.languageCode}`}
        >
          {saved ? <BookmarkCheck className="h-4 w-4 text-emerald-600" /> : <Bookmark className="h-4 w-4" />}
        </Button>
      </div>
      <SimilarityMeter value={match.similarity} />
      <AudioCard
        audio={match.audio}
        label={match.phrase}
        sublabel={match.language}
        accent="#0ea5e9"
        testId={`match-audio-${match.languageCode}`}
      />
    </div>
  );
}
