import { useState } from "react";
import { GitCompare, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useComparePhrases, useGetLanguages } from "@workspace/api-client-react";
import { AudioCard } from "@/components/AudioCard";
import { SimilarityMeter } from "@/components/SimilarityMeter";
import { MethodSelector } from "@/components/MethodSelector";

export function ComparePage() {
  const [p1, setP1] = useState("knee how");
  const [l1, setL1] = useState("en");
  const [p2, setP2] = useState("你好");
  const [l2, setL2] = useState("zh");
  const [scoringMethod, setScoringMethod] = useState("mfcc-dtw");

  const { data: languages = [] } = useGetLanguages();
  const compare = useComparePhrases();

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    compare.mutate({
      data: { phrase1: p1, language1: l1, phrase2: p2, language2: l2, scoringMethod },
    });
  };

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Phrase A</Label>
              <Input value={p1} onChange={(e) => setP1(e.target.value)} data-testid="input-p1" />
              <Select value={l1} onValueChange={setL1}>
                <SelectTrigger data-testid="select-l1"><SelectValue /></SelectTrigger>
                <SelectContent className="max-h-72">
                  {languages.map((l) => (
                    <SelectItem key={l.code} value={l.code}>{l.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Phrase B</Label>
              <Input value={p2} onChange={(e) => setP2(e.target.value)} data-testid="input-p2" />
              <Select value={l2} onValueChange={setL2}>
                <SelectTrigger data-testid="select-l2"><SelectValue /></SelectTrigger>
                <SelectContent className="max-h-72">
                  {languages.map((l) => (
                    <SelectItem key={l.code} value={l.code}>{l.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <MethodSelector value={scoringMethod} onChange={setScoringMethod} testId="select-method-compare" />
          <Button
            type="submit"
            size="lg"
            disabled={compare.isPending || !p1.trim() || !p2.trim()}
            onClick={(e) => {
              if (!p1.trim() || !p2.trim()) e.preventDefault();
            }}
            data-testid="button-compare"
          >
            {compare.isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <GitCompare className="mr-2 h-4 w-4" />}
            {compare.isPending ? "Comparing…" : "Compare audio"}
          </Button>
        </form>
      </Card>

      {compare.error ? (
        <Card className="p-4 border-destructive bg-destructive/5 text-sm text-destructive" data-testid="compare-error">
          {compare.error instanceof Error ? compare.error.message : String(compare.error)}
        </Card>
      ) : null}

      {compare.data ? (
        <Card className="p-5 space-y-4" data-testid="compare-result">
          <div className="space-y-2">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Acoustic similarity
            </div>
            <div className="flex items-baseline gap-3">
              <div className="text-4xl font-bold tabular-nums">
                {(compare.data.similarity * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">{compare.data.verdict}</div>
            </div>
            <SimilarityMeter value={compare.data.similarity} showLabel={false} />
            <div className="text-xs text-muted-foreground flex flex-wrap gap-x-3">
              <span>distance: <span className="tabular-nums">{compare.data.dtwDistance.toFixed(4)}</span></span>
              <span>· judged by <span className="font-medium">{compare.data.scoringMethodLabel}</span></span>
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <AudioCard audio={compare.data.audio1} label={compare.data.phrase1} accent="#6366f1" testId="audio-1" />
            <AudioCard audio={compare.data.audio2} label={compare.data.phrase2} accent="#0ea5e9" testId="audio-2" />
          </div>
        </Card>
      ) : null}
    </div>
  );
}
