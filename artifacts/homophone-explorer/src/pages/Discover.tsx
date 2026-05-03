import { useRef, useState } from "react";
import { Search, Loader2, Sparkles, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Card } from "@/components/ui/card";
import {
  useDiscoverHomophones,
  useGetLanguages,
  useGetFeaturedPairs,
} from "@workspace/api-client-react";
import type { DiscoverResponse } from "@workspace/api-client-react";
import { LanguageMultiselect } from "@/components/LanguageMultiselect";
import { AudioCard } from "@/components/AudioCard";
import { MatchCard } from "@/components/MatchCard";

export function DiscoverPage() {
  const [phrase, setPhrase] = useState("knee how");
  const [sourceLang, setSourceLang] = useState("en");
  const [targets, setTargets] = useState<string[]>([]);
  const [minSim, setMinSim] = useState(0.55);
  const [count, setCount] = useState(24);
  const [result, setResult] = useState<DiscoverResponse | null>(null);
  const requestIdRef = useRef(0);

  const { data: languages = [] } = useGetLanguages();
  const { data: featured = [] } = useGetFeaturedPairs();
  const discover = useDiscoverHomophones();

  const runSearch = (p: string, sl: string, useTargets: boolean) => {
    const trimmed = p.trim();
    if (!trimmed) return;
    const myId = ++requestIdRef.current;
    setResult(null);
    discover.mutate(
      {
        data: {
          phrase: trimmed,
          sourceLanguage: sl,
          targetLanguages: useTargets && targets.length > 0 ? targets : undefined,
          minSimilarity: minSim,
          candidateCount: count,
        },
      },
      {
        onSuccess: (data) => {
          // Ignore stale responses if a newer search has started.
          if (myId !== requestIdRef.current) return;
          setResult(data);
        },
      },
    );
  };

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runSearch(phrase, sourceLang, true);
  };

  const tryFeatured = (sourcePhrase: string, sourceLanguage: string) => {
    setPhrase(sourcePhrase);
    setSourceLang(sourceLanguage);
    runSearch(sourcePhrase, sourceLanguage, false);
  };

  return (
    <div className="space-y-8">
      <Card className="p-6">
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="grid gap-4 md:grid-cols-[1fr_220px]">
            <div className="space-y-2">
              <Label htmlFor="phrase">Source phrase</Label>
              <Input
                id="phrase"
                value={phrase}
                onChange={(e) => setPhrase(e.target.value)}
                placeholder="Type a word or multi-word phrase…"
                className="text-lg"
                data-testid="input-phrase"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="source-lang">Source language</Label>
              <Select value={sourceLang} onValueChange={setSourceLang}>
                <SelectTrigger id="source-lang" data-testid="select-source-lang">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-72">
                  {languages.map((l) => (
                    <SelectItem key={l.code} value={l.code} data-testid={`source-opt-${l.code}`}>
                      {l.name} <span className="text-muted-foreground ml-1">{l.nativeName}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Target languages</Label>
              <LanguageMultiselect
                languages={languages}
                selected={targets}
                onChange={setTargets}
                exclude={sourceLang}
              />
            </div>
            <div className="space-y-2">
              <Label>
                Min similarity: <span className="tabular-nums">{(minSim * 100).toFixed(0)}%</span>
              </Label>
              <Slider
                min={0.3}
                max={0.95}
                step={0.05}
                value={[minSim]}
                onValueChange={(v) => setMinSim(v[0]!)}
                data-testid="slider-min-sim"
              />
            </div>
            <div className="space-y-2">
              <Label>
                Candidates to evaluate: <span className="tabular-nums">{count}</span>
              </Label>
              <Slider
                min={8}
                max={48}
                step={4}
                value={[count]}
                onValueChange={(v) => setCount(v[0]!)}
                data-testid="slider-count"
              />
            </div>
          </div>
          <div className="flex items-center justify-between gap-4 pt-2">
            <p className="text-xs text-muted-foreground">
              We synthesize real TTS audio for each candidate and compare with MFCC + Dynamic Time
              Warping. Search takes ~30–90s.
            </p>
            <Button
              type="submit"
              size="lg"
              disabled={discover.isPending || !phrase.trim()}
              data-testid="button-search"
            >
              {discover.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Search className="mr-2 h-4 w-4" />
              )}
              {discover.isPending ? "Searching…" : "Find homophones"}
            </Button>
          </div>
        </form>
      </Card>

      {discover.isPending ? (
        <Card className="p-8 text-center space-y-3" data-testid="loading-state">
          <Loader2 className="mx-auto h-8 w-8 animate-spin text-primary" />
          <div className="font-medium">Synthesizing speech across languages…</div>
          <div className="text-sm text-muted-foreground">
            Generating candidates → calling TTS for each → measuring MFCC + DTW distance
          </div>
        </Card>
      ) : null}

      {discover.error ? (
        <Card className="p-4 border-destructive bg-destructive/5 text-sm text-destructive" data-testid="error-state">
          {discover.error instanceof Error ? discover.error.message : String(discover.error)}
        </Card>
      ) : null}

      {result ? <ResultsView result={result} /> : null}

      {!result && !discover.isPending ? (
        <Card className="p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            <h2 className="font-semibold">Featured pairs</h2>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {featured.map((p, i) => (
              <button
                key={i}
                onClick={() => tryFeatured(p.sourcePhrase, p.sourceLanguage)}
                className="text-left rounded-lg border p-3 hover-elevate"
                data-testid={`featured-${i}`}
              >
                <div className="text-xs text-muted-foreground uppercase">{p.sourceLanguage}</div>
                <div className="font-medium">{p.sourcePhrase}</div>
                <div className="flex items-center text-xs text-muted-foreground my-1">
                  <ArrowRight className="h-3 w-3 mx-1" /> <span className="uppercase">{p.matchLanguage}</span>
                </div>
                <div className="font-medium">{p.matchPhrase}</div>
                <div className="text-xs text-muted-foreground mt-1 italic">"{p.description}"</div>
              </button>
            ))}
          </div>
        </Card>
      ) : null}
    </div>
  );
}

function ResultsView({ result }: { result: DiscoverResponse }) {
  return (
    <div className="space-y-6">
      <Card className="p-5 space-y-3" data-testid="source-card">
        <div className="flex items-baseline justify-between gap-3 flex-wrap">
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Source</div>
            <h2 className="text-2xl font-bold">{result.sourcePhrase}</h2>
            <div className="text-sm text-muted-foreground">
              {result.sourceLanguageName} · "{result.sourceMeaning}"
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            {result.matches.length} match{result.matches.length === 1 ? "" : "es"} · {result.candidatesEvaluated} candidates · {(result.elapsedMs / 1000).toFixed(1)}s
          </div>
        </div>
        <AudioCard
          audio={result.sourceAudio}
          label={result.sourcePhrase}
          sublabel={result.sourceLanguageName}
          accent="#6366f1"
          testId="source-audio"
        />
      </Card>

      {result.matches.length === 0 ? (
        <Card className="p-6 text-center text-sm text-muted-foreground" data-testid="no-matches">
          No matches above your similarity threshold. Try lowering it or increasing candidate count.
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {result.matches.map((m, i) => (
            <MatchCard
              key={`${m.languageCode}-${i}`}
              source={{
                phrase: result.sourcePhrase,
                language: result.sourceLanguage,
                languageName: result.sourceLanguageName,
                meaning: result.sourceMeaning,
                audio: result.sourceAudio,
              }}
              match={m}
            />
          ))}
        </div>
      )}
    </div>
  );
}
