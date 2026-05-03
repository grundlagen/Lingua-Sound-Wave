import { useRef, useState } from "react";
import { Languages, Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  useTranslatePassage,
  useGetLanguages,
  type TranslatedPassage,
  type TranslatedChunk,
  type TranslatedChunkAlternativesItem,
} from "@workspace/api-client-react";
import { AudioCard } from "@/components/AudioCard";
import { SimilarityMeter } from "@/components/SimilarityMeter";

const SAMPLE_PASSAGES: { label: string; sourceLanguage: string; text: string }[] = [
  {
    label: "English nursery rhyme (4 short verses)",
    sourceLanguage: "en",
    text: `Mary had a little lamb, its fleece was white as snow.
And everywhere that Mary went, the lamb was sure to go.
It followed her to school one day, which was against the rule.
It made the children laugh and play to see a lamb at school.`,
  },
  {
    label: "Hamlet's soliloquy (opening lines)",
    sourceLanguage: "en",
    text: `To be, or not to be, that is the question.
Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune,
or to take arms against a sea of troubles and by opposing end them.`,
  },
  {
    label: "Spanish fragment",
    sourceLanguage: "es",
    text: `En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo.
De los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor.`,
  },
];

export function TranslatePage() {
  const [passage, setPassage] = useState(SAMPLE_PASSAGES[0]!.text);
  const [sourceLang, setSourceLang] = useState("en");
  const [targetLang, setTargetLang] = useState("fr");
  const [candidates, setCandidates] = useState(2);
  const [result, setResult] = useState<TranslatedPassage | null>(null);
  const requestIdRef = useRef(0);
  const [staleError, setStaleError] = useState<string | null>(null);

  const { data: languages = [] } = useGetLanguages();
  const translate = useTranslatePassage();

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = passage.trim();
    if (!trimmed) return;
    const myId = ++requestIdRef.current;
    setResult(null);
    setStaleError(null);
    translate.reset();
    translate.mutate(
      {
        data: {
          passage: trimmed,
          sourceLanguage: sourceLang,
          targetLanguage: targetLang,
          candidatesPerChunk: candidates,
        },
      },
      {
        onSuccess: (data) => {
          if (myId !== requestIdRef.current) return;
          setResult(data);
        },
        onError: (err) => {
          if (myId !== requestIdRef.current) return;
          setStaleError(err instanceof Error ? err.message : String(err));
        },
      },
    );
  };

  const useSample = (i: number) => {
    const s = SAMPLE_PASSAGES[i]!;
    setPassage(s.text);
    setSourceLang(s.sourceLanguage);
  };

  const charCount = passage.length;
  const isPending = translate.isPending;

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-baseline justify-between gap-2">
              <Label htmlFor="passage">Source passage</Label>
              <span
                className={`text-xs tabular-nums ${charCount > 4000 ? "text-destructive" : "text-muted-foreground"}`}
                data-testid="char-count"
              >
                {charCount} / 4000
              </span>
            </div>
            <Textarea
              id="passage"
              value={passage}
              onChange={(e) => setPassage(e.target.value)}
              rows={8}
              className="font-mono text-sm"
              data-testid="input-passage"
              placeholder="Paste a passage to translate (semantic + homophonic)"
            />
            <div className="flex flex-wrap gap-2 pt-1">
              {SAMPLE_PASSAGES.map((s, i) => (
                <Button
                  key={s.label}
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => useSample(i)}
                  data-testid={`sample-${i}`}
                >
                  <Sparkles className="h-3 w-3 mr-1" />
                  {s.label}
                </Button>
              ))}
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Source language</Label>
              <Select value={sourceLang} onValueChange={setSourceLang}>
                <SelectTrigger data-testid="select-source-lang"><SelectValue /></SelectTrigger>
                <SelectContent className="max-h-72">
                  {languages.map((l) => (
                    <SelectItem key={l.code} value={l.code}>{l.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Target language</Label>
              <Select value={targetLang} onValueChange={setTargetLang}>
                <SelectTrigger data-testid="select-target-lang"><SelectValue /></SelectTrigger>
                <SelectContent className="max-h-72">
                  {languages.map((l) => (
                    <SelectItem key={l.code} value={l.code}>{l.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Homophonic candidates per chunk: <span className="tabular-nums">{candidates}</span></Label>
              <Slider
                min={1}
                max={4}
                step={1}
                value={[candidates]}
                onValueChange={(v) => setCandidates(v[0]!)}
                data-testid="slider-candidates"
              />
              <p className="text-xs text-muted-foreground">More candidates = better matches but slower.</p>
            </div>
          </div>

          <Button
            type="submit"
            size="lg"
            disabled={isPending || !passage.trim() || charCount > 4000 || sourceLang === targetLang}
            data-testid="button-translate"
          >
            {isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Languages className="mr-2 h-4 w-4" />}
            {isPending ? "Translating passage…" : "Translate passage"}
          </Button>
          {sourceLang === targetLang ? (
            <p className="text-xs text-amber-600">Source and target languages must differ.</p>
          ) : null}
          {isPending ? (
            <p className="text-xs text-muted-foreground">
              Splitting into chunks, generating semantic + homophonic translations, synthesizing TTS, and
              ranking by acoustic similarity. This can take 1–3 minutes for a full page.
            </p>
          ) : null}
        </form>
      </Card>

      {staleError ? (
        <Card className="p-4 border-destructive bg-destructive/5 text-sm text-destructive" data-testid="translate-error">
          {staleError}
        </Card>
      ) : null}

      {result ? <TranslationResult result={result} /> : null}
    </div>
  );
}

function TranslationResult({ result }: { result: TranslatedPassage }) {
  const failedCount = result.chunks.filter((c: TranslatedChunk) => c.error).length;
  return (
    <div className="space-y-4" data-testid="translate-result">
      <Card className="p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              {result.sourceLanguageName} → {result.targetLanguageName}
            </div>
            <div className="text-lg font-semibold">
              {result.chunks.length} chunks · {(result.elapsedMs / 1000).toFixed(1)}s
            </div>
          </div>
          <div className="space-y-1 min-w-[200px]">
            <div className="flex items-baseline gap-2">
              <div className="text-3xl font-bold tabular-nums">
                {(result.averageSimilarity * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">avg homophonic similarity</div>
            </div>
            <SimilarityMeter value={result.averageSimilarity} showLabel={false} />
          </div>
        </div>
        {(failedCount > 0 || result.chunksDropped > 0) ? (
          <div className="mt-3 text-xs text-amber-600" data-testid="translate-warnings">
            {failedCount > 0 ? `${failedCount} chunks failed. ` : ""}
            {result.chunksDropped > 0 ? `${result.chunksDropped} chunks dropped (over cap).` : ""}
          </div>
        ) : null}
      </Card>

      {result.chunks.map((chunk: TranslatedChunk) => (
        <Card key={chunk.index} className="p-5 space-y-3" data-testid={`chunk-${chunk.index}`}>
          <div className="flex items-baseline justify-between gap-3 flex-wrap">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Chunk {chunk.index + 1}
            </div>
            {chunk.error ? (
              <span className="text-xs text-destructive">⚠ {chunk.error}</span>
            ) : (
              <div className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground">acoustic match</span>
                <span className="text-lg font-semibold tabular-nums" data-testid={`chunk-${chunk.index}-sim`}>
                  {(chunk.similarity * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-1">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                Source ({result.sourceLanguageName})
              </div>
              <div className="text-sm leading-relaxed">{chunk.sourceText}</div>
            </div>
            <div className="space-y-1">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                Semantic translation ({result.targetLanguageName})
              </div>
              <div className="text-sm leading-relaxed">{chunk.semanticTranslation || "—"}</div>
            </div>
            <div className="space-y-1">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                Homophonic ({result.targetLanguageName})
              </div>
              <div className="text-sm leading-relaxed font-medium">{chunk.homophonic || "—"}</div>
              {chunk.homophonicGloss ? (
                <div className="text-xs text-muted-foreground italic">"{chunk.homophonicGloss}"</div>
              ) : null}
            </div>
          </div>

          {chunk.sourceAudio || chunk.homophonicAudio ? (
            <div className="grid gap-3 md:grid-cols-2 pt-1">
              {chunk.sourceAudio ? (
                <AudioCard
                  audio={chunk.sourceAudio}
                  label={`Source: ${chunk.sourceText.slice(0, 40)}${chunk.sourceText.length > 40 ? "…" : ""}`}
                  accent="#6366f1"
                  testId={`chunk-${chunk.index}-source-audio`}
                />
              ) : null}
              {chunk.homophonicAudio ? (
                <AudioCard
                  audio={chunk.homophonicAudio}
                  label={`Homophonic: ${chunk.homophonic.slice(0, 40)}${chunk.homophonic.length > 40 ? "…" : ""}`}
                  accent="#0ea5e9"
                  testId={`chunk-${chunk.index}-homo-audio`}
                />
              ) : null}
            </div>
          ) : null}

          {chunk.alternatives.length > 0 ? (
            <details className="text-xs" data-testid={`chunk-${chunk.index}-alternatives`}>
              <summary className="cursor-pointer text-muted-foreground">
                {chunk.alternatives.length} other homophonic candidates
              </summary>
              <ul className="mt-2 space-y-1 pl-4">
                {chunk.alternatives.map((alt: TranslatedChunkAlternativesItem, i: number) => (
                  <li key={i} className="flex items-baseline gap-2">
                    <span className="font-medium">{alt.phrase}</span>
                    {alt.gloss ? <span className="text-muted-foreground italic">— "{alt.gloss}"</span> : null}
                    <span className="ml-auto tabular-nums text-muted-foreground">
                      {(alt.similarity * 100).toFixed(0)}%
                    </span>
                  </li>
                ))}
              </ul>
            </details>
          ) : null}
        </Card>
      ))}
    </div>
  );
}
