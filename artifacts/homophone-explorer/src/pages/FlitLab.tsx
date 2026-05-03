import { useState } from "react";
import { Loader2, Sparkles, FlaskConical, Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  useRunFlit,
  useListReservoirPairs,
  type FlitResponse,
  type FlitCandidate,
} from "@workspace/api-client-react";
import { SimilarityMeter } from "@/components/SimilarityMeter";
import { ComponentScores } from "@/components/ComponentScores";
import { ReservoirPairCard } from "@/components/ReservoirPairCard";

const SAMPLE_INPUTS: { label: string; lang: "en" | "fr"; text: string }[] = [
  { label: "EN — to be or not to be", lang: "en", text: "to be or not to be" },
  { label: "EN — open the door slowly", lang: "en", text: "open the door slowly" },
  { label: "EN — the cat is sleeping", lang: "en", text: "the cat is sleeping" },
  { label: "FR — il était une fois", lang: "fr", text: "il était une fois" },
  { label: "FR — au clair de la lune", lang: "fr", text: "au clair de la lune" },
  { label: "FR — viens chez moi", lang: "fr", text: "viens chez moi" },
];

export function FlitLabPage() {
  const [text, setText] = useState("to be or not to be");
  const [lang, setLang] = useState<"en" | "fr">("en");
  const [N, setN] = useState<number[]>([6]);
  const [M, setM] = useState<number[]>([4]);
  const [topK, setTopK] = useState<number[]>([5]);
  const [result, setResult] = useState<FlitResponse | null>(null);

  const flit = useRunFlit({
    mutation: {
      onSuccess: (data) => setResult(data),
    },
  });

  // Pull a few S-tier examples in the same direction to display as live in-context examples.
  const { data: examples = [] } = useListReservoirPairs({ tier: "S", limit: 3 });

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    flit.mutate({
      data: { text: text.trim(), language: lang, inputParaphrases: N[0]!, targetRenderings: M[0]!, topK: topK[0]! },
    });
  };

  return (
    <div className="space-y-6">
      <Card className="p-6 space-y-4">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-fuchsia-500 to-rose-500 grid place-items-center">
            <FlaskConical className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold">Flit Lab</h2>
            <p className="text-sm text-muted-foreground">
              Paraphrases your input on both sides, scores every cross-product pair acoustically,
              and verifies the meaning is preserved.
            </p>
          </div>
        </div>

        <form onSubmit={onSubmit} className="space-y-4">
          <div className="grid md:grid-cols-[1fr,200px] gap-3">
            <div className="space-y-2">
              <Label>Input text</Label>
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={3}
                data-testid="input-flit-text"
              />
            </div>
            <div className="space-y-2">
              <Label>Source language</Label>
              <Select value={lang} onValueChange={(v) => setLang(v as "en" | "fr")}>
                <SelectTrigger data-testid="select-flit-lang"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="en">English (→ French)</SelectItem>
                  <SelectItem value="fr">French (→ English)</SelectItem>
                </SelectContent>
              </Select>
              <div className="flex flex-wrap gap-1 pt-1">
                {SAMPLE_INPUTS.map((s) => (
                  <Button
                    key={s.label}
                    type="button"
                    size="sm"
                    variant="outline"
                    className="h-7 text-[11px]"
                    onClick={() => {
                      setText(s.text);
                      setLang(s.lang);
                    }}
                  >
                    {s.label}
                  </Button>
                ))}
              </div>
            </div>
          </div>

          <div className="grid sm:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label className="text-xs">Input paraphrases (N): <span className="font-mono">{N[0]}</span></Label>
              <Slider value={N} min={1} max={8} step={1} onValueChange={setN} />
            </div>
            <div className="space-y-2">
              <Label className="text-xs">Target renderings each (M): <span className="font-mono">{M[0]}</span></Label>
              <Slider value={M} min={1} max={6} step={1} onValueChange={setM} />
            </div>
            <div className="space-y-2">
              <Label className="text-xs">Top-K to verify: <span className="font-mono">{topK[0]}</span></Label>
              <Slider value={topK} min={1} max={8} step={1} onValueChange={setTopK} />
            </div>
          </div>

          <p className="text-[11px] text-muted-foreground">
            Will score up to <span className="font-mono">{(N[0] ?? 1) * (M[0] ?? 1)}</span> cross-product pairs in parallel.
          </p>

          <Button type="submit" disabled={flit.isPending || !text.trim()} data-testid="btn-flit-run">
            {flit.isPending ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Sparkles className="h-4 w-4 mr-2" />}
            Run Flit
          </Button>
        </form>
      </Card>

      {examples.length > 0 ? (
        <Card className="p-4 space-y-3">
          <div className="flex items-center gap-2 text-sm">
            <Badge variant="outline">Live in-context examples</Badge>
            <span className="text-xs text-muted-foreground">
              S-tier pairs from the reservoir, for reference.
            </span>
          </div>
          <div className="grid gap-3">
            {examples.map((p) => <ReservoirPairCard key={p.id} pair={p} />)}
          </div>
        </Card>
      ) : null}

      {flit.error ? (
        <Card className="p-4 border-destructive/40 bg-destructive/5 text-sm text-destructive">
          {String(flit.error)}
        </Card>
      ) : null}

      {result ? <FlitResultView result={result} /> : null}
    </div>
  );
}

function FlitResultView({ result }: { result: FlitResponse }) {
  const inLabel = result.inputLanguage === "en" ? "EN" : "FR";
  const outLabel = result.targetLanguage === "en" ? "EN" : "FR";

  return (
    <div className="space-y-4">
      <Card className="p-4 space-y-3">
        <div className="flex items-center gap-2 flex-wrap">
          <Badge>{inLabel} → {outLabel}</Badge>
          <span className="text-xs text-muted-foreground">
            Scored {result.attempted} pairs · {(result.elapsedMs / 1000).toFixed(1)}s
          </span>
        </div>
        <div>
          <Label className="text-xs text-muted-foreground">Input meaning</Label>
          <p className="text-sm italic">"{result.inputMeaning}"</p>
        </div>
        <div>
          <Label className="text-xs text-muted-foreground">Paraphrases ({result.inputParaphrases.length})</Label>
          <ul className="text-sm space-y-1 mt-1">
            {result.inputParaphrases.map((p, i) => (
              <li key={i} className="border-l-2 border-muted pl-2">
                <span className="font-medium">{p.text}</span>
                {p.gloss && p.gloss !== p.text ? <span className="text-muted-foreground italic ml-2 text-xs">— {p.gloss}</span> : null}
              </li>
            ))}
          </ul>
        </div>
      </Card>

      {result.best ? (
        <Card className="p-4 border-emerald-500/40 bg-emerald-500/5 space-y-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-emerald-600" />
            <h3 className="font-semibold">Best candidate</h3>
            {result.best.semanticOK ? (
              <Badge className="bg-emerald-500/20 text-emerald-700 border-emerald-500/30">
                <Check className="h-3 w-3 mr-1" /> meaning preserved
              </Badge>
            ) : (
              <Badge variant="outline" className="border-amber-500/30 text-amber-700">
                <X className="h-3 w-3 mr-1" /> meaning drifted
              </Badge>
            )}
          </div>
          <CandidateBlock candidate={result.best} inLabel={inLabel} outLabel={outLabel} />
        </Card>
      ) : null}

      {result.candidates.length > 0 ? (
        <Card className="p-4 space-y-3">
          <h3 className="font-semibold text-sm">All finalists ranked by acoustic similarity</h3>
          <div className="space-y-3">
            {result.candidates.map((c, i) => (
              <div key={i} className="rounded-lg border bg-card p-3" data-testid={`flit-candidate-${i}`}>
                <CandidateBlock candidate={c} inLabel={inLabel} outLabel={outLabel} />
              </div>
            ))}
          </div>
        </Card>
      ) : null}
    </div>
  );
}

function CandidateBlock({ candidate, inLabel, outLabel }: { candidate: FlitCandidate; inLabel: string; outLabel: string }) {
  return (
    <div className="space-y-2">
      <div className="grid sm:grid-cols-2 gap-3">
        <div>
          <Badge variant="secondary" className="text-[10px] mb-1">{inLabel} (paraphrase)</Badge>
          <div className="font-medium">{candidate.inputParaphrase}</div>
          {candidate.inputParaphraseGloss ? (
            <p className="text-xs text-muted-foreground italic">"{candidate.inputParaphraseGloss}"</p>
          ) : null}
        </div>
        <div>
          <Badge variant="secondary" className="text-[10px] mb-1">{outLabel} (sound-alike)</Badge>
          <div className="font-medium">{candidate.targetText}</div>
          {candidate.targetGloss ? (
            <p className="text-xs text-muted-foreground italic">"{candidate.targetGloss}"</p>
          ) : null}
        </div>
      </div>
      <SimilarityMeter value={candidate.similarity} />
      {candidate.componentScores.length > 0 ? (
        <ComponentScores
          components={candidate.componentScores.map((c) => ({ id: c.id, label: c.label, similarity: c.similarity, distance: c.distance }))}
          size="sm"
        />
      ) : null}
      {candidate.semanticNote ? (
        <p className={`text-[11px] border-l-2 pl-2 italic ${candidate.semanticOK ? "border-emerald-500 text-emerald-700" : "border-amber-500 text-amber-700"}`}>
          {candidate.semanticOK ? "✓" : "⚠"} {candidate.semanticNote}
        </p>
      ) : null}
    </div>
  );
}
