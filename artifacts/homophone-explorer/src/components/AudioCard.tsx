import { useEffect, useRef, useState } from "react";
import { Play, Pause } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Waveform } from "./Waveform";
import { Spectrogram } from "./Spectrogram";
import { audioUrlFor } from "@/lib/audio";
import type { AudioPayload } from "@workspace/api-client-react";

interface Props {
  audio: AudioPayload;
  label: string;
  sublabel?: string;
  accent?: string;
  testId?: string;
}

export function AudioCard({ audio, label, sublabel, accent = "#6366f1", testId }: Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const url = audioUrlFor(audio.wavBase64);

  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    const onEnd = () => setPlaying(false);
    a.addEventListener("ended", onEnd);
    return () => a.removeEventListener("ended", onEnd);
  }, []);

  const toggle = () => {
    const a = audioRef.current;
    if (!a) return;
    if (playing) {
      a.pause();
      setPlaying(false);
    } else {
      a.currentTime = 0;
      a.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    }
  };

  return (
    <div className="rounded-lg border bg-card/40 p-3 space-y-2" data-testid={testId}>
      <div className="flex items-center gap-3">
        <Button
          size="icon"
          variant="default"
          onClick={toggle}
          className="rounded-full"
          style={{ background: accent }}
          aria-label={playing ? `Pause ${label}` : `Play ${label}`}
          data-testid={testId ? `${testId}-play` : "audio-play"}
        >
          {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </Button>
        <div className="flex-1 min-w-0">
          <div className="font-medium truncate">{label}</div>
          {sublabel ? (
            <div className="text-xs text-muted-foreground truncate">{sublabel}</div>
          ) : null}
        </div>
        <div className="text-xs text-muted-foreground tabular-nums">
          {(audio.durationMs / 1000).toFixed(2)}s
        </div>
      </div>
      <Waveform data={audio.waveform} color={accent} />
      <Spectrogram
        frames={audio.melSpectrogram}
        melMin={audio.melMin}
        melMax={audio.melMax}
      />
      <audio ref={audioRef} src={url} preload="auto" />
    </div>
  );
}
