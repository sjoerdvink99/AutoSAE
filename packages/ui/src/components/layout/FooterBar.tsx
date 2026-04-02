import { useCallback, useEffect, useState } from "react";
import { Download } from "lucide-react";
import { useStore } from "../../stores/useStore";
import { Button } from "../ui/Button";

const HINTS = [
  { key: "1–9", label: "select" },
  { key: "↑↓", label: "α ±0.1" },
  { key: "⇧↑↓", label: "±0.5" },
  { key: "[/]", label: "α ±1" },
  { key: "H", label: "trajectory" },
  { key: "G", label: "grid" },
  { key: "R", label: "regenerate" },
  { key: "⌘Z", label: "undo" },
  { key: "⌘K", label: "commands" },
];

function useDurationTimer(sessionStart: number) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setElapsed(Date.now() - sessionStart), 1000);
    return () => clearInterval(id);
  }, [sessionStart]);

  const s = Math.floor(elapsed / 1000) % 60;
  const m = Math.floor(elapsed / 60000);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function FooterBar() {
  const steeringEvents = useStore((s) => s.steeringEvents);
  const sessionStart = useStore((s) => s.sessionStart);
  const tokens = useStore((s) => s.tokens);
  const cards = useStore((s) => s.cards);
  const prompt = useStore((s) => s.prompt);
  const duration = useDurationTimer(sessionStart);

  const exportSession = useCallback(() => {
    const session = {
      prompt,
      finalOutput: tokens.map((t) => t.token).join(""),
      events: steeringEvents,
      conceptCards: cards.map((c) => ({ concept: c.concept, finalAlpha: c.alpha })),
      duration: Date.now() - sessionStart,
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(session, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `autosae-session-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [tokens, cards, steeringEvents, sessionStart, prompt]);

  return (
    <div className="flex items-center h-6 border-t border-bg-border px-3 shrink-0 bg-bg/80">
      <div className="flex items-center gap-x-3 gap-y-0 flex-wrap">
        {HINTS.map(({ key, label }) => (
          <div key={key} className="flex items-center gap-1">
            <kbd className="font-mono text-[9px] text-text-subtle border border-bg-border rounded px-1 py-px bg-bg-elevated">
              {key}
            </kbd>
            <span className="font-mono text-[9px] text-text-subtle opacity-60">{label}</span>
          </div>
        ))}
      </div>
      <div className="ml-auto flex items-center gap-3">
        <span className="font-mono text-[10px] text-text-subtle tabular-nums">
          {steeringEvents.length} events · {duration}
        </span>
        <Button variant="ghost" size="sm" onClick={exportSession} disabled={steeringEvents.length === 0}>
          <Download size={10} />
          Export
        </Button>
      </div>
    </div>
  );
}
