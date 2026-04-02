import { useCallback } from "react";
import { Download } from "lucide-react";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";
import { Button } from "../ui/Button";

function valueToColor(val: number): string {
  const clamped = Math.max(-1, Math.min(1, val));
  if (clamped >= 0) {
    const alpha = Math.round(clamped * 200);
    return `rgba(0,230,118,${(alpha / 255).toFixed(2)})`;
  } else {
    const alpha = Math.round(-clamped * 200);
    return `rgba(255,23,68,${(alpha / 255).toFixed(2)})`;
  }
}

function GramMatrixHeatmap() {
  const geometry = useStore((s) => s.geometry);

  if (!geometry || geometry.concepts.length < 2) {
    return (
      <div className="flex items-center justify-center p-4">
        <span className="font-mono text-xs text-text-subtle text-center">
          load 2+ concepts to see gram matrix
        </span>
      </div>
    );
  }

  const { gram, concepts } = geometry;

  return (
    <div className="flex flex-col gap-2">
      <span className="font-mono text-[10px] text-text-subtle uppercase tracking-wider">
        Concept Orthogonality (Gram Matrix)
      </span>
      <div className="flex gap-2">
        <div className="flex flex-col gap-px">
          {concepts.map((c) => (
            <div key={c} className="h-8 flex items-center">
              <span
                className="font-mono text-[10px] w-16 truncate text-right pr-1"
                style={{ color: getConceptColor(c) }}
              >
                {c}
              </span>
            </div>
          ))}
        </div>
        <div className="flex flex-col gap-px flex-1">
          {gram.map((row, i) => (
            <div key={i} className="flex gap-px h-8">
              {row.map((val, j) => (
                <div
                  key={j}
                  className="flex-1 rounded-sm flex items-center justify-center"
                  style={{ backgroundColor: valueToColor(val) }}
                  title={`${concepts[i]} × ${concepts[j]}: ${val.toFixed(3)}`}
                >
                  <span className="font-mono text-[8px] text-white/70">
                    {val.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-2 justify-end">
        <span className="font-mono text-[9px] text-danger">−1 orthogonal</span>
        <div className="flex h-1.5 w-20 rounded overflow-hidden">
          <div className="flex-1" style={{ background: "rgba(255,23,68,0.8)" }} />
          <div className="flex-1 bg-bg-border" />
          <div className="flex-1" style={{ background: "rgba(0,230,118,0.8)" }} />
        </div>
        <span className="font-mono text-[9px] text-accent">+1 parallel</span>
      </div>
    </div>
  );
}

function SessionExport() {
  const tokens = useStore((s) => s.tokens);
  const cards = useStore((s) => s.cards);
  const steeringEvents = useStore((s) => s.steeringEvents);
  const sessionStart = useStore((s) => s.sessionStart);
  const prompt = useStore((s) => s.prompt);

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
    <div className="flex items-center justify-between">
      <div className="flex flex-col gap-0.5">
        <span className="font-mono text-[10px] text-text-subtle uppercase tracking-wider">
          Session Recording
        </span>
        <span className="font-mono text-xs text-text-muted">
          {steeringEvents.length} events recorded
        </span>
      </div>
      <Button variant="ghost" size="sm" onClick={exportSession} disabled={steeringEvents.length === 0}>
        <Download size={12} />
        Export
      </Button>
    </div>
  );
}

export function HistoryView() {
  const alphaHistory = useStore((s) => s.alphaHistory);

  return (
    <div className="flex flex-col gap-4 overflow-auto h-full px-1 py-1">
      <SessionExport />

      <div className="border-t border-bg-border pt-3">
        <GramMatrixHeatmap />
      </div>

      {alphaHistory.length > 0 && (
        <div className="border-t border-bg-border pt-3 flex flex-col gap-2">
          <span className="font-mono text-[10px] text-text-subtle uppercase tracking-wider">
            Alpha History ({alphaHistory.length} snapshots)
          </span>
          <div className="flex flex-col gap-1 max-h-32 overflow-y-auto">
            {[...alphaHistory].reverse().map((snapshot, i) => (
              <div key={snapshot.timestamp} className="flex items-center gap-2 font-mono text-[10px] text-text-muted">
                <span className="text-text-subtle tabular-nums w-4">{alphaHistory.length - i}</span>
                <span className="text-text-subtle">
                  {new Date(snapshot.timestamp).toLocaleTimeString()}
                </span>
                <span className="text-text-subtle truncate">
                  {Object.entries(snapshot.alphas)
                    .map(([c, a]) => `${c}=${a >= 0 ? "+" : ""}${a.toFixed(1)}`)
                    .join(" ")}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
