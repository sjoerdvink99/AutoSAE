import { useMemo, useState } from "react";
import { ChevronUp, ChevronDown } from "lucide-react";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";

const WINDOW = 80;

export function MonitorStrip() {
  const tokens = useStore((s) => s.tokens);
  const cards = useStore((s) => s.cards);
  const [expanded, setExpanded] = useState(false);
  const theme = useStore((s) => s.theme);

  const sparklines = useMemo(() => {
    const window = tokens.slice(-WINDOW);
    return Object.fromEntries(
      cards.map((card) => [
        card.concept,
        window.map((chunk) => chunk.activations[card.concept] ?? 0),
      ])
    );
  }, [tokens, cards]);

  const averages = useMemo(() => {
    return Object.fromEntries(
      cards.map((card) => {
        const spark = sparklines[card.concept] ?? [];
        const sum = spark.reduce((a, b) => a + b, 0);
        return [card.concept, spark.length > 0 ? sum / spark.length : 0];
      })
    );
  }, [sparklines, cards]);

  if (cards.length === 0) return null;

  return (
    <div
      className="border-t border-bg-border bg-bg/90 backdrop-blur-sm shrink-0 transition-all"
      style={{ height: expanded ? 120 : 48 }}
    >
      <div className="flex items-center px-3 h-8 gap-2">
        <span className="font-mono text-[10px] text-text-subtle uppercase tracking-wider">
          Activations
        </span>
        <div className="flex items-center gap-3 flex-1 overflow-x-auto">
          {cards.map((card) => {
            const color = getConceptColor(card.concept, theme);
            const avg = averages[card.concept] ?? 0;
            return (
              <div key={card.concept} className="flex items-center gap-1.5 shrink-0">
                <span className="h-1.5 w-1.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                <span className="font-mono text-[10px] text-text-muted">{card.concept}</span>
                <span
                  className="font-mono text-[10px] tabular-nums font-medium"
                  style={{ color: Math.abs(avg) > 0.1 ? color : "var(--color-text-subtle)" }}
                >
                  {avg >= 0 ? "+" : ""}{avg.toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
        <button
          onClick={() => setExpanded((v) => !v)}
          className="text-text-subtle hover:text-text transition-colors p-0.5 shrink-0"
        >
          {expanded ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
        </button>
      </div>

      {expanded && (
        <div className="flex gap-3 px-3 pb-2 overflow-x-auto">
          {cards.map((card) => {
            const color = getConceptColor(card.concept, theme);
            const spark = sparklines[card.concept] ?? [];
            const sparkMax = Math.max(...spark.map(Math.abs), 0.01);
            return (
              <div key={card.concept} className="flex flex-col gap-1 shrink-0">
                <span className="font-mono text-[10px]" style={{ color }}>{card.concept}</span>
                <svg width={80} height={48} className="overflow-visible">
                  <line x1={0} y1={24} x2={80} y2={24} stroke="var(--color-bg-border)" strokeWidth={1} />
                  {spark.map((v, i) => {
                    const x = (i / Math.max(spark.length - 1, 1)) * 80;
                    const h = Math.abs(v / sparkMax) * 20;
                    const isPos = v >= 0;
                    return (
                      <rect
                        key={i}
                        x={x - 0.5}
                        y={isPos ? 24 - h : 24}
                        width={1}
                        height={h}
                        fill={color}
                        opacity={0.6}
                      />
                    );
                  })}
                </svg>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
