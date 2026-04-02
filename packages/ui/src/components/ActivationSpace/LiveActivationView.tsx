import { useMemo } from "react";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";

const WINDOW = 80;
const BAR_RANGE = 1;

export function LiveActivationView() {
  const tokens = useStore((s) => s.tokens);
  const cards = useStore((s) => s.cards);
  const theme = useStore((s) => s.theme);

  const latestActivations = useMemo(() => {
    const window = tokens.slice(-WINDOW);
    const result: Record<string, number> = {};
    for (const card of cards) {
      let sum = 0;
      let count = 0;
      for (const chunk of window) {
        const val = chunk.activations[card.concept];
        if (val !== undefined) {
          sum += val;
          count++;
        }
      }
      result[card.concept] = count > 0 ? sum / count : 0;
    }
    return result;
  }, [tokens, cards]);

  const sparklines = useMemo(() => {
    const window = tokens.slice(-WINDOW);
    const result: Record<string, number[]> = {};
    for (const card of cards) {
      result[card.concept] = window.map((chunk) => chunk.activations[card.concept] ?? 0);
    }
    return result;
  }, [tokens, cards]);

  if (cards.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center p-4">
        <span className="font-mono text-xs text-text-subtle text-center leading-relaxed">
          load a concept to
          <br />
          monitor activations
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 px-4 py-4 flex-1 min-h-0 overflow-auto">
      {cards.map((card) => {
        const color = getConceptColor(card.concept, theme);
        const value = latestActivations[card.concept] ?? 0;
        const spark = sparklines[card.concept] ?? [];
        const isActive = Math.abs(value) > 0.1;
        const pct = Math.min(Math.abs(value) / BAR_RANGE, 1);
        const isPositive = value >= 0;

        const sparkMax = Math.max(...spark.map(Math.abs), 0.01);
        const sparkPoints = spark
          .map((v, i) => {
            const x = (i / Math.max(spark.length - 1, 1)) * 60;
            const y = 12 - ((v / sparkMax) * 0.5 + 0.5) * 12;
            return `${x},${y}`;
          })
          .join(" ");

        return (
          <div key={card.concept} className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span
                  className="h-1.5 w-1.5 rounded-full shrink-0"
                  style={{
                    backgroundColor: color,
                    boxShadow: isActive ? `0 0 6px ${color}` : undefined,
                  }}
                />
                <span className="font-mono text-xs text-text-muted">{card.concept}</span>
              </div>
              <div className="flex items-center gap-2">
                {spark.length > 1 && (
                  <svg width={60} height={14} className="opacity-50">
                    <polyline
                      points={sparkPoints}
                      fill="none"
                      stroke={color}
                      strokeWidth={1}
                      strokeLinejoin="round"
                    />
                  </svg>
                )}
                <span
                  className="font-mono text-xs tabular-nums font-medium w-12 text-right"
                  style={{ color: isActive ? color : "var(--color-text-subtle)" }}
                >
                  {value >= 0 ? "+" : ""}
                  {value.toFixed(3)}
                </span>
              </div>
            </div>

            <div className="relative h-1.5 rounded-full bg-bg-border overflow-hidden">
              <div
                className="absolute top-0 h-full rounded-full transition-all duration-150"
                style={{
                  width: `${pct * 50}%`,
                  left: isPositive ? "50%" : `${50 - pct * 50}%`,
                  backgroundColor: color,
                  opacity: 0.3 + pct * 0.7,
                }}
              />
              <div
                className="absolute top-0 bottom-0 w-px bg-bg-border"
                style={{ left: "50%" }}
              />
            </div>
          </div>
        );
      })}

      <div className="mt-auto pt-2 border-t border-bg-border flex items-center justify-between">
        <span className="font-mono text-[10px] text-text-subtle">−1</span>
        <span className="font-mono text-[10px] text-text-subtle">rolling avg ({WINDOW} tokens)</span>
        <span className="font-mono text-[10px] text-text-subtle">+1</span>
      </div>
    </div>
  );
}
