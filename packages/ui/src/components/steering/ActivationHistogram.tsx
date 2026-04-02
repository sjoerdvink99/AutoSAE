import { useMemo } from "react";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";

const BINS = 20;
const MIN = -1;
const MAX = 1;
const WIDTH = 80;
const HEIGHT = 40;

function computeBins(values: number[]): number[] {
  const counts = new Array<number>(BINS).fill(0);
  for (const v of values) {
    const idx = Math.floor(((v - MIN) / (MAX - MIN)) * BINS);
    const clamped = Math.max(0, Math.min(BINS - 1, idx));
    counts[clamped]++;
  }
  return counts;
}

export function ActivationHistogram() {
  const tokens = useStore((s) => s.tokens);
  const cards = useStore((s) => s.cards);
  const theme = useStore((s) => s.theme);

  const histograms = useMemo(() => {
    return Object.fromEntries(
      cards.map((card) => {
        const values = tokens.map((t) => t.activations[card.concept] ?? 0);
        return [card.concept, computeBins(values)];
      })
    );
  }, [tokens, cards]);

  if (cards.length === 0 || tokens.length === 0) return null;

  return (
    <div className="flex flex-col gap-2 px-2 py-2 border-t border-bg-border">
      <span className="font-mono text-[10px] text-text-subtle uppercase tracking-wider">
        Activation Distributions
      </span>
      <div className="flex flex-wrap gap-3">
        {cards.map((card) => {
          const color = getConceptColor(card.concept, theme);
          const bins = histograms[card.concept] ?? [];
          const maxCount = Math.max(...bins, 1);
          return (
            <div key={card.concept} className="flex flex-col gap-1">
              <span className="font-mono text-[10px]" style={{ color }}>{card.concept}</span>
              <svg width={WIDTH} height={HEIGHT}>
                {bins.map((count, i) => {
                  const h = (count / maxCount) * HEIGHT;
                  const x = (i / BINS) * WIDTH;
                  const w = WIDTH / BINS - 0.5;
                  const midVal = MIN + ((i + 0.5) / BINS) * (MAX - MIN);
                  return (
                    <rect
                      key={i}
                      x={x}
                      y={HEIGHT - h}
                      width={w}
                      height={h}
                      fill={color}
                      opacity={midVal >= 0 ? 0.7 : 0.4}
                    />
                  );
                })}
                <line x1={WIDTH / 2} y1={0} x2={WIDTH / 2} y2={HEIGHT} stroke="var(--color-bg-border)" strokeWidth={0.5} />
              </svg>
            </div>
          );
        })}
      </div>
    </div>
  );
}
