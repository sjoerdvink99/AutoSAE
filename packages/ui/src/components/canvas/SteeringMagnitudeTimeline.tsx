import { useMemo } from "react";
import * as d3 from "d3";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";

const HEIGHT = 60;
const PAD_X = 4;

interface Props {
  width: number;
}

export function SteeringMagnitudeTimeline({ width }: Props) {
  const tokens = useStore((s) => s.tokens);
  const cards = useStore((s) => s.cards);
  const theme = useStore((s) => s.theme);
  const conceptColors = useStore((s) => s.conceptColors);

  const data = useMemo(() => {
    return tokens.map((t) => {
      const entry: Record<string, number> = {};
      for (const card of cards) {
        entry[card.concept] = Math.abs((t.activations[card.concept] ?? 0) * card.alpha);
      }
      return entry;
    });
  }, [tokens, cards]);

  const stacked = useMemo(() => {
    if (cards.length === 0 || data.length === 0) return null;
    const series = d3
      .stack<Record<string, number>>()
      .keys(cards.map((c) => c.concept))
      .order(d3.stackOrderNone)
      .offset(d3.stackOffsetNone)(data);
    return series;
  }, [data, cards]);

  if (!stacked || cards.length === 0 || tokens.length === 0) return null;

  const maxVal = Math.max(...data.map((d) => Object.values(d).reduce((a, b) => a + b, 0)), 0.01);
  const xScale = d3.scaleLinear([0, data.length - 1], [PAD_X, width - PAD_X]);
  const yScale = d3.scaleLinear([0, maxVal], [HEIGHT, 0]);

  const areaGen = d3
    .area<d3.SeriesPoint<Record<string, number>>>()
    .x((_, i) => xScale(i))
    .y0((d) => yScale(d[0]))
    .y1((d) => yScale(d[1]))
    .curve(d3.curveMonotoneX);

  return (
    <div className="border-t border-bg-border shrink-0" style={{ height: HEIGHT }}>
      <svg width={width} height={HEIGHT} style={{ display: "block" }}>
        {stacked.map((series) => {
          const color = getConceptColor(series.key, theme, conceptColors);
          const path = areaGen(series);
          return (
            <path
              key={series.key}
              d={path ?? ""}
              fill={color}
              fillOpacity={0.4}
              stroke={color}
              strokeWidth={0.5}
              strokeOpacity={0.6}
            />
          );
        })}
        <text x={PAD_X + 2} y={HEIGHT - 3} fill="var(--color-text-subtle)" fontSize={8} fontFamily="monospace">
          steering magnitude
        </text>
      </svg>
    </div>
  );
}
