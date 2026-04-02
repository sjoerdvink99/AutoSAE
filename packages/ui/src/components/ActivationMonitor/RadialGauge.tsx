import { useMemo } from "react";
import * as d3 from "d3";
import { getConceptColor } from "../../lib/constants";
import { useStore } from "../../stores/useStore";
import type { CardInfo } from "../../types";

const ROLLING_WINDOW = 10;

interface RadialGaugeProps {
  cards: CardInfo[];
  seriesData: Record<string, { value: number }[]>;
  size: number;
}

export function RadialGauge({ cards, seriesData, size }: RadialGaugeProps) {
  const theme = useStore((s) => s.theme);
  const cx = size / 2;
  const cy = size / 2;
  const maxR = size / 2 - 24;
  const innerR = maxR * 0.22;

  const segments = useMemo(() => {
    if (cards.length === 0) return [];
    const anglePerConcept = (2 * Math.PI) / cards.length;
    return cards.map((card, i) => {
      const data = seriesData[card.concept] ?? [];
      const latest = data.at(-1)?.value ?? 0;
      const rollingMean =
        data.length > 0
          ? data.slice(-ROLLING_WINDOW).reduce((s, d) => s + d.value, 0) /
            Math.min(data.length, ROLLING_WINDOW)
          : 0;
      const color = getConceptColor(card.concept, theme);
      const startAngle = i * anglePerConcept - Math.PI / 2;
      const endAngle = startAngle + anglePerConcept;
      const absVal = Math.abs(latest);
      const r = innerR + (maxR - innerR) * Math.min(absVal, 1);
      const meanR = innerR + (maxR - innerR) * Math.min(Math.abs(rollingMean), 1);
      const spokeAngle = (startAngle + endAngle) / 2;
      const labelR = maxR + 14;

      const arc = d3
        .arc()
        .innerRadius(innerR)
        .outerRadius(r)
        .startAngle(startAngle)
        .endAngle(endAngle)
        .padAngle(0.04)
        .cornerRadius(2);

      const trailArc = d3
        .arc()
        .innerRadius(meanR - 1.5)
        .outerRadius(meanR + 1.5)
        .startAngle(startAngle + 0.02)
        .endAngle(endAngle - 0.02);

      return {
        concept: card.concept,
        color,
        latest,
        rollingMean,
        arcPath: arc(null as never) ?? "",
        trailPath: trailArc(null as never) ?? "",
        spokeAngle,
        labelR,
        r,
      };
    });
  }, [cards, seriesData, maxR, innerR, theme]);

  if (segments.length === 0) return null;

  return (
    <svg width={size} height={size} style={{ display: "block", margin: "0 auto" }}>
      <circle cx={cx} cy={cy} r={innerR} fill="var(--color-grid)" stroke="var(--color-grid)" strokeWidth={1} />
      <circle cx={cx} cy={cy} r={maxR} fill="none" stroke="var(--color-grid)" strokeWidth={1} />

      {segments.map((seg) => (
        <g key={seg.concept} transform={`translate(${cx}, ${cy})`}>
          <path
            d={seg.arcPath}
            fill={seg.color}
            fillOpacity={seg.latest < 0 ? 0.25 : 0.55}
            stroke={seg.color}
            strokeWidth={0.5}
            strokeOpacity={0.6}
            style={{ transition: "d 150ms ease-out, fill-opacity 150ms ease-out" }}
          />
          <path
            d={seg.trailPath}
            fill={seg.color}
            fillOpacity={0.35}
            style={{ transition: "d 200ms ease-out" }}
          />
          <text
            x={Math.cos(seg.spokeAngle) * seg.labelR}
            y={Math.sin(seg.spokeAngle) * seg.labelR}
            textAnchor="middle"
            dominantBaseline="middle"
            fill={seg.color}
            fontSize={8}
            fontFamily="monospace"
            fillOpacity={0.85}
          >
            {seg.concept.slice(0, 5)}
          </text>
        </g>
      ))}

      <circle cx={cx} cy={cy} r={innerR - 2} fill="var(--color-bg)" />
      <text
        x={cx}
        y={cy - 5}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="var(--color-text-subtle)"
        fontSize={8}
        fontFamily="monospace"
      >
        act
      </text>
    </svg>
  );
}
