import { useMemo } from "react";
import * as d3 from "d3";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";

const W = 200;
const H = 120;
const COL_W = 56;
const PAD = 8;

interface FlowNode {
  concept: string;
  absAlpha: number;
  meanActivation: number;
  layer: number;
  color: string;
}

export function InfluenceFlow() {
  const cards = useStore((s) => s.cards);
  const tokens = useStore((s) => s.tokens);
  const theme = useStore((s) => s.theme);

  const nodes: FlowNode[] = useMemo(() => {
    return cards.map((card) => {
      const activations = tokens.map((t) => Math.abs(t.activations[card.concept] ?? 0));
      const mean = activations.length > 0 ? activations.reduce((a, b) => a + b, 0) / activations.length : 0;
      return {
        concept: card.concept,
        absAlpha: Math.abs(card.alpha),
        meanActivation: mean,
        layer: card.layer,
        color: getConceptColor(card.concept, theme),
      };
    });
  }, [cards, tokens, theme]);

  if (nodes.length < 2) return null;

  const totalAbsAlpha = nodes.reduce((s, n) => s + n.absAlpha, 0) || 1;
  const maxMean = Math.max(...nodes.map((n) => n.meanActivation), 0.01);
  const maxLayer = Math.max(...nodes.map((n) => n.layer), 1);

  const cardH = (H - PAD * 2) / nodes.length;

  const linkGen = d3.linkHorizontal<{ source: [number, number]; target: [number, number] }, [number, number]>()
    .x((d) => d[0])
    .y((d) => d[1]);

  return (
    <div className="flex flex-col gap-1 px-2 py-2 border-t border-bg-border">
      <span className="font-mono text-[10px] text-text-subtle uppercase tracking-wider">
        Influence Flow
      </span>
      <svg width={W} height={H}>
        {nodes.map((node, i) => {
          const cy = PAD + (i + 0.5) * cardH;
          const alphaH = Math.max(4, (node.absAlpha / (totalAbsAlpha / nodes.length)) * (cardH - 4));
          const layerX = COL_W + ((node.layer / maxLayer) * (W - COL_W * 2 - PAD * 2));
          const actH = Math.max(4, (node.meanActivation / maxMean) * (cardH - 4));

          const leftLink = {
            source: [COL_W, cy] as [number, number],
            target: [layerX, cy] as [number, number],
          };
          const rightLink = {
            source: [layerX, cy] as [number, number],
            target: [W - COL_W, cy] as [number, number],
          };

          const strokeW = Math.max(1, node.absAlpha * node.meanActivation * 4);

          return (
            <g key={node.concept}>
              <rect
                x={2}
                y={cy - alphaH / 2}
                width={COL_W - 4}
                height={alphaH}
                rx={2}
                fill={node.color}
                fillOpacity={0.3}
                stroke={node.color}
                strokeOpacity={0.5}
                strokeWidth={0.5}
              />
              <text x={COL_W / 2} y={cy} fill={node.color} fontSize={8} fontFamily="monospace" textAnchor="middle" dominantBaseline="middle">
                {node.concept.slice(0, 5)}
              </text>

              <path
                d={linkGen(leftLink) ?? ""}
                fill="none"
                stroke={node.color}
                strokeWidth={strokeW}
                strokeOpacity={0.35}
              />

              <circle cx={layerX} cy={cy} r={3} fill={node.color} fillOpacity={0.6} />
              <text x={layerX} y={cy + cardH / 2 - 1} fill="var(--color-text-subtle)" fontSize={7} fontFamily="monospace" textAnchor="middle">
                L{node.layer}
              </text>

              <path
                d={linkGen(rightLink) ?? ""}
                fill="none"
                stroke={node.color}
                strokeWidth={strokeW}
                strokeOpacity={0.35}
              />

              <rect
                x={W - COL_W + 2}
                y={cy - actH / 2}
                width={COL_W - 4}
                height={actH}
                rx={2}
                fill={node.color}
                fillOpacity={0.3}
                stroke={node.color}
                strokeOpacity={0.5}
                strokeWidth={0.5}
              />
              <text x={W - COL_W / 2} y={cy} fill={node.color} fontSize={8} fontFamily="monospace" textAnchor="middle" dominantBaseline="middle">
                {node.meanActivation.toFixed(2)}
              </text>
            </g>
          );
        })}

        <text x={COL_W / 2} y={H - 2} fill="var(--color-text-subtle)" fontSize={7} fontFamily="monospace" textAnchor="middle">cards</text>
        <text x={W / 2} y={H - 2} fill="var(--color-text-subtle)" fontSize={7} fontFamily="monospace" textAnchor="middle">layers</text>
        <text x={W - COL_W / 2} y={H - 2} fill="var(--color-text-subtle)" fontSize={7} fontFamily="monospace" textAnchor="middle">avg act</text>
      </svg>
    </div>
  );
}
