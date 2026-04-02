import { useMemo } from "react";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";

interface Props {
  totalLayers?: number;
}

export function LayerDiagram({ totalLayers }: Props) {
  const cards = useStore((s) => s.cards);
  const isGenerating = useStore((s) => s.isGenerating);
  const tokens = useStore((s) => s.tokens);

  const injectionLayers = useMemo(
    () => cards.map((c) => ({ layer: c.layer, concept: c.concept, color: getConceptColor(c.concept) })),
    [cards],
  );

  if (cards.length === 0) return null;

  const effectiveLayers = totalLayers ?? Math.max(32, ...cards.map((c) => c.layer + 1));
  const lastToken = tokens.at(-1);
  const hasActivity = isGenerating && lastToken && Object.keys(lastToken.activations).length > 0;
  const labelLayers = [0, 8, 16, 24, effectiveLayers - 1];

  return (
    <div className="flex flex-col gap-0.5 px-3 py-2 border-b border-bg-border overflow-hidden shrink-0">
      <div className="flex items-center gap-0 relative" style={{ height: 36 }}>
        <span className="font-mono text-[9px] text-text-subtle shrink-0 mr-1 self-end pb-1">L</span>
        {Array.from({ length: effectiveLayers }, (_, i) => {
          const injection = injectionLayers.find((il) => il.layer === i);
          const isInjected = !!injection;
          const showLabel = labelLayers.includes(i);
          return (
            <div key={i} className="flex-1 flex flex-col items-center gap-0.5">
              {showLabel ? (
                <span className="font-mono text-[7px] text-text-muted tabular-nums">{i}</span>
              ) : (
                <span className="text-[7px]">&nbsp;</span>
              )}
              <div
                className="h-5 w-full rounded-sm transition-all duration-100"
                style={{
                  backgroundColor: isInjected ? injection.color : "#1a1a1a",
                  boxShadow:
                    isInjected && hasActivity
                      ? `0 0 8px ${injection.color}aa`
                      : undefined,
                  opacity: isInjected ? (hasActivity ? 1 : 0.6) : 0.3,
                  transform: isInjected && hasActivity ? "scaleY(1.3)" : "scaleY(1)",
                }}
                title={isInjected ? `Layer ${i}: ${injection.concept}` : `Layer ${i}`}
              />
            </div>
          );
        })}
        <span className="font-mono text-[9px] text-text-subtle shrink-0 ml-1 self-end pb-1">Out</span>
      </div>
    </div>
  );
}
