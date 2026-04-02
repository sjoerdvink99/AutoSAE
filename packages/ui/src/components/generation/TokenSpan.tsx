import { memo } from "react";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";
import type { TokenChunk } from "../../types";

function getHighlightStyle(
  activations: Record<string, number>,
  alphas: Record<string, number>,
  theme: "dark" | "light"
): React.CSSProperties | undefined {
  let maxConcept: string | null = null;
  let maxVal = 0.2;

  for (const [concept, val] of Object.entries(activations)) {
    const alpha = alphas[concept] ?? 1;
    const effectiveVal = Math.abs(val) * Math.abs(alpha);
    if (effectiveVal > maxVal) {
      maxVal = effectiveVal;
      maxConcept = concept;
    }
  }

  if (!maxConcept) return undefined;

  const color = getConceptColor(maxConcept, theme);
  const rawVal = Math.abs(activations[maxConcept] ?? 0);
  const alpha = Math.abs(alphas[maxConcept] ?? 1);
  const intensity = Math.min(rawVal * alpha * 0.25, 0.4);

  return {
    backgroundColor: `${color}${Math.round(intensity * 255).toString(16).padStart(2, "0")}`,
    borderRadius: "2px",
    boxShadow: rawVal > 0.5 ? `0 0 8px ${color}30` : undefined,
    transition: "background-color 0.15s ease",
  };
}

interface Props {
  chunk: TokenChunk;
  index: number;
  isNew: boolean;
  isHovered: boolean;
  hasCards: boolean;
  alphas: Record<string, number>;
  onMouseEnter: (index: number, e: React.MouseEvent) => void;
  onMouseLeave: () => void;
  onClick: (index: number, e: React.MouseEvent) => void;
}

export const TokenSpan = memo(function TokenSpan({
  chunk,
  index,
  isNew,
  isHovered,
  hasCards,
  alphas,
  onMouseEnter,
  onMouseLeave,
  onClick,
}: Props) {
  const theme = useStore((s) => s.theme);
  const style = hasCards ? getHighlightStyle(chunk.activations, alphas, theme) : undefined;

  return (
    <span
      data-token-index={index}
      className={`cursor-default hover:ring-1 hover:ring-accent/30 hover:-translate-y-px transition-transform${isHovered ? " ring-1 ring-accent/50" : ""}${isNew ? " token-new" : ""}`}
      style={style}
      onMouseEnter={(e) => hasCards && onMouseEnter(index, e)}
      onMouseLeave={onMouseLeave}
      onClick={(e) => hasCards && onClick(index, e)}
    >
      {chunk.token}
    </span>
  );
});
