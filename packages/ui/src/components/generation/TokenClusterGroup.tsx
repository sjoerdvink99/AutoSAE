import { memo, useState } from "react";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";
import { TokenSpan } from "./TokenSpan";
import type { TokenChunk, TokenCluster } from "../../types";

interface Props {
  cluster: TokenCluster;
  tokens: TokenChunk[];
  hoveredTokenIndex: number | null;
  hasCards: boolean;
  alphas: Record<string, number>;
  onMouseEnter: (index: number, e: React.MouseEvent) => void;
  onMouseLeave: () => void;
  onClick: (index: number, e: React.MouseEvent) => void;
  newTokenThreshold: number;
}

export const TokenClusterGroup = memo(function TokenClusterGroup({
  cluster,
  tokens,
  hoveredTokenIndex,
  hasCards,
  alphas,
  onMouseEnter,
  onMouseLeave,
  onClick,
  newTokenThreshold,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const theme = useStore((s) => s.theme);
  const color = cluster.dominantConcept
    ? getConceptColor(cluster.dominantConcept, theme)
    : null;

  const isMultiToken = cluster.endIndex - cluster.startIndex > 1;

  if (!isMultiToken || expanded) {
    const inner = tokens.slice(cluster.startIndex, cluster.endIndex).map((chunk, j) => {
      const i = cluster.startIndex + j;
      return (
        <TokenSpan
          key={i}
          chunk={chunk}
          index={i}
          isNew={i >= newTokenThreshold}
          isHovered={hoveredTokenIndex === i}
          hasCards={hasCards}
          alphas={alphas}
          onMouseEnter={onMouseEnter}
          onMouseLeave={onMouseLeave}
          onClick={onClick}
        />
      );
    });

    if (!isMultiToken) return <>{inner}</>;

    return (
      <span
        className="rounded cursor-pointer"
        style={{
          backgroundColor: color ? `${color}18` : undefined,
          outline: color ? `1px solid ${color}30` : undefined,
          outlineOffset: "1px",
        }}
        onClick={() => setExpanded(false)}
        title="Click to collapse"
      >
        {inner}
      </span>
    );
  }

  return (
    <span
      className="rounded cursor-pointer inline"
      style={{
        backgroundColor: color ? `${color}18` : undefined,
        outline: color ? `1px solid ${color}30` : undefined,
        outlineOffset: "1px",
      }}
      onClick={() => setExpanded(true)}
      title={cluster.dominantConcept ? `Cluster: ${cluster.dominantConcept}` : "Cluster"}
    >
      {tokens.slice(cluster.startIndex, cluster.endIndex).map((chunk) => chunk.token).join("")}
    </span>
  );
});
