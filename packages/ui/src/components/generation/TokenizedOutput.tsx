import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";
import { clusterTokens } from "../../lib/clustering";
import { ConceptDot } from "../ui/ConceptDot";
import { TokenSpan } from "./TokenSpan";
import { TokenClusterGroup } from "./TokenClusterGroup";
import { TokenAlphaAdjuster } from "./TokenAlphaAdjuster";
import { TextSelectionPopup, useTextSelection } from "./TextSelectionSteering";
import { useTokenRows } from "./useTokenRows";
import type { TokenChunk } from "../../types";

const NEW_TOKEN_WINDOW = 8;

interface ActivationTooltipProps {
  activations: Record<string, number>;
  position: { x: number; y: number };
}

function ActivationTooltip({ activations, position }: ActivationTooltipProps) {
  const theme = useStore((s) => s.theme);
  const entries = Object.entries(activations).sort(
    ([, a], [, b]) => Math.abs(b) - Math.abs(a),
  );

  if (entries.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 4 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95, y: 4 }}
      transition={{ duration: 0.1 }}
      className="fixed z-50 pointer-events-none"
      style={{
        left: position.x,
        top: position.y - 8,
        transform: "translate(-50%, -100%)",
      }}
    >
      <div className="bg-bg-elevated border border-bg-border rounded-lg shadow-xl px-3 py-2 min-w-[140px]">
        <div className="text-[10px] uppercase tracking-wider text-text-subtle mb-1.5 font-medium">
          Activations
        </div>
        <div className="flex flex-col gap-1">
          {entries.map(([concept, value]) => {
            const color = getConceptColor(concept, theme);
            const isActive = Math.abs(value) > 0.25;
            return (
              <div key={concept} className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-1.5">
                  <ConceptDot color={color} glow={isActive} />
                  <span className="font-mono text-xs text-text-muted">{concept}</span>
                </div>
                <span
                  className="font-mono text-xs tabular-nums font-medium"
                  style={{ color: isActive ? color : "var(--color-text-subtle)" }}
                >
                  {value >= 0 ? "+" : ""}
                  {value.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </motion.div>
  );
}

interface Props {
  tokens: TokenChunk[];
  clusterPhrases?: boolean;
}

export function TokenizedOutput({ tokens, clusterPhrases = false }: Props) {
  const cards = useStore((s) => s.cards);
  const hoveredTokenIndex = useStore((s) => s.hoveredTokenIndex);
  const setHoveredTokenIndex = useStore((s) => s.setHoveredTokenIndex);
  const setSelectedTokenIndex = useStore((s) => s.setSelectedTokenIndex);
  const isGenerating = useStore((s) => s.isGenerating);
  const hasCards = cards.length > 0;

  const clusters = useMemo(
    () => (clusterPhrases && hasCards ? clusterTokens(tokens) : null),
    [clusterPhrases, hasCards, tokens],
  );

  const alphas = useMemo(() => {
    const result: Record<string, number> = {};
    for (const card of cards) {
      result[card.concept] = card.alpha;
    }
    return result;
  }, [cards]);

  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(600);
  const [hoveredToken, setHoveredToken] = useState<{
    index: number;
    position: { x: number; y: number };
  } | null>(null);
  const [selectedToken, setSelectedToken] = useState<{
    index: number;
    position: { x: number; y: number };
  } | null>(null);

  const { selection, clearSelection } = useTextSelection(tokens);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const e = entries[0];
      if (e) setContainerWidth(e.contentRect.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const rows = useTokenRows(tokens, containerWidth);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => containerRef.current,
    estimateSize: () => 24,
    overscan: 5,
  });

  useEffect(() => {
    if (isGenerating && rows.length > 0) {
      virtualizer.scrollToIndex(rows.length - 1, { behavior: "smooth" });
    }
  }, [rows.length, isGenerating, virtualizer]);

  const handleMouseEnter = useCallback(
    (index: number, e: React.MouseEvent) => {
      const rect = e.currentTarget.getBoundingClientRect();
      setHoveredToken({
        index,
        position: { x: rect.left + rect.width / 2, y: rect.top },
      });
      setHoveredTokenIndex(index);
    },
    [setHoveredTokenIndex],
  );

  const handleMouseLeave = useCallback(() => {
    setHoveredToken(null);
    setHoveredTokenIndex(null);
  }, [setHoveredTokenIndex]);

  const handleTokenClick = useCallback(
    (index: number, e: React.MouseEvent) => {
      e.stopPropagation();
      const rect = e.currentTarget.getBoundingClientRect();
      const pos = { x: rect.left + rect.width / 2, y: rect.top };
      if (selectedToken?.index === index) {
        setSelectedToken(null);
        setSelectedTokenIndex(null);
      } else {
        setSelectedToken({ index, position: pos });
        setSelectedTokenIndex(index);
      }
    },
    [selectedToken, setSelectedTokenIndex],
  );

  const handleContainerClick = useCallback(() => {
    setSelectedToken(null);
    setSelectedTokenIndex(null);
  }, [setSelectedTokenIndex]);

  const newTokenThreshold = tokens.length - NEW_TOKEN_WINDOW;

  return (
    <div
      ref={containerRef}
      className="relative overflow-auto flex-1 min-h-0"
      onClick={handleContainerClick}
    >
      <div
        style={{
          height: virtualizer.getTotalSize(),
          width: "100%",
          position: "relative",
        }}
      >
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const row = rows[virtualRow.index];
          if (!row) return null;
          return (
            <div
              key={virtualRow.key}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
              className="font-mono text-sm text-text leading-relaxed whitespace-pre-wrap"
            >
              {clusters
                ? clusters
                    .filter((cl) => cl.startIndex < row.end && cl.endIndex > row.start)
                    .map((cl) => (
                      <TokenClusterGroup
                        key={cl.startIndex}
                        cluster={cl}
                        tokens={tokens}
                        hoveredTokenIndex={hoveredTokenIndex}
                        hasCards={hasCards}
                        alphas={alphas}
                        onMouseEnter={handleMouseEnter}
                        onMouseLeave={handleMouseLeave}
                        onClick={handleTokenClick}
                        newTokenThreshold={newTokenThreshold}
                      />
                    ))
                : Array.from({ length: row.end - row.start }, (_, j) => {
                    const i = row.start + j;
                    const chunk = tokens[i];
                    if (!chunk) return null;
                    return (
                      <TokenSpan
                        key={i}
                        chunk={chunk}
                        index={i}
                        isNew={i >= newTokenThreshold}
                        isHovered={hoveredTokenIndex === i}
                        hasCards={hasCards}
                        alphas={alphas}
                        onMouseEnter={handleMouseEnter}
                        onMouseLeave={handleMouseLeave}
                        onClick={handleTokenClick}
                      />
                    );
                  })}
            </div>
          );
        })}
      </div>

      {isGenerating && tokens.length > 0 && (
        <motion.span
          className="inline-block w-0.5 h-4 bg-accent ml-0.5 align-middle"
          animate={{ opacity: [1, 0] }}
          transition={{ duration: 0.8, repeat: Infinity, repeatType: "reverse" }}
        />
      )}

      <AnimatePresence>
        {hoveredToken !== null && hasCards && tokens[hoveredToken.index] && (
          <ActivationTooltip
            activations={tokens[hoveredToken.index].activations}
            position={hoveredToken.position}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {selectedToken !== null && tokens[selectedToken.index] && hasCards && (
          <TokenAlphaAdjuster
            tokenIndex={selectedToken.index}
            activations={tokens[selectedToken.index].activations}
            position={selectedToken.position}
            onClose={() => {
              setSelectedToken(null);
              setSelectedTokenIndex(null);
            }}
          />
        )}
      </AnimatePresence>

      {hasCards && (
        <TextSelectionPopup
          selection={selection}
          tokens={tokens}
          onClose={clearSelection}
        />
      )}
    </div>
  );
}
