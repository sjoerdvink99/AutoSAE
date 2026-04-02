import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ThumbsUp, ThumbsDown } from "lucide-react";
import { useStore } from "../../stores/useStore";
import { cardsApi } from "../../api/cards";
import type { TokenChunk, SteeringDirection } from "../../types";

interface SelectionInfo {
  tokenIndices: number[];
  rect: DOMRect;
}

function findTokenIndex(node: Node): number | null {
  let el: Element | null = node instanceof Element ? node : node.parentElement;
  while (el) {
    const idx = el.getAttribute("data-token-index");
    if (idx !== null) return parseInt(idx, 10);
    el = el.parentElement;
  }
  return null;
}

export function useTextSelection(tokens: TokenChunk[]) {
  const [selection, setSelection] = useState<SelectionInfo | null>(null);

  useEffect(() => {
    const handler = () => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
        setSelection(null);
        return;
      }
      const range = sel.getRangeAt(0);
      const startIdx = findTokenIndex(range.startContainer);
      const endIdx = findTokenIndex(range.endContainer);
      if (startIdx === null || endIdx === null) {
        setSelection(null);
        return;
      }
      const lo = Math.min(startIdx, endIdx);
      const hi = Math.max(startIdx, endIdx);
      const indices: number[] = [];
      for (let i = lo; i <= hi; i++) {
        if (i < tokens.length) indices.push(i);
      }
      if (indices.length === 0) {
        setSelection(null);
        return;
      }
      setSelection({ tokenIndices: indices, rect: range.getBoundingClientRect() });
    };

    document.addEventListener("selectionchange", handler);
    return () => document.removeEventListener("selectionchange", handler);
  }, [tokens]);

  return { selection, clearSelection: () => setSelection(null) };
}

interface Props {
  selection: SelectionInfo | null;
  tokens: TokenChunk[];
  onClose: () => void;
}

export function TextSelectionPopup({ selection, tokens, onClose }: Props) {
  const cards = useStore((s) => s.cards);
  const updateCardAlpha = useStore((s) => s.updateCardAlpha);
  const pushAlphaSnapshot = useStore((s) => s.pushAlphaSnapshot);
  const ref = useRef<HTMLDivElement>(null);

  const steer = useCallback(
    (dir: SteeringDirection) => {
      if (!selection) return;
      const avgActivations: Record<string, number> = {};
      for (const idx of selection.tokenIndices) {
        const chunk = tokens[idx];
        if (!chunk) continue;
        for (const [concept, val] of Object.entries(chunk.activations)) {
          avgActivations[concept] = (avgActivations[concept] ?? 0) + val / selection.tokenIndices.length;
        }
      }
      pushAlphaSnapshot();
      const sign = dir === "more" ? 1 : -1;
      for (const [concept, avg] of Object.entries(avgActivations)) {
        if (Math.abs(avg) > 0.15) {
          const card = cards.find((c) => c.concept === concept);
          if (!card) continue;
          const delta = sign * 0.4 * avg;
          const newAlpha = Math.max(-3, Math.min(3, card.alpha + delta));
          updateCardAlpha(concept, newAlpha);
          void cardsApi.setAlpha(concept, newAlpha);
        }
      }
      onClose();
    },
    [selection, tokens, cards, updateCardAlpha, pushAlphaSnapshot, onClose]
  );

  if (!selection) return null;

  const x = selection.rect.left + selection.rect.width / 2;
  const y = selection.rect.top;

  return (
    <AnimatePresence>
      <motion.div
        ref={ref}
        initial={{ opacity: 0, scale: 0.9, y: 4 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 4 }}
        transition={{ duration: 0.1 }}
        className="fixed z-50 flex items-center gap-1 bg-bg-elevated border border-bg-border rounded-lg shadow-xl px-2 py-1.5"
        style={{
          left: x,
          top: y - 8,
          transform: "translate(-50%, -100%)",
        }}
      >
        <button
          onClick={() => steer("more")}
          className="flex items-center gap-1.5 px-2 py-1 rounded font-mono text-[11px] text-accent hover:bg-accent/10 transition-colors"
          title="More like this"
        >
          <ThumbsUp size={11} />
          More
        </button>
        <span className="w-px h-4 bg-bg-border" />
        <button
          onClick={() => steer("less")}
          className="flex items-center gap-1.5 px-2 py-1 rounded font-mono text-[11px] text-danger hover:bg-danger/10 transition-colors"
          title="Less like this"
        >
          <ThumbsDown size={11} />
          Less
        </button>
      </motion.div>
    </AnimatePresence>
  );
}
