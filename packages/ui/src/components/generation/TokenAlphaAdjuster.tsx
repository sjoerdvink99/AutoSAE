import { useCallback, useEffect, useRef } from "react";
import { ChevronDown, ChevronUp, RotateCcw } from "lucide-react";
import { motion } from "framer-motion";
import { useStore } from "../../stores/useStore";
import { cardsApi } from "../../api/cards";
import { getConceptColor } from "../../lib/constants";
import { ConceptDot } from "../ui/ConceptDot";
import { useGeneration } from "../../hooks/useGeneration";
import type { SteeringDirection } from "../../types";

interface Props {
  tokenIndex: number;
  activations: Record<string, number>;
  position: { x: number; y: number };
  onClose: () => void;
}

export function TokenAlphaAdjuster({ tokenIndex, activations, position, onClose }: Props) {
  const cards = useStore((s) => s.cards);
  const updateCardAlpha = useStore((s) => s.updateCardAlpha);
  const pushAlphaSnapshot = useStore((s) => s.pushAlphaSnapshot);
  const { regenerateFrom } = useGeneration();
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    window.addEventListener("keydown", handleKey);
    window.addEventListener("mousedown", handleClick);
    return () => {
      window.removeEventListener("keydown", handleKey);
      window.removeEventListener("mousedown", handleClick);
    };
  }, [onClose]);

  let dominantConcept: string | null = null;
  let dominantVal = 0.1;
  for (const [concept, activation] of Object.entries(activations)) {
    const absVal = Math.abs(activation);
    if (absVal > dominantVal && cards.find((c) => c.concept === concept)) {
      dominantVal = absVal;
      dominantConcept = concept;
    }
  }

  const steer = useCallback(
    (dir: SteeringDirection) => {
      pushAlphaSnapshot();
      for (const [concept, activation] of Object.entries(activations)) {
        if (Math.abs(activation) > 0.1) {
          const card = cards.find((c) => c.concept === concept);
          if (!card) continue;
          const sign = dir === "more" ? 1 : -1;
          const delta = sign * 0.3 * Math.abs(activation);
          const newAlpha = Math.max(-3, Math.min(3, card.alpha + delta));
          updateCardAlpha(concept, newAlpha);
          void cardsApi.setAlpha(concept, newAlpha);
        }
      }
    },
    [activations, cards, updateCardAlpha, pushAlphaSnapshot]
  );

  const theme = useStore((s) => s.theme);
  const dominantCard = dominantConcept ? cards.find((c) => c.concept === dominantConcept) : null;
  const color = dominantConcept ? getConceptColor(dominantConcept, theme) : "var(--color-text-muted)";

  if (!dominantCard || !dominantConcept) {
    return null;
  }

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, scale: 0.9, y: 4 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9, y: 4 }}
      transition={{ duration: 0.12 }}
      className="fixed z-50 bg-bg-elevated border border-bg-border rounded-lg shadow-xl px-3 py-2"
      style={{
        left: position.x,
        top: position.y - 8,
        transform: "translate(-50%, -100%)",
        borderColor: `${color}44`,
      }}
    >
      <div className="flex items-center gap-2">
        <ConceptDot color={color} glow />
        <span className="font-mono text-xs text-text-muted">{dominantConcept}</span>
        <span
          className="font-mono text-xs tabular-nums font-medium ml-1"
          style={{ color }}
        >
          α {dominantCard.alpha.toFixed(1)}
        </span>
        <div className="flex flex-col gap-0.5 ml-2">
          <button
            onClick={() => steer("more")}
            className="text-text-subtle hover:text-accent transition-colors"
            title="Amplify"
          >
            <ChevronUp size={14} />
          </button>
          <button
            onClick={() => steer("less")}
            className="text-text-subtle hover:text-danger transition-colors"
            title="Suppress"
          >
            <ChevronDown size={14} />
          </button>
        </div>
      </div>
      <div className="mt-1.5 flex items-center justify-between gap-2">
        <span className="text-[10px] font-mono text-text-subtle">
          token #{tokenIndex}
        </span>
        <button
          onClick={() => {
            onClose();
            regenerateFrom(tokenIndex);
          }}
          className="flex items-center gap-1 text-[10px] font-mono text-text-subtle hover:text-accent transition-colors"
          title="Regenerate from here"
        >
          <RotateCcw size={10} />
          regen
        </button>
      </div>
    </motion.div>
  );
}
