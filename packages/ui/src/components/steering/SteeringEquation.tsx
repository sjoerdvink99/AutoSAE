import { motion } from "framer-motion";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";

export function SteeringEquation() {
  const cards = useStore((s) => s.cards);

  if (cards.length === 0) return null;

  return (
    <div className="px-2 py-1.5 font-mono text-[11px] overflow-x-auto whitespace-nowrap scrollbar-none">
      <span className="text-text-subtle">h′ = h</span>
      {cards.map((card) => {
        const color = getConceptColor(card.concept);
        const isZero = card.alpha === 0;
        const sign = card.alpha >= 0 ? "+" : "−";
        const absVal = Math.abs(card.alpha).toFixed(1);
        return (
          <motion.span
            key={card.concept}
            animate={{ opacity: isZero ? 0.3 : 1 }}
            transition={{ duration: 0.2 }}
            className="tabular-nums"
            style={{ color }}
          >
            {" "}
            <motion.span
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 0.3 }}
              style={{ display: "inline-block" }}
            >
              {sign} {absVal}·v
              <sub>{card.concept}</sub>
            </motion.span>
          </motion.span>
        );
      })}
    </div>
  );
}
