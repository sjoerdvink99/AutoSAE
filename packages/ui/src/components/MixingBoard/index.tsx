import { useState } from "react";
import { Plus, Wand2 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { cardsApi } from "../../api/cards";
import { useStore } from "../../stores/useStore";
import { ConceptSlider } from "./ConceptSlider";
import { AlphaTimeline } from "./AlphaTimeline";
import { CardLoader } from "../CardLoader";
import { CardBuilder } from "../CardBuilder";
import { ShortcutHints } from "../ShortcutHints";
import { Button } from "../ui/Button";

export function MixingBoard() {
  const [loaderOpen, setLoaderOpen] = useState(false);
  const [builderOpen, setBuilderOpen] = useState(false);
  const cards = useStore((s) => s.cards);
  const setCards = useStore((s) => s.setCards);
  const capabilities = useStore((s) => s.capabilities);
  const selectedCardIndex = useStore((s) => s.selectedCardIndex);

  useQuery({
    queryKey: ["cards"],
    queryFn: async () => {
      const result = await cardsApi.list();
      setCards(result);
      return result;
    },
    staleTime: 10_000,
  });

  return (
    <aside className="flex h-full w-64 flex-shrink-0 flex-col border-r border-bg-border bg-bg/80 backdrop-blur-sm">
      <div className="flex items-center gap-3 border-b border-bg-border px-4 py-3 shrink-0">
        <h2 className="font-mono text-sm font-semibold text-text">Concept Cards</h2>
        {cards.length > 0 && (
          <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-bg-elevated text-text-subtle border border-bg-border tabular-nums">
            {cards.length}
          </span>
        )}
        <div className="flex items-center gap-1.5 ml-auto">
          {capabilities.supports_extraction && (
            <Button variant="ghost" size="sm" onClick={() => setBuilderOpen(true)} title="Create card">
              <Wand2 size={12} />
            </Button>
          )}
          <Button variant="primary" size="sm" onClick={() => setLoaderOpen(true)}>
            <Plus size={12} />
            Load
          </Button>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-2 overflow-y-auto p-3">
        <AnimatePresence initial={false}>
          {cards.length === 0 && (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-1 flex-col items-center justify-center gap-3 text-center"
            >
              <div className="h-10 w-10 rounded-full border border-bg-border flex items-center justify-center">
                <Plus size={16} className="text-text-subtle" />
              </div>
              <p className="font-mono text-xs text-text-muted max-w-[180px]">
                Load a Concept Card to begin steering model behaviour.
              </p>
              <div className="flex flex-col gap-2 items-center">
                <Button variant="ghost" size="sm" onClick={() => setLoaderOpen(true)}>
                  Browse registry
                </Button>
                {capabilities.supports_extraction && (
                  <Button variant="ghost" size="sm" onClick={() => setBuilderOpen(true)}>
                    <Wand2 size={12} />
                    Create card
                  </Button>
                )}
              </div>
            </motion.div>
          )}

          {cards.map((card, i) => (
            <motion.div
              key={card.concept}
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
            >
              <ConceptSlider card={card} isSelected={selectedCardIndex === i} />
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      <AlphaTimeline />
      <ShortcutHints />

      <CardLoader open={loaderOpen} onClose={() => setLoaderOpen(false)} />
      <CardBuilder open={builderOpen} onClose={() => setBuilderOpen(false)} />
    </aside>
  );
}
