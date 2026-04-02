import { useState } from "react";
import { Plus, Wand2, Map } from "lucide-react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { cardsApi } from "../../api/cards";
import { useStore } from "../../stores/useStore";
import { Button } from "../ui/Button";
import { CardLoader } from "../CardLoader";
import { CardBuilder } from "../CardBuilder";
import { ConceptExplorer } from "../ConceptExplorer";
import { ConceptChip } from "./ConceptChip";

export function ConceptStrip() {
  const [loaderOpen, setLoaderOpen] = useState(false);
  const [builderOpen, setBuilderOpen] = useState(false);
  const [explorerOpen, setExplorerOpen] = useState(false);
  const cards = useStore((s) => s.cards);
  const setCards = useStore((s) => s.setCards);
  const selectedCardIndex = useStore((s) => s.selectedCardIndex);
  const capabilities = useStore((s) => s.capabilities);
  const queryClient = useQueryClient();

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
    <div className="flex items-center gap-2 border-t border-bg-border bg-bg-surface px-3 shrink-0" style={{ height: 56 }}>
      <AnimatePresence initial={false}>
        {cards.length === 0 ? (
          <motion.span
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="font-mono text-xs text-text-subtle"
          >
            No concepts loaded — click Load or Map to get started
          </motion.span>
        ) : (
          <motion.div
            key="chips"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-2 flex-1 overflow-x-auto"
            style={{ scrollbarWidth: "none" }}
          >
            {cards.map((card, i) => (
              <motion.div
                key={card.concept}
                initial={{ opacity: 0, x: -12, scale: 0.9 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 12, scale: 0.9 }}
                transition={{ duration: 0.18 }}
              >
                <ConceptChip
                  card={card}
                  isSelected={selectedCardIndex === i}
                />
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex items-center gap-1.5 ml-auto shrink-0">
        <button
          onClick={() => setExplorerOpen(true)}
          className={`flex items-center gap-1 font-mono text-[11px] px-2 py-1 rounded border transition-colors ${
            explorerOpen
              ? "bg-accent/10 text-accent border-accent/20"
              : "text-text-subtle border-bg-border hover:border-bg-elevated hover:text-text"
          }`}
          title="Concept map"
        >
          <Map size={12} />
          Map
        </button>
        {capabilities.supports_extraction && (
          <button
            onClick={() => setBuilderOpen(true)}
            className="p-1.5 rounded border border-bg-border text-text-subtle hover:border-bg-elevated hover:text-text transition-colors"
            title="Create card from contrastive prompts"
          >
            <Wand2 size={13} />
          </button>
        )}
        <Button
          variant="primary"
          size="sm"
          onClick={() => {
            setLoaderOpen(true);
            queryClient.invalidateQueries({ queryKey: ["concept-map"] });
          }}
        >
          <Plus size={12} />
          Load
        </Button>
      </div>

      <CardLoader open={loaderOpen} onClose={() => setLoaderOpen(false)} />
      <CardBuilder open={builderOpen} onClose={() => setBuilderOpen(false)} />
      <ConceptExplorer open={explorerOpen} onClose={() => setExplorerOpen(false)} />
    </div>
  );
}
