import { useEffect } from "react";
import { useStore } from "../stores/useStore";
import { cardsApi } from "../api/cards";

export function useAlphaHistory() {
  const undoAlpha = useStore((s) => s.undoAlpha);
  const cards = useStore((s) => s.cards);
  const alphaHistory = useStore((s) => s.alphaHistory);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if ((e.metaKey || e.ctrlKey) && e.key === "z") {
        e.preventDefault();
        undoAlpha();
        const history = alphaHistory;
        if (history.length < 2) return;
        const snapshot = history[history.length - 2];
        if (!snapshot) return;
        for (const [concept, alpha] of Object.entries(snapshot.alphas)) {
          const card = cards.find((c) => c.concept === concept);
          if (card) void cardsApi.setAlpha(concept, alpha);
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [undoAlpha, cards, alphaHistory]);
}
