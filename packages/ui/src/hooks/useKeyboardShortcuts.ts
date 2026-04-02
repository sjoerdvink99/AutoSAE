import { useEffect } from "react";
import { useStore } from "../stores/useStore";
import { cardsApi } from "../api/cards";
import { useGeneration } from "./useGeneration";

export function useKeyboardShortcuts() {
  const cards = useStore((s) => s.cards);
  const selectedCardIndex = useStore((s) => s.selectedCardIndex);
  const setSelectedCardIndex = useStore((s) => s.setSelectedCardIndex);
  const updateCardAlpha = useStore((s) => s.updateCardAlpha);
  const pushAlphaSnapshot = useStore((s) => s.pushAlphaSnapshot);
  const isGenerating = useStore((s) => s.isGenerating);
  const toggleTrajectory = useStore((s) => s.toggleTrajectory);
  const toggleGrid = useStore((s) => s.toggleGrid);
  const resetCanvasViewport = useStore((s) => s.resetCanvasViewport);

  const { generate, stop } = useGeneration();
  const prompt = useStore((s) => s.prompt);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;

      if (e.ctrlKey && e.key === "0") {
        e.preventDefault();
        resetCanvasViewport();
        return;
      }

      const n = parseInt(e.key, 10);
      if (!isNaN(n) && n >= 1 && n <= 9 && !e.shiftKey) {
        setSelectedCardIndex(n - 1 < cards.length ? n - 1 : null);
        return;
      }

      if (e.shiftKey && !isNaN(n) && n >= 1 && n <= 9) {
        const card = cards[n - 1];
        if (card) {
          pushAlphaSnapshot();
          updateCardAlpha(card.concept, 0);
          void cardsApi.setAlpha(card.concept, 0);
        }
        return;
      }

      if (e.key === "Tab") {
        e.preventDefault();
        if (cards.length === 0) return;
        const next = selectedCardIndex === null ? 0 : (selectedCardIndex + 1) % cards.length;
        setSelectedCardIndex(next);
        return;
      }

      if (e.key === "h" || e.key === "H") {
        toggleTrajectory();
        return;
      }

      if (e.key === "g" || e.key === "G") {
        toggleGrid();
        return;
      }

      if ((e.key === "r" || e.key === "R") && !isGenerating) {
        if (prompt.trim()) generate(prompt.trim());
        return;
      }

      if (e.key === " ") {
        if (isGenerating) {
          e.preventDefault();
          stop();
        }
        return;
      }

      if (selectedCardIndex !== null && cards[selectedCardIndex]) {
        const card = cards[selectedCardIndex];
        const step = e.shiftKey ? 0.5 : 0.1;
        const coarseStep = 1.0;

        if (e.key === "ArrowUp") {
          e.preventDefault();
          pushAlphaSnapshot();
          const newAlpha = Math.min(3, card.alpha + step);
          updateCardAlpha(card.concept, newAlpha);
          void cardsApi.setAlpha(card.concept, newAlpha);
          return;
        }

        if (e.key === "ArrowDown") {
          e.preventDefault();
          pushAlphaSnapshot();
          const newAlpha = Math.max(-3, card.alpha - step);
          updateCardAlpha(card.concept, newAlpha);
          void cardsApi.setAlpha(card.concept, newAlpha);
          return;
        }

        if (e.key === "]") {
          e.preventDefault();
          pushAlphaSnapshot();
          const newAlpha = Math.min(3, card.alpha + coarseStep);
          updateCardAlpha(card.concept, newAlpha);
          void cardsApi.setAlpha(card.concept, newAlpha);
          return;
        }

        if (e.key === "[") {
          e.preventDefault();
          pushAlphaSnapshot();
          const newAlpha = Math.max(-3, card.alpha - coarseStep);
          updateCardAlpha(card.concept, newAlpha);
          void cardsApi.setAlpha(card.concept, newAlpha);
          return;
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    cards,
    selectedCardIndex,
    setSelectedCardIndex,
    updateCardAlpha,
    pushAlphaSnapshot,
    isGenerating,
    toggleTrajectory,
    toggleGrid,
    resetCanvasViewport,
    generate,
    stop,
    prompt,
  ]);
}
