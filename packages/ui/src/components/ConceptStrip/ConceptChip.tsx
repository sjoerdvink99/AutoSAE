import { useCallback, useRef } from "react";
import * as RadixSlider from "@radix-ui/react-slider";
import * as Tooltip from "@radix-ui/react-tooltip";
import { X } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cardsApi } from "../../api/cards";
import { useStore } from "../../stores/useStore";
import { cn } from "../../lib/cn";
import { getConceptColor } from "../../lib/constants";
import type { CardInfo } from "../../types";

const ALPHA_MIN = -10;
const ALPHA_MAX = 10;
const DEBOUNCE_MS = 300;

interface Props {
  card: CardInfo;
  isSelected: boolean;
}

export function ConceptChip({ card, isSelected }: Props) {
  const updateCardAlpha = useStore((s) => s.updateCardAlpha);
  const removeCard = useStore((s) => s.removeCard);
  const pushAlphaSnapshot = useStore((s) => s.pushAlphaSnapshot);
  const recomputeDisplayColors = useStore((s) => s.recomputeDisplayColors);
  const recordEvent = useStore((s) => s.recordEvent);
  const conceptColors = useStore((s) => s.conceptColors);
  const theme = useStore((s) => s.theme);
  const queryClient = useQueryClient();
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const alphaMutation = useMutation({
    mutationFn: (alpha: number) => cardsApi.setAlpha(card.concept, alpha),
  });

  const unloadMutation = useMutation({
    mutationFn: () => cardsApi.unload(card.concept),
    onSuccess: () => {
      removeCard(card.concept);
      queryClient.invalidateQueries({ queryKey: ["cards"] });
      queryClient.invalidateQueries({ queryKey: ["concept-map"] });
    },
  });

  const handleAlphaChange = useCallback(
    (values: number[]) => {
      const alpha = values[0] ?? card.alpha;
      updateCardAlpha(card.concept, alpha);
      recomputeDisplayColors();
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        alphaMutation.mutate(alpha);
      }, DEBOUNCE_MS);
    },
    [card.alpha, card.concept, updateCardAlpha, recomputeDisplayColors, alphaMutation]
  );

  const handleAlphaCommit = useCallback(() => {
    pushAlphaSnapshot();
    recordEvent({ type: "alpha_change", payload: { concept: card.concept, alpha: card.alpha } });
  }, [pushAlphaSnapshot, recordEvent, card.concept, card.alpha]);

  const color = getConceptColor(card.concept, theme, conceptColors);
  const absAlpha = Math.abs(card.alpha);
  const glowIntensity = Math.min(absAlpha / ALPHA_MAX, 1);
  const isDanger =
    card.mean_hidden_norm != null ? absAlpha / card.mean_hidden_norm > 0.15 : absAlpha > 2.5;
  const activeColor = isDanger ? "var(--color-danger)" : color;

  return (
    <Tooltip.Provider delayDuration={400}>
      <div
        className={cn(
          "group flex items-center gap-2 rounded-lg border px-2.5 py-1.5 transition-all shrink-0",
          isSelected ? "border-accent/40" : "border-bg-border hover:border-bg-elevated"
        )}
        style={{
          backgroundColor: "var(--color-bg-surface)",
          minWidth: 190,
          maxWidth: 260,
          boxShadow:
            glowIntensity > 0.2
              ? `0 0 ${6 + glowIntensity * 10}px ${isDanger ? "color-mix(in srgb, var(--color-danger) 12%, transparent)" : color + "1a"}`
              : undefined,
        }}
      >
        <motion.span
          className="h-2 w-2 rounded-full shrink-0"
          animate={{ boxShadow: `0 0 ${3 + glowIntensity * 7}px ${activeColor}` }}
          transition={{ duration: 0.3 }}
          style={{ backgroundColor: activeColor }}
        />

        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <span className="font-mono text-xs font-medium text-text truncate cursor-help" style={{ maxWidth: 72 }}>
              {card.concept}
            </span>
          </Tooltip.Trigger>
          <Tooltip.Portal>
            <Tooltip.Content
              className="z-50 rounded-lg border border-bg-border bg-bg-elevated px-3 py-2 shadow-xl"
              sideOffset={8}
            >
              <div className="flex flex-col gap-0.5 min-w-[140px]">
                {card.description && (
                  <p className="font-mono text-[11px] text-text-muted mb-1 border-b border-bg-border pb-1">
                    {card.description}
                  </p>
                )}
                {[
                  ["layer", card.layer],
                  ["model", card.model_id],
                  ["p-value", card.p_value != null ? card.p_value.toFixed(4) : null],
                  ["separability", card.separability_score != null ? card.separability_score.toFixed(3) : null],
                ].map(([label, value]) =>
                  value != null ? (
                    <div key={String(label)} className="flex items-center justify-between gap-4">
                      <span className="font-mono text-[10px] text-text-subtle">{label}</span>
                      <span className="font-mono text-[10px] text-text tabular-nums">{value}</span>
                    </div>
                  ) : null
                )}
              </div>
              <Tooltip.Arrow className="fill-bg-border" />
            </Tooltip.Content>
          </Tooltip.Portal>
        </Tooltip.Root>

        <motion.span
          className="font-mono text-[11px] tabular-nums shrink-0 w-7 text-right"
          animate={{ color: absAlpha > 0.5 ? activeColor : "var(--color-text-subtle)" }}
          transition={{ duration: 0.2 }}
        >
          {card.alpha.toFixed(1)}
        </motion.span>

        <div className="relative flex-1" style={{ minWidth: 72 }}>
          <div
            className="absolute top-1/2 -translate-y-1/2 pointer-events-none"
            style={{ left: "50%", width: 1, height: 10, backgroundColor: "var(--color-text-subtle)", zIndex: 1 }}
          />
          <RadixSlider.Root
            className="relative flex h-5 w-full touch-none select-none items-center"
            min={ALPHA_MIN}
            max={ALPHA_MAX}
            step={0.1}
            value={[card.alpha]}
            onValueChange={handleAlphaChange}
            onValueCommit={handleAlphaCommit}
          >
            <RadixSlider.Track
              className="relative h-px flex-grow rounded-full bg-bg-border"
              style={{
                boxShadow:
                  glowIntensity > 0.1
                    ? `0 0 ${3 + glowIntensity * 6}px ${isDanger ? "color-mix(in srgb, var(--color-danger) 22%, transparent)" : color + "33"}`
                    : undefined,
              }}
            >
              <RadixSlider.Range
                className="absolute h-full rounded-full"
                style={{ backgroundColor: activeColor, opacity: 0.4 + glowIntensity * 0.4 }}
              />
            </RadixSlider.Track>
            <RadixSlider.Thumb
              className={cn(
                "block h-3.5 w-3.5 rounded-full border-2",
                "focus-visible:outline-none",
                "transition-transform hover:scale-110"
              )}
              style={{
                backgroundColor: activeColor,
                borderColor: activeColor,
                boxShadow: `0 0 ${5 + glowIntensity * 8}px ${activeColor}`,
              }}
            />
          </RadixSlider.Root>
        </div>

        <button
          onClick={() => unloadMutation.mutate()}
          disabled={unloadMutation.isPending}
          className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 text-text-subtle hover:text-danger shrink-0"
        >
          <X size={11} />
        </button>
      </div>
    </Tooltip.Provider>
  );
}
