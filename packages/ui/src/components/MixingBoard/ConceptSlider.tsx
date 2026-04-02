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
  isSelected?: boolean;
}

function ConceptTooltipContent({ card, absAlpha }: { card: CardInfo; absAlpha: number }) {
  const perturbPct =
    card.mean_hidden_norm != null
      ? `${((absAlpha / card.mean_hidden_norm) * 100).toFixed(1)}%`
      : null;

  const rows: [string, string | number | null][] = [
    ["layer", card.layer],
    ["hidden dim", card.hidden_dim],
    ["model", card.model_id],
    ["α perturbation", perturbPct],
    ["p-value", card.p_value != null ? card.p_value.toFixed(4) : null],
    ["separability", card.separability_score != null ? card.separability_score.toFixed(3) : null],
    ["positives", card.num_positive],
    ["negatives", card.num_negative],
    ["bootstrap var", card.bootstrap_variance != null ? card.bootstrap_variance.toFixed(5) : null],
  ];

  return (
    <div className="flex flex-col gap-0.5 min-w-[160px]">
      {card.description && (
        <p className="font-mono text-[11px] text-text-muted mb-1 border-b border-bg-border pb-1">
          {card.description}
        </p>
      )}
      {rows.map(([label, value]) =>
        value != null ? (
          <div key={label} className="flex items-center justify-between gap-4">
            <span className="font-mono text-[10px] text-text-subtle">{label}</span>
            <span className="font-mono text-[10px] text-text tabular-nums">{value}</span>
          </div>
        ) : null
      )}
    </div>
  );
}

export function ConceptSlider({ card, isSelected }: Props) {
  const updateCardAlpha = useStore((s) => s.updateCardAlpha);
  const removeCard = useStore((s) => s.removeCard);
  const pushAlphaSnapshot = useStore((s) => s.pushAlphaSnapshot);
  const recomputeDisplayColors = useStore((s) => s.recomputeDisplayColors);
  const recordEvent = useStore((s) => s.recordEvent);
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

  const theme = useStore((s) => s.theme);
  const conceptColors = useStore((s) => s.conceptColors);
  const color = getConceptColor(card.concept, theme, conceptColors);
  const absAlpha = Math.abs(card.alpha);
  const glowIntensity = Math.min(absAlpha / ALPHA_MAX, 1);
  const isDanger =
    card.mean_hidden_norm != null ? absAlpha / card.mean_hidden_norm > 0.15 : absAlpha > 2.5;

  return (
    <Tooltip.Provider delayDuration={400}>
      <div
        className={`group flex flex-col gap-1.5 rounded-lg border bg-bg-surface p-2.5 transition-all hover:border-bg-elevated hover:scale-[1.02]${isSelected ? " border-accent/50" : " border-bg-border"}`}
        style={{
          boxShadow: glowIntensity > 0.2 ? `0 0 ${8 + glowIntensity * 12}px ${isDanger ? "color-mix(in srgb, var(--color-danger) 13%, transparent)" : color + "22"}` : undefined,
        }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <motion.span
              className="h-2 w-2 rounded-full shrink-0"
              animate={{
                boxShadow: `0 0 ${4 + glowIntensity * 8}px ${isDanger ? "var(--color-danger)" : color}`,
              }}
              transition={{ duration: 0.3 }}
              style={{ backgroundColor: isDanger ? "var(--color-danger)" : color }}
            />
            <Tooltip.Root>
              <Tooltip.Trigger asChild>
                <span className="font-mono text-sm font-medium text-text cursor-help">{card.concept}</span>
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content
                  className="z-50 rounded-lg border border-bg-border bg-bg-elevated px-3 py-2 shadow-xl"
                  sideOffset={6}
                >
                  <ConceptTooltipContent card={card} absAlpha={absAlpha} />
                  <Tooltip.Arrow className="fill-bg-border" />
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
            <span className="font-mono text-xs text-text-muted">L{card.layer}</span>
          </div>
          <button
            onClick={() => unloadMutation.mutate()}
            disabled={unloadMutation.isPending}
            className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 text-text-subtle hover:text-danger"
          >
            <X size={14} />
          </button>
        </div>

        <div className="flex items-center gap-3">
          <motion.span
            className="w-7 font-mono text-xs text-right tabular-nums"
            animate={{ color: absAlpha > 0.5 ? (isDanger ? "var(--color-danger)" : color) : "var(--color-text-subtle)" }}
            transition={{ duration: 0.2 }}
          >
            {card.alpha.toFixed(1)}
          </motion.span>
          <div className="relative flex-1">
            <div
              className="absolute top-1/2 -translate-y-1/2 h-full rounded pointer-events-none"
              style={{
                left: "calc(50% - 1px)",
                width: 1,
                height: 10,
                backgroundColor: "var(--color-text-subtle)",
                zIndex: 1,
              }}
            />
            {absAlpha > 2.0 && (
              <div
                className="absolute top-1/2 -translate-y-1/2 rounded pointer-events-none"
                style={{
                  left: card.alpha > 0 ? `calc(${(2.0 / ALPHA_MAX) * 50 + 50}%)` : 0,
                  right: card.alpha < 0 ? `calc(${(2.0 / ALPHA_MAX) * 50 + 50}%)` : 0,
                  height: 4,
                  background: `linear-gradient(${card.alpha > 0 ? "to right" : "to left"}, color-mix(in srgb, var(--color-warn) 38%, transparent), color-mix(in srgb, var(--color-danger) 38%, transparent))`,
                  zIndex: 0,
                  borderRadius: 2,
                }}
              />
            )}
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
                  boxShadow: glowIntensity > 0.1 ? `0 0 ${4 + glowIntensity * 8}px ${isDanger ? "color-mix(in srgb, var(--color-danger) 27%, transparent)" : color + "44"}` : undefined,
                }}
              >
                <RadixSlider.Range
                  className="absolute h-full rounded-full"
                  style={{ backgroundColor: isDanger ? "var(--color-danger)" : color, opacity: 0.4 + glowIntensity * 0.4 }}
                />
              </RadixSlider.Track>
              <RadixSlider.Thumb
                className={cn(
                  "block h-4 w-4 rounded-full border-2 shadow-glow-sm",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-1",
                  "transition-transform hover:scale-110"
                )}
                style={{
                  backgroundColor: isDanger ? "var(--color-danger)" : color,
                  borderColor: isDanger ? "var(--color-danger)" : color,
                  boxShadow: `0 0 ${6 + glowIntensity * 10}px ${isDanger ? "var(--color-danger)" : color}`,
                }}
              />
            </RadixSlider.Root>
          </div>
          <div className="flex gap-1 text-xs font-mono text-text-subtle">
            <span>{ALPHA_MIN}</span>
            <span className="text-bg-border">/</span>
            <span>{ALPHA_MAX}</span>
          </div>
        </div>
      </div>
    </Tooltip.Provider>
  );
}
