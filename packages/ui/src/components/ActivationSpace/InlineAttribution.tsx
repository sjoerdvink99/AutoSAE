import { useState } from "react";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";
import { ConceptDot } from "../ui/ConceptDot";

interface TokenAnnotationProps {
  token: string;
  activations: Record<string, number>;
  activeFilter: string | null;
  showTopOnly: boolean;
  theme: "dark" | "light";
}

function TokenAnnotation({ token, activations, activeFilter, showTopOnly, theme }: TokenAnnotationProps) {
  const [hovered, setHovered] = useState(false);

  let dominantConcept: string | null = null;
  let dominantVal = showTopOnly ? 0.15 : 0.05;

  for (const [concept, val] of Object.entries(activations)) {
    if (activeFilter && concept !== activeFilter) continue;
    const absVal = Math.abs(val);
    if (absVal > dominantVal) {
      dominantVal = absVal;
      dominantConcept = concept;
    }
  }

  const color = dominantConcept ? getConceptColor(dominantConcept, theme) : null;
  const opacity = dominantConcept && color
    ? Math.min(Math.abs(activations[dominantConcept] ?? 0) * 1.2, 1)
    : 0;

  const entries = Object.entries(activations).sort(([, a], [, b]) => Math.abs(b) - Math.abs(a));

  return (
    <span className="relative inline">
      <span
        className="transition-colors duration-150"
        style={{
          borderBottom: color ? `2px solid ${color}` : undefined,
          borderBottomColor: color ? `color-mix(in srgb, ${color} ${Math.round(opacity * 100)}%, transparent)` : undefined,
          paddingBottom: "1px",
          cursor: dominantConcept ? "default" : undefined,
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {token}
      </span>
      {hovered && entries.length > 0 && (
        <span
          className="absolute z-50 bottom-full left-1/2 mb-1 pointer-events-none"
          style={{ transform: "translateX(-50%)" }}
        >
          <span className="block bg-bg-elevated border border-bg-border rounded-lg shadow-xl px-3 py-2 min-w-[140px] whitespace-nowrap">
            <span className="block text-[10px] uppercase tracking-wider text-text-subtle mb-1.5 font-medium">
              Activations
            </span>
            {entries.map(([concept, value]) => {
              const c = getConceptColor(concept, theme);
              const isActive = Math.abs(value) > 0.25;
              return (
                <span key={concept} className="flex items-center justify-between gap-3 mb-0.5">
                  <span className="flex items-center gap-1.5">
                    <ConceptDot color={c} glow={isActive} />
                    <span className="font-mono text-xs text-text-muted">{concept}</span>
                  </span>
                  <span
                    className="font-mono text-xs tabular-nums font-medium"
                    style={{ color: isActive ? c : "var(--color-text-subtle)" }}
                  >
                    {value >= 0 ? "+" : ""}
                    {value.toFixed(2)}
                  </span>
                </span>
              );
            })}
          </span>
        </span>
      )}
    </span>
  );
}

export function InlineAttribution() {
  const tokens = useStore((s) => s.tokens);
  const cards = useStore((s) => s.cards);
  const theme = useStore((s) => s.theme);
  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const [showTopOnly, setShowTopOnly] = useState(false);

  if (tokens.length === 0 || cards.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="font-mono text-xs text-text-subtle">
          {cards.length === 0
            ? "load a concept card to see attribution"
            : "generate text to see attribution"}
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 h-full min-h-0">
      <div className="flex items-center gap-3 flex-wrap shrink-0">
        <div className="flex items-center gap-2 flex-wrap">
          {cards.map((card) => {
            const color = getConceptColor(card.concept, theme);
            const isActive = activeFilter === card.concept;
            return (
              <button
                key={card.concept}
                onClick={() => setActiveFilter(isActive ? null : card.concept)}
                className="flex items-center gap-1.5 px-2 py-1 rounded font-mono text-xs transition-colors"
                style={{
                  backgroundColor: isActive ? `${color}22` : undefined,
                  border: `1px solid ${isActive ? color : "var(--color-bg-border)"}`,
                  color: isActive ? color : "var(--color-text-muted)",
                }}
              >
                <ConceptDot color={color} glow={isActive} />
                {card.concept}
              </button>
            );
          })}
        </div>
        <label className="flex items-center gap-1.5 font-mono text-xs text-text-muted cursor-pointer ml-auto">
          <input
            type="checkbox"
            checked={showTopOnly}
            onChange={(e) => setShowTopOnly(e.target.checked)}
            className="accent-accent"
          />
          Show top only
        </label>
      </div>

      <div className="overflow-y-auto flex-1 font-mono text-sm leading-relaxed text-text">
        {tokens.map((chunk, i) => (
          <TokenAnnotation
            key={i}
            token={chunk.token}
            activations={chunk.activations}
            activeFilter={activeFilter}
            showTopOnly={showTopOnly}
            theme={theme}
          />
        ))}
      </div>
    </div>
  );
}
