import { useCallback, useRef } from "react";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";
import { cardsApi } from "../../api/cards";

const ALPHA_MIN = -3;
const ALPHA_MAX = 3;
const ALPHA_RANGE = ALPHA_MAX - ALPHA_MIN;

export function ConceptPad() {
  const cards = useStore((s) => s.cards);
  const updateCardAlpha = useStore((s) => s.updateCardAlpha);
  const pushAlphaSnapshot = useStore((s) => s.pushAlphaSnapshot);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  const concept1 = cards[0];
  const concept2 = cards[1];

  const persistAlphas = useCallback(
    (c1: string, a1: number, c2: string, a2: number) => {
      abortRef.current?.abort();
      abortRef.current = new AbortController();
      const signal = abortRef.current.signal;
      setTimeout(() => {
        if (!signal.aborted) {
          void cardsApi.setAlpha(c1, a1);
          void cardsApi.setAlpha(c2, a2);
        }
      }, 300);
    },
    []
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!isDragging.current || !containerRef.current || !concept1 || !concept2) return;
      const rect = containerRef.current.getBoundingClientRect();
      const relX = (e.clientX - rect.left) / rect.width;
      const relY = (e.clientY - rect.top) / rect.height;
      const alpha2 = ALPHA_MIN + relX * ALPHA_RANGE;
      const alpha1 = ALPHA_MAX - relY * ALPHA_RANGE;
      const clamped1 = Math.max(ALPHA_MIN, Math.min(ALPHA_MAX, alpha1));
      const clamped2 = Math.max(ALPHA_MIN, Math.min(ALPHA_MAX, alpha2));
      updateCardAlpha(concept1.concept, clamped1);
      updateCardAlpha(concept2.concept, clamped2);
      persistAlphas(concept1.concept, clamped1, concept2.concept, clamped2);
    },
    [concept1, concept2, updateCardAlpha, persistAlphas]
  );

  if (!concept1 || !concept2) return null;

  const color1 = getConceptColor(concept1.concept);
  const color2 = getConceptColor(concept2.concept);

  const relX = (concept2.alpha - ALPHA_MIN) / ALPHA_RANGE;
  const relY = 1 - (concept1.alpha - ALPHA_MIN) / ALPHA_RANGE;
  const dotX = `${Math.max(0, Math.min(100, relX * 100))}%`;
  const dotY = `${Math.max(0, Math.min(100, relY * 100))}%`;

  return (
    <div className="flex flex-col h-full items-center justify-center gap-4 p-4">
      <div
        ref={containerRef}
        className="relative w-full flex-1 max-h-64 rounded-lg border border-bg-border bg-bg-surface cursor-crosshair select-none overflow-hidden"
        onMouseDown={() => {
          isDragging.current = true;
          pushAlphaSnapshot();
        }}
        onMouseMove={handleMouseMove}
        onMouseUp={() => { isDragging.current = false; }}
        onMouseLeave={() => { isDragging.current = false; }}
      >
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-bg-border" />
        <div className="absolute top-1/2 left-0 right-0 h-px bg-bg-border" />

        <span
          className="absolute top-2 left-1/2 -translate-x-1/2 font-mono text-[10px] pointer-events-none"
          style={{ color: color1 }}
        >
          +{concept1.concept}
        </span>
        <span
          className="absolute bottom-2 left-1/2 -translate-x-1/2 font-mono text-[10px] pointer-events-none"
          style={{ color: color1, opacity: 0.5 }}
        >
          -{concept1.concept}
        </span>
        <span
          className="absolute right-2 top-1/2 -translate-y-1/2 font-mono text-[10px] pointer-events-none"
          style={{ color: color2 }}
        >
          +{concept2.concept}
        </span>
        <span
          className="absolute left-2 top-1/2 -translate-y-1/2 font-mono text-[10px] pointer-events-none"
          style={{ color: color2, opacity: 0.5 }}
        >
          -{concept2.concept}
        </span>

        <div
          className="absolute w-5 h-5 rounded-full -translate-x-1/2 -translate-y-1/2 pointer-events-none"
          style={{
            left: dotX,
            top: dotY,
            background: `radial-gradient(circle, ${color1}, ${color2})`,
            boxShadow: `0 0 12px ${color1}88, 0 0 6px ${color2}66`,
          }}
        />
      </div>

      <div className="flex items-center gap-6 font-mono text-xs text-text-muted">
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color1 }} />
          <span>{concept1.concept}</span>
          <span className="tabular-nums" style={{ color: color1 }}>
            {concept1.alpha >= 0 ? "+" : ""}{concept1.alpha.toFixed(1)}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color2 }} />
          <span>{concept2.concept}</span>
          <span className="tabular-nums" style={{ color: color2 }}>
            {concept2.alpha >= 0 ? "+" : ""}{concept2.alpha.toFixed(1)}
          </span>
        </div>
      </div>

      <p className="font-mono text-[10px] text-text-subtle text-center">
        drag to steer both concepts simultaneously
      </p>
    </div>
  );
}
