import { useMemo } from "react";
import { useStore } from "../../stores/useStore";
import { cardsApi } from "../../api/cards";

const HEIGHT = 32;
const BAR_W = 4;
const BAR_GAP = 2;

export function AlphaTimeline() {
  const alphaHistory = useStore((s) => s.alphaHistory);
  const restoreAlphaSnapshot = useStore((s) => s.restoreAlphaSnapshot);
  const updateCardAlpha = useStore((s) => s.updateCardAlpha);

  const bars = useMemo(() => {
    return alphaHistory.map((snapshot) => ({
      total: Object.values(snapshot.alphas).reduce((s, a) => s + Math.abs(a), 0),
      snapshot,
    }));
  }, [alphaHistory]);

  if (alphaHistory.length < 2) return null;

  const maxTotal = Math.max(...bars.map((b) => b.total), 0.01);
  const totalW = bars.length * (BAR_W + BAR_GAP);

  const handleRestore = (index: number) => {
    restoreAlphaSnapshot(index);
    const snapshot = alphaHistory[index];
    if (!snapshot) return;
    for (const [concept, alpha] of Object.entries(snapshot.alphas)) {
      updateCardAlpha(concept, alpha);
      void cardsApi.setAlpha(concept, alpha);
    }
  };

  return (
    <div className="border-t border-bg-border px-3 py-2 shrink-0">
      <div className="flex items-center gap-2 mb-1">
        <span className="font-mono text-[10px] text-text-subtle">history</span>
        <span className="font-mono text-[10px] text-text-subtle ml-auto">
          {alphaHistory.length}
        </span>
      </div>
      <svg width={Math.min(totalW, 240)} height={HEIGHT} style={{ display: "block" }}>
        {bars.map((bar, i) => {
          const h = Math.max(2, (bar.total / maxTotal) * (HEIGHT - 4));
          const x = i * (BAR_W + BAR_GAP);
          const isCurrent = i === alphaHistory.length - 1;
          return (
            <rect
              key={i}
              x={x}
              y={HEIGHT - h}
              width={BAR_W}
              height={h}
              fill={isCurrent ? "var(--color-accent)" : "var(--color-bg-border)"}
              fillOpacity={isCurrent ? 0.9 : 0.5}
              rx={1}
              style={{ cursor: "pointer" }}
              onClick={() => handleRestore(i)}
            />
          );
        })}
      </svg>
    </div>
  );
}
