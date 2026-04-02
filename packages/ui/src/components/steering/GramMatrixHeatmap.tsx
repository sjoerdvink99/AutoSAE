import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";

function valueToColor(val: number, theme: "dark" | "light"): string {
  const clamped = Math.max(-1, Math.min(1, val));
  if (clamped >= 0) {
    const alpha = Math.round(clamped * 200);
    if (theme === "light") return `rgba(5,150,105,${(alpha / 255).toFixed(2)})`;
    return `rgba(0,230,118,${(alpha / 255).toFixed(2)})`;
  } else {
    const alpha = Math.round(-clamped * 200);
    if (theme === "light") return `rgba(220,38,38,${(alpha / 255).toFixed(2)})`;
    return `rgba(255,23,68,${(alpha / 255).toFixed(2)})`;
  }
}

export function GramMatrixHeatmap() {
  const geometry = useStore((s) => s.geometry);
  const theme = useStore((s) => s.theme);
  const conceptColors = useStore((s) => s.conceptColors);

  if (!geometry || geometry.concepts.length < 2) return null;

  const { gram, concepts } = geometry;

  return (
    <div className="flex flex-col gap-2 px-2 py-2 border-t border-bg-border">
      <span className="font-mono text-[10px] text-text-subtle uppercase tracking-wider">
        Gram Matrix
      </span>
      <div className="flex gap-1.5">
        <div className="flex flex-col gap-px">
          {concepts.map((c) => (
            <div key={c} className="h-6 flex items-center">
              <span
                className="font-mono text-[10px] w-14 truncate text-right pr-1"
                style={{ color: getConceptColor(c, theme, conceptColors) }}
              >
                {c}
              </span>
            </div>
          ))}
        </div>
        <div className="flex flex-col gap-px flex-1">
          {gram.map((row, i) => (
            <div key={i} className="flex gap-px h-6">
              {row.map((val, j) => {
                const isCollinear = i !== j && Math.abs(val) > 0.8;
                return (
                  <div
                    key={j}
                    className="flex-1 rounded-sm flex items-center justify-center relative"
                    style={{ backgroundColor: valueToColor(val, theme) }}
                    title={isCollinear ? `${concepts[i]} × ${concepts[j]}: ${val.toFixed(3)} — Near-collinear concepts — steering may be redundant.` : `${concepts[i]} × ${concepts[j]}: ${val.toFixed(3)}`}
                  >
                    <span
                      className="font-mono text-[9px]"
                      style={{ color: Math.abs(val) > 0.3 ? "rgba(255,255,255,0.7)" : "var(--color-text)" }}
                    >
                      {val.toFixed(2)}
                    </span>
                    {isCollinear && (
                      <span className="absolute top-0 right-0 font-mono text-[7px] font-bold text-yellow-400 leading-none px-px">
                        !
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
