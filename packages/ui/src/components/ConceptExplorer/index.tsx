import { useEffect, useRef, useState, useCallback } from "react";
import * as d3 from "d3";
import * as Dialog from "@radix-ui/react-dialog";
import { X, RefreshCw } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { cardsApi } from "../../api/cards";
import { useConceptMap } from "../../hooks/useConceptMap";
import { useStore } from "../../stores/useStore";
import { getConceptColor } from "../../lib/constants";
import type { ConceptMapPoint } from "../../types";

const W = 560;
const H = 380;
const PAD = 48;

interface Props {
  open: boolean;
  onClose: () => void;
}

function ScatterPlot({
  points,
  conceptColors,
  theme,
  loadedSet,
  onToggle,
  pendingConcept,
}: {
  points: ConceptMapPoint[];
  conceptColors: Record<string, string>;
  theme: "dark" | "light";
  loadedSet: Set<string>;
  onToggle: (pt: ConceptMapPoint) => void;
  pendingConcept: string | null;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hovered, setHovered] = useState<ConceptMapPoint | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  const xExt = d3.extent(points, (d) => d.x) as [number, number];
  const yExt = d3.extent(points, (d) => d.y) as [number, number];
  const pad = 0.15;
  const xScale = d3.scaleLinear([xExt[0] - pad, xExt[1] + pad], [PAD, W - PAD]);
  const yScale = d3.scaleLinear([yExt[0] - pad, yExt[1] + pad], [H - PAD, PAD]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const el = d3.select(svg);
    el.selectAll(".grid-x")
      .data(xScale.ticks(6))
      .join("line")
      .attr("class", "grid-x")
      .attr("x1", (d) => xScale(d))
      .attr("x2", (d) => xScale(d))
      .attr("y1", PAD)
      .attr("y2", H - PAD)
      .attr("stroke", "var(--color-grid)")
      .attr("stroke-width", 1);

    el.selectAll(".grid-y")
      .data(yScale.ticks(5))
      .join("line")
      .attr("class", "grid-y")
      .attr("x1", PAD)
      .attr("x2", W - PAD)
      .attr("y1", (d) => yScale(d))
      .attr("y2", (d) => yScale(d))
      .attr("stroke", "var(--color-grid)")
      .attr("stroke-width", 1);
  }, [xScale, yScale]);

  return (
    <div className="relative" style={{ width: W, height: H }}>
      <svg
        ref={svgRef}
        width={W}
        height={H}
        style={{ display: "block", overflow: "visible" }}
      >
        <line x1={xScale(0)} y1={PAD} x2={xScale(0)} y2={H - PAD} stroke="var(--color-bg-border)" strokeWidth={1} />
        <line x1={PAD} y1={yScale(0)} x2={W - PAD} y2={yScale(0)} stroke="var(--color-bg-border)" strokeWidth={1} />

        {points.map((pt) => {
          const cx = xScale(pt.x);
          const cy = yScale(pt.y);
          const isLoaded = loadedSet.has(pt.concept);
          const isPending = pendingConcept === pt.concept;
          const color = isLoaded
            ? getConceptColor(pt.concept, theme, conceptColors)
            : "var(--color-text-subtle)";
          const r = 5 + (pt.separability_score ?? 0.5) * 5;

          return (
            <g
              key={pt.concept}
              style={{ cursor: isPending ? "wait" : "pointer" }}
              onClick={() => !isPending && onToggle(pt)}
              onMouseEnter={(e) => {
                setHovered(pt);
                const rect = svgRef.current?.getBoundingClientRect();
                if (rect) {
                  setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
                }
              }}
              onMouseLeave={() => setHovered(null)}
            >
              <circle
                cx={cx}
                cy={cy}
                r={r + 4}
                fill="transparent"
              />
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={color}
                fillOpacity={isLoaded ? 0.85 : 0.28}
                stroke={isLoaded ? color : "var(--color-bg-border)"}
                strokeWidth={isLoaded ? 2 : 1}
                strokeDasharray={isLoaded ? "none" : "3 2"}
                style={{
                  filter: isLoaded ? `drop-shadow(0 0 4px ${color}88)` : "none",
                  transition: "all 0.2s",
                }}
              />
              {isPending && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={r + 5}
                  fill="none"
                  stroke="var(--color-accent)"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  style={{ animation: "spin 1s linear infinite", transformOrigin: `${cx}px ${cy}px` }}
                />
              )}
              <text
                x={cx + r + 5}
                y={cy + 4}
                fill={isLoaded ? color : "var(--color-text-subtle)"}
                fontSize={10}
                fontFamily="monospace"
                fontWeight={isLoaded ? "600" : "400"}
              >
                {pt.concept}
              </text>
            </g>
          );
        })}
      </svg>

      <AnimatePresence>
        {hovered && (
          <motion.div
            key="tooltip"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.1 }}
            className="absolute pointer-events-none z-20 rounded-lg border border-bg-border bg-bg-elevated px-3 py-2 shadow-xl"
            style={{
              left: Math.min(tooltipPos.x + 12, W - 160),
              top: Math.max(tooltipPos.y - 48, 4),
            }}
          >
            <div className="font-mono text-xs font-semibold text-text mb-1">{hovered.concept}</div>
            <div className="flex flex-col gap-0.5">
              {[
                ["status", hovered.loaded ? "loaded" : "available"],
                ["separability", hovered.separability_score != null ? hovered.separability_score.toFixed(3) : "—"],
                ["model", hovered.model_id ?? "—"],
              ].map(([label, value]) => (
                <div key={label} className="flex items-center justify-between gap-4">
                  <span className="font-mono text-[10px] text-text-subtle">{label}</span>
                  <span
                    className="font-mono text-[10px] text-text tabular-nums"
                    style={{ color: label === "status" && hovered.loaded ? "var(--color-accent)" : undefined }}
                  >
                    {value}
                  </span>
                </div>
              ))}
            </div>
            <p className="font-mono text-[9px] text-text-muted mt-1.5 border-t border-bg-border pt-1">
              {hovered.loaded ? "Click to unload" : "Click to load"}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function ConceptExplorer({ open, onClose }: Props) {
  const { data: mapData, isLoading, refetch } = useConceptMap(open);
  const conceptColors = useStore((s) => s.conceptColors);
  const theme = useStore((s) => s.theme);
  const cards = useStore((s) => s.cards);
  const setCards = useStore((s) => s.setCards);
  const queryClient = useQueryClient();
  const [pendingConcept, setPendingConcept] = useState<string | null>(null);
  const loadedSet = new Set(cards.map((c) => c.concept));

  const loadMutation = useMutation({
    mutationFn: (concept: string) =>
      cardsApi.load({ registry_concept: concept, registry_model: "llama-3.1-8b" }),
    onSuccess: async () => {
      const result = await cardsApi.list();
      setCards(result);
      queryClient.invalidateQueries({ queryKey: ["cards"] });
      queryClient.invalidateQueries({ queryKey: ["concept-map"] });
      setPendingConcept(null);
    },
    onError: () => setPendingConcept(null),
  });

  const unloadMutation = useMutation({
    mutationFn: (concept: string) => cardsApi.unload(concept),
    onSuccess: async () => {
      const result = await cardsApi.list();
      setCards(result);
      queryClient.invalidateQueries({ queryKey: ["cards"] });
      queryClient.invalidateQueries({ queryKey: ["concept-map"] });
      setPendingConcept(null);
    },
    onError: () => setPendingConcept(null),
  });

  const handleToggle = useCallback(
    (pt: ConceptMapPoint) => {
      setPendingConcept(pt.concept);
      if (pt.loaded || loadedSet.has(pt.concept)) {
        unloadMutation.mutate(pt.concept);
      } else {
        loadMutation.mutate(pt.concept);
      }
    },
    [loadedSet, loadMutation, unloadMutation]
  );

  const points = mapData?.points ?? [];

  return (
    <Dialog.Root open={open} onOpenChange={(v) => !v && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm" />
        <Dialog.Content
          className="fixed z-50 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-xl border border-bg-border bg-bg shadow-2xl outline-none"
          style={{ width: W + 48, maxWidth: "95vw" }}
        >
          <div className="flex items-center justify-between px-5 py-3 border-b border-bg-border">
            <div>
              <h2 className="font-mono text-sm font-semibold text-text">Concept Map</h2>
              <p className="font-mono text-[11px] text-text-subtle mt-0.5">
                PCA of concept vectors — size = separability · click to load / unload
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => refetch()}
                className="p-1.5 rounded border border-bg-border text-text-subtle hover:text-text hover:border-bg-elevated transition-colors"
                title="Refresh"
              >
                <RefreshCw size={13} />
              </button>
              <Dialog.Close asChild>
                <button
                  onClick={onClose}
                  className="p-1.5 rounded border border-bg-border text-text-subtle hover:text-text hover:border-bg-elevated transition-colors"
                >
                  <X size={13} />
                </button>
              </Dialog.Close>
            </div>
          </div>

          <div className="p-4">
            {isLoading ? (
              <div
                className="flex items-center justify-center font-mono text-xs text-text-subtle"
                style={{ width: W, height: H }}
              >
                Computing concept space…
              </div>
            ) : points.length === 0 ? (
              <div
                className="flex flex-col items-center justify-center gap-2 font-mono text-xs text-text-subtle"
                style={{ width: W, height: H }}
              >
                <span>No concepts available</span>
                <span className="text-[10px] text-text-muted">Start the server with a model to explore the registry</span>
              </div>
            ) : (
              <ScatterPlot
                points={points}
                conceptColors={conceptColors}
                theme={theme}
                loadedSet={loadedSet}
                onToggle={handleToggle}
                pendingConcept={pendingConcept}
              />
            )}
          </div>

          {mapData && (
            <div className="flex items-center justify-between px-5 py-2 border-t border-bg-border">
              <span className="font-mono text-[10px] text-text-muted">
                {points.filter((p) => p.loaded).length} loaded · {points.filter((p) => !p.loaded).length} available · model: {mapData.model_id}
              </span>
              <span className="font-mono text-[10px] text-text-muted">
                Loaded concepts have colored fills · available concepts are outlined
              </span>
            </div>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
