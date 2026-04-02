import { useEffect, useRef, useCallback, useMemo, useState } from "react";
import * as d3 from "d3";
import { useStore } from "../../stores/useStore";
import { useGeometry } from "../../hooks/useGeometry";
import { getConceptColor } from "../../lib/constants";
import { LassoOverlay } from "./LassoOverlay";
import type { CanvasViewport, ConceptGeometry, TrajectoryPoint } from "../../types";

const ARROWHEAD = "ah";
const CHI2_95 = 2.4477;
const PAD = 56;

function autoFitRadius(geometry: ConceptGeometry, trajectory: TrajectoryPoint[]): number {
  const allX = [0, ...geometry.vectors_2d.map((v) => v[0]), ...trajectory.map((t) => t.x)];
  const allY = [0, ...geometry.vectors_2d.map((v) => v[1]), ...trajectory.map((t) => t.y)];
  const xExt = d3.extent(allX) as [number, number];
  const yExt = d3.extent(allY) as [number, number];
  return (
    Math.max(
      Math.abs(xExt[0]),
      Math.abs(xExt[1]),
      Math.abs(yExt[0]),
      Math.abs(yExt[1]),
      0.15
    ) * 1.35
  );
}

function buildScales(
  geometry: ConceptGeometry,
  trajectory: TrajectoryPoint[],
  width: number,
  height: number,
  viewport: CanvasViewport
) {
  const rData = autoFitRadius(geometry, trajectory) / viewport.zoom;
  const aspect = (width - 2 * PAD) / (height - 2 * PAD);
  return {
    x: d3.scaleLinear(
      [viewport.cx - rData * aspect, viewport.cx + rData * aspect],
      [PAD, width - PAD]
    ),
    y: d3.scaleLinear(
      [viewport.cy + rData, viewport.cy - rData],
      [height - PAD, PAD]
    ),
  };
}

function computeEllipse(cov: number[][]): { rx: number; ry: number; rotation: number } {
  const a = cov[0]?.[0] ?? 0;
  const b = cov[0]?.[1] ?? 0;
  const dd = cov[1]?.[1] ?? 0;
  const disc = Math.sqrt(Math.max(0, ((a - dd) / 2) ** 2 + b * b));
  const lambda1 = (a + dd) / 2 + disc;
  const lambda2 = (a + dd) / 2 - disc;
  const rx = CHI2_95 * Math.sqrt(Math.max(0, lambda1));
  const ry = CHI2_95 * Math.sqrt(Math.max(0, lambda2));
  const rotation = Math.atan2(lambda1 - a, b) * (180 / Math.PI);
  return { rx, ry, rotation };
}

function computeRibbonWidth(
  cards: { concept: string; alpha: number; bootstrap_variance: number | null }[],
  confidenceEllipses?: Record<string, number[][]> | null
): number {
  let variance = 0;
  for (const card of cards) {
    if (confidenceEllipses != null) {
      const cov = confidenceEllipses[card.concept];
      if (cov != null) {
        const varXX = cov[0]?.[0] ?? 0;
        const varYY = cov[1]?.[1] ?? 0;
        variance += (varXX + varYY) * card.alpha * card.alpha;
        continue;
      }
    }
    if (card.bootstrap_variance == null) continue;
    variance += card.bootstrap_variance * card.alpha * card.alpha;
  }
  return Math.sqrt(Math.max(0, variance));
}

interface DraggablePointProps {
  cx: number;
  cy: number;
  onDelta: (dx: number, dy: number, altKey?: boolean) => void;
  onDragStateChange: (dragging: boolean) => void;
}

function DraggablePoint({ cx, cy, onDelta, onDragStateChange }: DraggablePointProps) {
  const ref = useRef<SVGCircleElement>(null);
  const prev = useRef<{ x: number; y: number } | null>(null);
  const onDeltaRef = useRef(onDelta);
  onDeltaRef.current = onDelta;
  const onDragStateRef = useRef(onDragStateChange);
  onDragStateRef.current = onDragStateChange;

  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const drag = d3
      .drag<SVGCircleElement, unknown>()
      .on("start", (e) => {
        prev.current = { x: e.x, y: e.y };
        onDragStateRef.current(true);
      })
      .on("drag", (e) => {
        if (!prev.current) return;
        const altKey = (e.sourceEvent as MouseEvent)?.altKey ?? false;
        onDeltaRef.current(e.x - prev.current.x, e.y - prev.current.y, altKey);
        prev.current = { x: e.x, y: e.y };
      })
      .on("end", () => {
        prev.current = null;
        setTimeout(() => onDragStateRef.current(false), 0);
      });
    d3.select(el).call(drag);
    return () => { d3.select(el).on(".drag", null); };
  }, []);

  return (
    <g>
      <circle cx={cx} cy={cy} r={12} fill="var(--color-accent)" fillOpacity={0.08} stroke="var(--color-accent)" strokeOpacity={0.19} strokeWidth={1} />
      <circle
        ref={ref}
        cx={cx}
        cy={cy}
        r={5}
        fill="var(--color-accent)"
        stroke="var(--color-bg)"
        strokeWidth={1.5}
        style={{ cursor: "crosshair", filter: "drop-shadow(0 0 5px var(--color-accent-glow))" }}
      />
    </g>
  );
}

interface TrajectoryParticlesProps {
  pathRef: React.RefObject<SVGPathElement | null>;
  isGenerating: boolean;
}

function TrajectoryParticles({ pathRef, isGenerating }: TrajectoryParticlesProps) {
  const refs = [
    useRef<SVGCircleElement>(null),
    useRef<SVGCircleElement>(null),
    useRef<SVGCircleElement>(null),
  ];
  const rafRef = useRef<number | null>(null);
  const offsets = [0, 0.33, 0.66];

  useEffect(() => {
    if (!isGenerating) {
      refs.forEach((r) => {
        if (r.current) r.current.setAttribute("opacity", "0");
      });
      return;
    }

    const animate = (time: number) => {
      const pathEl = pathRef.current;
      if (!pathEl) {
        rafRef.current = requestAnimationFrame(animate);
        return;
      }
      const totalLen = pathEl.getTotalLength();
      if (totalLen < 1) {
        rafRef.current = requestAnimationFrame(animate);
        return;
      }
      refs.forEach((r, i) => {
        if (!r.current) return;
        const t = ((time / 2000 + offsets[i]) % 1);
        const pt = pathEl.getPointAtLength(t * totalLen);
        r.current.setAttribute("cx", String(pt.x));
        r.current.setAttribute("cy", String(pt.y));
        r.current.setAttribute("opacity", "0.65");
      });
      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isGenerating, pathRef]);

  return (
    <>
      {refs.map((r, i) => (
        <circle
          key={i}
          ref={r}
          r={2.5}
          fill="var(--color-accent)"
          opacity={0}
          style={{ filter: "blur(1.5px)" }}
        />
      ))}
    </>
  );
}

export function ConceptSpaceCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const pathRef = useRef<SVGPathElement | null>(null);
  const [dims, setDims] = useState({ width: 420, height: 300 });
  const [showEllipses, setShowEllipses] = useState(true);
  const [showRibbons, setShowRibbons] = useState(true);
  const [clickPulse, setClickPulse] = useState<{ x: number; y: number } | null>(null);
  const [previewTarget, setPreviewTarget] = useState<[number, number] | null>(null);
  const [alphaSettled, setAlphaSettled] = useState(true);
  const [steerTooltip, setSteerTooltip] = useState<{
    px: number;
    py: number;
    deltas: { concept: string; delta: number }[];
  } | null>(null);

  const isDragging = useRef(false);
  const isPanning = useRef(false);
  const panStart = useRef<{ px: number; py: number; cx: number; cy: number } | null>(null);

  const geometry = useStore((s) => s.geometry);
  const trajectory = useStore((s) => s.trajectory);
  const isGenerating = useStore((s) => s.isGenerating);
  const cards = useStore((s) => s.cards);
  const hoveredTokenIndex = useStore((s) => s.hoveredTokenIndex);
  const setHoveredTokenIndex = useStore((s) => s.setHoveredTokenIndex);
  const trajectoryVisible = useStore((s) => s.trajectoryVisible);
  const canvasViewport = useStore((s) => s.canvasViewport);
  const setCanvasViewport = useStore((s) => s.setCanvasViewport);
  const theme = useStore((s) => s.theme);
  const conceptColors = useStore((s) => s.conceptColors);
  const { inverseProject } = useGeometry();

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const e = entries[0];
      if (!e) return;
      setDims({ width: e.contentRect.width, height: Math.max(e.contentRect.height, 200) });
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const alphaKey = cards.map((c) => c.alpha).join(",");
  useEffect(() => {
    setAlphaSettled(false);
    const t = setTimeout(() => setAlphaSettled(true), 600);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [alphaKey]);

  const scales = useMemo(() => {
    if (!geometry) return null;
    return buildScales(geometry, trajectory, dims.width, dims.height, canvasViewport);
  }, [geometry, trajectory, dims, canvasViewport]);

  const handleDelta = useCallback(
    (dpx: number, dpy: number, altKey?: boolean) => {
      if (!scales) return;
      const factor = altKey ? 0.25 : 1.0;
      const dx = (scales.x.invert(dims.width / 2 + dpx * factor) - scales.x.invert(dims.width / 2));
      const dy = scales.y.invert(dims.height / 2 + dpy * factor) - scales.y.invert(dims.height / 2);
      const cur = trajectory.at(-1);
      if (cur) {
        setPreviewTarget([cur.x + dx, cur.y + dy]);
      }
      inverseProject([dx, dy], "pca");
    },
    [scales, dims, trajectory, inverseProject]
  );

  const handleLasso = useCallback(
    (dataX: number, dataY: number) => {
      if (!scales) return;
      const cur = trajectory.at(-1);
      if (!cur) return;
      const dx = dataX - cur.x;
      const dy = dataY - cur.y;
      inverseProject([dx, dy], "pca");
    },
    [scales, trajectory, inverseProject]
  );

  const handleSvgClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (isDragging.current) return;
      if (!scales) return;
      const svg = e.currentTarget;
      const rect = svg.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const dataX = scales.x.invert(px);
      const dataY = scales.y.invert(py);
      const cur = trajectory.at(-1) ?? { x: 0, y: 0 };
      const dx = dataX - cur.x;
      const dy = dataY - cur.y;

      const MAX_CLICK_STEP = 0.5;
      const mag = Math.sqrt(dx * dx + dy * dy);
      const [clampedDx, clampedDy] =
        mag > MAX_CLICK_STEP ? [dx * MAX_CLICK_STEP / mag, dy * MAX_CLICK_STEP / mag] : [dx, dy];

      setClickPulse({ x: px, y: py });
      setTimeout(() => setClickPulse(null), 600);

      setPreviewTarget([cur.x + clampedDx, cur.y - clampedDy]);
      setTimeout(() => setPreviewTarget(null), 800);

      if (scales && geometry?.projection_jacobian) {
        const J = geometry.projection_jacobian;
        const deltas: { concept: string; delta: number }[] = [];
        for (let i = 0; i < geometry.concepts.length; i++) {
          const row = J[i];
          if (!row) continue;
          const d = (row[0] ?? 0) * clampedDx + (row[1] ?? 0) * clampedDy;
          if (Math.abs(d) > 0.001) {
            deltas.push({ concept: geometry.concepts[i]!, delta: d });
          }
        }
        if (deltas.length > 0) {
          setSteerTooltip({ px, py, deltas });
          setTimeout(() => setSteerTooltip(null), 1500);
        }
      }

      inverseProject([clampedDx, clampedDy], "pca");
    },
    [scales, trajectory, inverseProject, geometry]
  );

  const handleWheelZoom = useCallback(
    (e: React.WheelEvent<SVGSVGElement>) => {
      e.preventDefault();
      if (!scales) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const cursorX = scales.x.invert(px);
      const cursorY = scales.y.invert(py);
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = canvasViewport.zoom * factor;
      const clampedZoom = Math.max(0.25, Math.min(8.0, newZoom));
      const ratio = canvasViewport.zoom / clampedZoom;
      setCanvasViewport({
        cx: cursorX + (canvasViewport.cx - cursorX) * ratio,
        cy: cursorY + (canvasViewport.cy - cursorY) * ratio,
        zoom: clampedZoom,
      });
    },
    [scales, canvasViewport, setCanvasViewport]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      isDragging.current = false;
      const target = e.target as SVGElement;
      const isBackground =
        target === svgRef.current ||
        target.tagName === "line" ||
        (target.tagName === "text" && !target.closest("g[data-interactive]"));
      if (isBackground && !e.shiftKey) {
        isPanning.current = true;
        panStart.current = {
          px: e.clientX,
          py: e.clientY,
          cx: canvasViewport.cx,
          cy: canvasViewport.cy,
        };
      }
    },
    [canvasViewport]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (isPanning.current && panStart.current && scales) {
        const dpx = e.clientX - panStart.current.px;
        const dpy = e.clientY - panStart.current.py;
        if (Math.sqrt(dpx * dpx + dpy * dpy) > 3) {
          isDragging.current = true;
          const startDataX = scales.x.invert(panStart.current.px - (e.currentTarget.getBoundingClientRect().left));
          const endDataX = scales.x.invert(panStart.current.px - (e.currentTarget.getBoundingClientRect().left) + dpx);
          const ddx = startDataX - endDataX;
          const startDataY = scales.y.invert(panStart.current.py - (e.currentTarget.getBoundingClientRect().top));
          const endDataY = scales.y.invert(panStart.current.py - (e.currentTarget.getBoundingClientRect().top) + dpy);
          const ddy = startDataY - endDataY;
          setCanvasViewport({
            cx: panStart.current.cx + ddx,
            cy: panStart.current.cy + ddy,
            zoom: canvasViewport.zoom,
          });
        }
      }
    },
    [scales, canvasViewport, setCanvasViewport]
  );

  const handleMouseUp = useCallback(() => {
    isPanning.current = false;
    panStart.current = null;
    setPreviewTarget(null);
  }, []);

  const lineGen = useMemo(() => {
    if (!scales) return null;
    return d3
      .line<TrajectoryPoint>()
      .x((d) => scales.x(d.x))
      .y((d) => scales.y(d.y))
      .curve(d3.curveCatmullRom.alpha(0.5));
  }, [scales]);

  const ribbonAreaGen = useMemo(() => {
    if (!scales || !geometry) return null;
    const scaleFactorY = Math.abs(scales.y(0) - scales.y(1));
    return d3
      .area<TrajectoryPoint>()
      .x((d) => scales.x(d.x))
      .y0((d) => {
        const w = computeRibbonWidth(cards, geometry.confidence_ellipses) * scaleFactorY;
        return scales.y(d.y) - w;
      })
      .y1((d) => {
        const w = computeRibbonWidth(cards, geometry.confidence_ellipses) * scaleFactorY;
        return scales.y(d.y) + w;
      })
      .curve(d3.curveCatmullRom.alpha(0.5));
  }, [scales, geometry, cards]);

  const cursorStyle = isPanning.current ? "grabbing" : "crosshair";

  if (!geometry || !scales) {
    return (
      <div ref={containerRef} className="w-full h-full flex items-center justify-center relative">
        <svg width="100%" height="100%" style={{ position: "absolute", inset: 0 }}>
          <line x1="10%" y1="50%" x2="90%" y2="50%" stroke="var(--color-grid)" strokeWidth={1} />
          <line x1="50%" y1="10%" x2="50%" y2="90%" stroke="var(--color-grid)" strokeWidth={1} />
          {[30, 150, 270].map((deg) => {
            const rad = (deg * Math.PI) / 180;
            return (
              <line
                key={deg}
                x1="50%"
                y1="50%"
                x2={`${50 + Math.cos(rad) * 30}%`}
                y2={`${50 - Math.sin(rad) * 30}%`}
                stroke="var(--color-accent)"
                strokeOpacity={0.13}
                strokeWidth={1.5}
                markerEnd="none"
              />
            );
          })}
          <circle cx="50%" cy="50%" r={4} fill="var(--color-accent)" fillOpacity={0.15} style={{ animation: "click-pulse 2s ease-out infinite" }} />
        </svg>
        <div className="relative z-10 flex flex-col items-center gap-3 text-center px-6">
          <span className="font-mono text-xs text-text-subtle leading-relaxed">
            Load a concept to begin exploring the model's hidden geometry
          </span>
        </div>
      </div>
    );
  }

  const { width: W, height: H } = dims;
  const ox = scales.x(0);
  const oy = scales.y(0);
  const current = trajectory.at(-1);
  const pathD = lineGen && trajectory.length >= 2 ? lineGen(trajectory) : null;
  const uniqueColors = [...new Set(geometry.concepts.map((c) => getConceptColor(c, theme, conceptColors)))];
  const lastPt = trajectory.at(-1);
  const ribbonVisible = showRibbons && alphaSettled && !isGenerating && ribbonAreaGen && trajectory.length >= 2;

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <div className="absolute top-0 right-0 flex items-center gap-3 z-10 p-1">
        {geometry.confidence_ellipses && (
          <label className="flex items-center gap-1.5 font-mono text-[10px] text-text-subtle cursor-pointer">
            <input
              type="checkbox"
              checked={showEllipses}
              onChange={(e) => setShowEllipses(e.target.checked)}
              className="accent-accent"
            />
            Confidence ellipses
          </label>
        )}
        {trajectory.length >= 2 && cards.some((c) => c.bootstrap_variance != null) && (
          <label className="flex items-center gap-1.5 font-mono text-[10px] text-text-subtle cursor-pointer">
            <input
              type="checkbox"
              checked={showRibbons}
              onChange={(e) => setShowRibbons(e.target.checked)}
              className="accent-accent"
            />
            Confidence ribbons
          </label>
        )}
      </div>
      <svg
        ref={svgRef}
        width={W}
        height={H}
        style={{ overflow: "visible", display: "block", cursor: cursorStyle }}
        onClick={handleSvgClick}
        onWheel={handleWheelZoom}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          isPanning.current = false;
          panStart.current = null;
          setPreviewTarget(null);
        }}
      >
        <defs>
          {uniqueColors.map((col) => (
            <marker
              key={col}
              id={`${ARROWHEAD}-${col.replace("#", "")}`}
              markerWidth={7}
              markerHeight={7}
              refX={3.5}
              refY={3.5}
              orient="auto"
            >
              <path d="M0,1 L0,6 L5,3.5 z" fill={col} fillOpacity={0.85} />
            </marker>
          ))}
        </defs>

        <line x1={PAD} y1={oy} x2={W - PAD} y2={oy} stroke="var(--color-grid)" strokeWidth={1} />
        <line x1={ox} y1={PAD} x2={ox} y2={H - PAD} stroke="var(--color-grid)" strokeWidth={1} />

        <text x={W - PAD + 2} y={oy - 5} fill="var(--color-text-subtle)" fontSize={10} fontFamily="monospace" textAnchor="end">
          PC₁ {(geometry.variance_ratio[0] * 100).toFixed(0)}%
        </text>
        <text x={ox + 4} y={PAD + 8} fill="var(--color-text-subtle)" fontSize={10} fontFamily="monospace">
          PC₂ {(geometry.variance_ratio[1] * 100).toFixed(0)}%
        </text>

        {trajectoryVisible && pathD && (
          <>
            {ribbonVisible && (
              <>
                <path
                  d={ribbonAreaGen(trajectory) ?? ""}
                  fill="var(--color-accent)"
                  fillOpacity={0.07}
                  stroke="none"
                />
                {lastPt && (
                  <text
                    x={scales.x(lastPt.x) + 8}
                    y={scales.y(lastPt.y)}
                    fill="var(--color-text-muted)"
                    fontSize={8}
                    fontFamily="monospace"
                  >
                    ±1σ (bootstrap)
                  </text>
                )}
              </>
            )}
            <path
              ref={pathRef}
              d={pathD}
              fill="none"
              stroke="none"
              strokeWidth={0}
            />
            {trajectory.length >= 2 && trajectory.map((pt, i) => {
              if (i === 0) return null;
              const t = i / Math.max(trajectory.length - 1, 1);
              const color = d3.interpolateViridis(t * 0.85 + 0.05);
              const maxAct = Math.max(
                1e-9,
                ...Object.values(pt.activations).map(Math.abs)
              );
              const isHovered = hoveredTokenIndex === pt.index;
              const r = isHovered ? 8 : 1 + Math.min(maxAct, 1) * 3;
              const prev = trajectory[i - 1]!;
              return (
                <g key={i}>
                  <line
                    x1={scales.x(prev.x)}
                    y1={scales.y(prev.y)}
                    x2={scales.x(pt.x)}
                    y2={scales.y(pt.y)}
                    stroke={color}
                    strokeWidth={1.5}
                    strokeOpacity={0.7}
                    strokeLinecap="round"
                  />
                  <circle
                    cx={scales.x(pt.x)}
                    cy={scales.y(pt.y)}
                    r={r}
                    fill={color}
                    fillOpacity={isHovered ? 0.9 : 0.5}
                    stroke={isHovered ? "var(--color-bg)" : "none"}
                    strokeWidth={isHovered ? 2 : 0}
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHoveredTokenIndex(pt.index)}
                    onMouseLeave={() => setHoveredTokenIndex(null)}
                  />
                  {i % 10 === 0 && (
                    <text
                      x={scales.x(pt.x) + 6}
                      y={scales.y(pt.y) - 2}
                      fill="var(--color-text-muted)"
                      fontSize={8}
                      fontFamily="monospace"
                    >
                      t={pt.index}
                    </text>
                  )}
                </g>
              );
            })}
            <TrajectoryParticles pathRef={pathRef} isGenerating={isGenerating} />
          </>
        )}

        {geometry.concepts.map((concept, i) => {
          const v = geometry.vectors_2d[i];
          if (!v) return null;
          const color = getConceptColor(concept, theme, conceptColors);
          const card = cards.find((c) => c.concept === concept);
          const alpha = card?.alpha ?? 1;
          const absAlpha = Math.abs(alpha);
          const opacity = 0.45 + Math.min(absAlpha / 3, 1) * 0.55;

          if (absAlpha < 0.05) {
            const ghostTx = scales.x(v[0]);
            const ghostTy = scales.y(v[1]);
            return (
              <g key={concept} opacity={0.15}>
                <line
                  x1={ox}
                  y1={oy}
                  x2={ghostTx}
                  y2={ghostTy}
                  stroke={color}
                  strokeWidth={1}
                  strokeDasharray="4 3"
                />
              </g>
            );
          }

          const tx = scales.x(v[0] * alpha);
          const ty = scales.y(v[1] * alpha);
          const dx = tx - ox;
          const dy = ty - oy;
          const len = Math.sqrt(dx * dx + dy * dy);
          const lx = len > 0 ? tx + (dx / len) * 16 : tx;
          const ly = len > 0 ? ty + (dy / len) * 16 : ty;

          const cov = geometry.confidence_ellipses?.[concept];
          const ellipse = cov && showEllipses ? computeEllipse(cov) : null;

          const scaleFactorX = scales.x(1) - scales.x(0);
          const scaleFactorY = Math.abs(scales.y(0) - scales.y(1));

          return (
            <g key={concept} data-interactive="true">
              {ellipse && (
                <ellipse
                  cx={tx}
                  cy={ty}
                  rx={ellipse.rx * scaleFactorX}
                  ry={ellipse.ry * scaleFactorY}
                  fill={color}
                  fillOpacity={0.08}
                  stroke={color}
                  strokeOpacity={0.25}
                  strokeWidth={1}
                  strokeDasharray="4 3"
                  transform={`rotate(${-ellipse.rotation}, ${tx}, ${ty})`}
                />
              )}
              <line
                x1={ox}
                y1={oy}
                x2={tx}
                y2={ty}
                stroke={color}
                strokeWidth={1.5}
                strokeOpacity={opacity}
                markerEnd={`url(#${ARROWHEAD}-${color.replace("#", "")})`}
                style={absAlpha > 0.5 ? { filter: `drop-shadow(0 0 3px ${color}60)` } : undefined}
              />
              <g opacity={opacity}>
                <rect
                  x={lx - 32}
                  y={ly - 9}
                  width={64}
                  height={18}
                  rx={3}
                  fill={`${color}26`}
                  stroke={`${color}44`}
                  strokeWidth={1}
                />
                <text
                  x={lx}
                  y={ly}
                  fill={color}
                  fontSize={11}
                  fontFamily="monospace"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {concept} a={alpha.toFixed(1)}
                </text>
              </g>
            </g>
          );
        })}

        {current && previewTarget && (
          <line
            x1={scales.x(current.x)}
            y1={scales.y(current.y)}
            x2={scales.x(previewTarget[0])}
            y2={scales.y(previewTarget[1])}
            stroke="var(--color-accent)"
            strokeDasharray="4 4"
            strokeOpacity={0.5}
            strokeWidth={1.5}
          />
        )}

        {current && (
          <DraggablePoint
            cx={scales.x(current.x)}
            cy={scales.y(current.y)}
            onDelta={handleDelta}
            onDragStateChange={(dragging) => { isDragging.current = dragging; }}
          />
        )}

        <LassoOverlay
          width={W}
          height={H}
          scaleX={scales.x}
          scaleY={scales.y}
          onLasso={handleLasso}
        />

        <text x={W - PAD} y={H - 6} fill="var(--color-text-subtle)" fontSize={9} fontFamily="monospace" textAnchor="end">
          click to steer · shift+drag to lasso · scroll to zoom · drag to pan
        </text>
        {trajectory.length >= 2 && (
          <text x={PAD + 4} y={H - 6} fill="var(--color-text-subtle)" fontSize={9} fontFamily="monospace">
            {trajectory.length} pts
          </text>
        )}

        {clickPulse && (
          <circle
            cx={clickPulse.x}
            cy={clickPulse.y}
            r={12}
            fill="none"
            stroke="var(--color-accent)"
            strokeWidth={1.5}
            strokeOpacity={0.6}
            style={{ animation: "click-pulse 0.6s ease-out forwards" }}
          />
        )}

        {steerTooltip && (
          <foreignObject
            x={Math.min(steerTooltip.px + 10, W - 140)}
            y={Math.min(steerTooltip.py - 10, H - 60)}
            width={130}
            height={steerTooltip.deltas.length * 16 + 8}
            style={{ pointerEvents: "none" }}
          >
            <div
              style={{
                background: "var(--color-bg-surface)",
                border: "1px solid var(--color-bg-border)",
                borderRadius: 4,
                padding: "3px 6px",
                fontFamily: "monospace",
                fontSize: 10,
                color: "var(--color-text-muted)",
              }}
            >
              {steerTooltip.deltas.map(({ concept, delta }) => (
                <div key={concept}>
                  {concept} {delta > 0 ? "+" : ""}{delta.toFixed(2)}
                </div>
              ))}
            </div>
          </foreignObject>
        )}
      </svg>
    </div>
  );
}
