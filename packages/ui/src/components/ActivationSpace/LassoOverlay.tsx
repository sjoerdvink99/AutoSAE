import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

interface Point {
  x: number;
  y: number;
}

interface Props {
  width: number;
  height: number;
  scaleX: d3.ScaleLinear<number, number>;
  scaleY: d3.ScaleLinear<number, number>;
  onLasso: (centroidX: number, centroidY: number) => void;
}

export function LassoOverlay({ width, height, scaleX, scaleY, onLasso }: Props) {
  const [points, setPoints] = useState<Point[]>([]);
  const [active, setActive] = useState(false);
  const [shiftHeld, setShiftHeld] = useState(false);
  const ref = useRef<SVGRectElement>(null);
  const onLassoRef = useRef(onLasso);
  onLassoRef.current = onLasso;

  const polygonD =
    points.length >= 2
      ? `M ${points.map((p) => `${p.x},${p.y}`).join(" L ")} Z`
      : null;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => setShiftHeld(e.shiftKey);
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;

    const drag = d3
      .drag<SVGRectElement, unknown>()
      .filter((event) => (event as MouseEvent).shiftKey)
      .on("start", () => {
        setPoints([]);
        setActive(true);
      })
      .on("drag", (event) => {
        const e = event as { x: number; y: number };
        setPoints((prev) => [...prev, { x: e.x, y: e.y }]);
      })
      .on("end", () => {
        setActive(false);
        setPoints((prev) => {
          if (prev.length === 0) return [];
          const centX = prev.reduce((s, p) => s + p.x, 0) / prev.length;
          const centY = prev.reduce((s, p) => s + p.y, 0) / prev.length;
          onLassoRef.current(scaleX.invert(centX), scaleY.invert(centY));
          return [];
        });
      });

    d3.select(el).call(drag);
    return () => {
      d3.select(el).on(".drag", null);
    };
  }, [scaleX, scaleY]);

  const cursor = active ? "crosshair" : shiftHeld ? "cell" : "default";

  return (
    <g>
      <rect
        ref={ref}
        x={0}
        y={0}
        width={width}
        height={height}
        fill="transparent"
        style={{ cursor, pointerEvents: shiftHeld || active ? "all" : "none" }}
      />
      {polygonD && (
        <path
          d={polygonD}
          fill="var(--color-accent)"
          fillOpacity={0.08}
          stroke="var(--color-accent)"
          strokeWidth={1.5}
          strokeOpacity={0.7}
          strokeDasharray="6 3"
        />
      )}
    </g>
  );
}
