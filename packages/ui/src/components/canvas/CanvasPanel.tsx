import { useStore } from "../../stores/useStore";
import { useGeometry } from "../../hooks/useGeometry";
import { ConceptSpaceCanvas } from "../ActivationSpace/ConceptSpaceCanvas";

export function CanvasPanel() {
  const geometry = useStore((s) => s.geometry);
  const trajectory = useStore((s) => s.trajectory);
  const canvasViewport = useStore((s) => s.canvasViewport);
  const resetCanvasViewport = useStore((s) => s.resetCanvasViewport);
  const trajectoryVisible = useStore((s) => s.trajectoryVisible);
  const toggleTrajectory = useStore((s) => s.toggleTrajectory);
  const gridVisible = useStore((s) => s.gridVisible);
  const toggleGrid = useStore((s) => s.toggleGrid);

  useGeometry();

  return (
    <div className="flex flex-col h-full min-h-0 bg-bg">
      <div className="flex items-center gap-3 px-3 py-1.5 border-b border-bg-border shrink-0">
        {geometry && (
          <>
            <span className="font-mono text-[10px] text-text-subtle tabular-nums">
              PC1 {(geometry.variance_ratio[0] * 100).toFixed(0)}% · PC2 {(geometry.variance_ratio[1] * 100).toFixed(0)}%
            </span>
            {trajectory.length > 0 && (
              <span className="font-mono text-[10px] text-text-subtle tabular-nums">
                {trajectory.length} pts
              </span>
            )}
          </>
        )}
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={toggleTrajectory}
            className={`font-mono text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
              trajectoryVisible
                ? "bg-accent/10 text-accent border-accent/20"
                : "text-text-subtle border-bg-border hover:border-bg-elevated"
            }`}
          >
            traj
          </button>
          <button
            onClick={toggleGrid}
            className={`font-mono text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
              gridVisible
                ? "bg-accent/10 text-accent border-accent/20"
                : "text-text-subtle border-bg-border hover:border-bg-elevated"
            }`}
          >
            grid
          </button>
          {canvasViewport.zoom !== 1.0 && (
            <button
              onClick={resetCanvasViewport}
              className="font-mono text-[10px] px-1.5 py-0.5 rounded border border-bg-border text-text-subtle hover:border-bg-elevated transition-colors"
              title="Reset zoom (Ctrl+0)"
            >
              {(canvasViewport.zoom * 100).toFixed(0)}%
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-hidden">
        <ConceptSpaceCanvas />
      </div>
    </div>
  );
}
