export function GhostSlider() {
  return (
    <div
      className="flex flex-col gap-1.5 rounded-lg border border-bg-border bg-bg-surface p-2.5 pointer-events-none select-none"
      style={{ opacity: 0.3 }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-bg-elevated shrink-0" />
          <span className="font-mono text-sm text-text-subtle">concept</span>
          <span className="font-mono text-xs text-text-muted">L0</span>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <span className="w-7 font-mono text-xs text-text-subtle text-right tabular-nums">0.0</span>
        <div className="relative flex-1 h-5 flex items-center">
          <div className="w-full h-px bg-bg-elevated rounded-full" />
          <div
            className="absolute h-4 w-4 rounded-full border-2 border-bg-elevated bg-bg-elevated"
            style={{ left: "50%", transform: "translateX(-50%)" }}
          />
        </div>
        <div className="flex gap-1 text-xs font-mono text-text-muted">
          <span>-3</span>
          <span>/</span>
          <span>3</span>
        </div>
      </div>
    </div>
  );
}
