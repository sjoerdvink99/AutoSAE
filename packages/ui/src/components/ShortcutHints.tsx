const HINTS = [
  { key: "1–9", label: "select concept" },
  { key: "↑↓", label: "adjust α" },
  { key: "⇧↑↓", label: "±0.5" },
  { key: "tab", label: "cycle" },
  { key: "⌘K", label: "commands" },
  { key: "⌘Z", label: "undo" },
];

export function ShortcutHints() {
  return (
    <div className="border-t border-bg-border px-3 py-2 flex flex-wrap gap-x-3 gap-y-1">
      {HINTS.map(({ key, label }) => (
        <div key={key} className="flex items-center gap-1">
          <kbd className="font-mono text-[9px] text-text-subtle border border-bg-border rounded px-1 py-px bg-bg-elevated">
            {key}
          </kbd>
          <span className="font-mono text-[9px] text-text-subtle opacity-60">{label}</span>
        </div>
      ))}
    </div>
  );
}
