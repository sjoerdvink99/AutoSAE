import { Cpu, Sun, Moon } from "lucide-react";
import { useStore } from "../../stores/useStore";

interface Props {
  connected: boolean | null;
}

export function Header({ connected }: Props) {
  const capabilities = useStore((s) => s.capabilities);
  const cards = useStore((s) => s.cards);
  const theme = useStore((s) => s.theme);
  const toggleTheme = useStore((s) => s.toggleTheme);
  const modelId = capabilities.model_id;

  return (
    <header className="flex items-center gap-3 border-b border-bg-border px-5 py-2 shrink-0">
      <Cpu size={16} className="text-accent" />
      <span className="font-mono text-sm font-semibold text-gradient">AutoSAE</span>
      {modelId && (
        <span className="font-mono text-[11px] text-text-subtle truncate max-w-[200px]">
          {modelId}
        </span>
      )}
      {cards.length > 0 && (
        <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-accent/10 text-accent border border-accent/20 tabular-nums">
          {cards.length} concept{cards.length !== 1 ? "s" : ""}
        </span>
      )}
      <div className="ml-auto flex items-center gap-3">
        <kbd
          className="font-mono text-[10px] text-text-subtle border border-bg-border rounded px-1.5 py-0.5 bg-bg-elevated cursor-pointer"
          onClick={() => {
            window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true, bubbles: true }));
          }}
        >
          ⌘K
        </kbd>
        <button
          onClick={toggleTheme}
          className="text-text-subtle hover:text-text transition-colors p-0.5"
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
        </button>
        <div className="flex items-center gap-1.5">
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{
              backgroundColor:
                connected === null ? "var(--color-text-subtle)" : connected ? "var(--color-accent)" : "var(--color-danger)",
              boxShadow: connected === true ? "0 0 6px var(--color-accent)" : undefined,
            }}
          />
          <span className="font-mono text-[11px] text-text-subtle">
            {connected === null ? "connecting" : connected ? "connected" : "offline"}
          </span>
        </div>
      </div>
    </header>
  );
}
