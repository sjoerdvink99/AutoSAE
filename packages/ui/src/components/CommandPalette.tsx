import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Search } from "lucide-react";
import { useStore } from "../stores/useStore";
import { cardsApi } from "../api/cards";
import { useGeneration } from "../hooks/useGeneration";

interface Command {
  id: string;
  label: string;
  pattern: RegExp;
  action: (match: RegExpMatchArray) => void;
}

function useCommands(): Command[] {
  const cards = useStore((s) => s.cards);
  const updateCardAlpha = useStore((s) => s.updateCardAlpha);
  const pushAlphaSnapshot = useStore((s) => s.pushAlphaSnapshot);
  const setCards = useStore((s) => s.setCards);
  const setCanvasViewport = useStore((s) => s.setCanvasViewport);
  const canvasViewport = useStore((s) => s.canvasViewport);
  const { generate, regenerateFrom } = useGeneration();

  return [
    {
      id: "load",
      label: 'load "<concept>"',
      pattern: /^load\s+["']?(\w+)["']?$/i,
      action: (match) => {
        const concept = match[1]!;
        void cardsApi
          .load({ registry_concept: concept, registry_model: "llama-3.1-8b" })
          .then(() => cardsApi.list())
          .then((result) => setCards(result));
      },
    },
    {
      id: "set",
      label: "set <concept> [alpha] to <value>",
      pattern: /^set\s+(\w+)\s+(?:alpha\s+)?to\s+([-\d.]+)$/i,
      action: (match) => {
        const concept = match[1]!;
        const alpha = parseFloat(match[2]!);
        if (isNaN(alpha)) return;
        pushAlphaSnapshot();
        updateCardAlpha(concept, Math.max(-3, Math.min(3, alpha)));
        void cardsApi.setAlpha(concept, alpha);
      },
    },
    {
      id: "reset-all",
      label: "reset all",
      pattern: /^reset\s+all$/i,
      action: () => {
        pushAlphaSnapshot();
        for (const card of cards) {
          updateCardAlpha(card.concept, 0);
          void cardsApi.setAlpha(card.concept, 0);
        }
      },
    },
    {
      id: "zoom",
      label: "zoom <level>",
      pattern: /^zoom\s+([\d.]+)$/i,
      action: (match) => {
        const z = parseFloat(match[1]!);
        if (!isNaN(z)) setCanvasViewport({ ...canvasViewport, zoom: z });
      },
    },
    {
      id: "regenerate",
      label: "regenerate from <token-index>",
      pattern: /^regenerate\s+from\s+(\d+)$/i,
      action: (match) => {
        const idx = parseInt(match[1]!, 10);
        regenerateFrom(idx);
      },
    },
    {
      id: "generate",
      label: 'generate "<prompt>"',
      pattern: /^generate\s+["'](.+)["']$/i,
      action: (match) => {
        generate(match[1]!);
      },
    },
    {
      id: "unload",
      label: "unload <concept>",
      pattern: /^unload\s+(\w+)$/i,
      action: (match) => {
        const concept = match[1]!;
        void cardsApi.unload(concept).then(() => cardsApi.list()).then((result) => setCards(result));
      },
    },
    {
      id: "alpha",
      label: "<concept> <alpha>",
      pattern: /^(\w+)\s+([-\d.]+)$/i,
      action: (match) => {
        const concept = match[1]!;
        const alpha = parseFloat(match[2]!);
        if (isNaN(alpha)) return;
        const card = cards.find((c) => c.concept === concept);
        if (!card) return;
        pushAlphaSnapshot();
        updateCardAlpha(concept, Math.max(-3, Math.min(3, alpha)));
        void cardsApi.setAlpha(concept, alpha);
      },
    },
  ];
}

function buildSuggestions(input: string, commands: Command[]): string[] {
  if (!input.trim()) {
    return commands.map((c) => c.label);
  }
  return commands
    .filter((c) => c.label.toLowerCase().includes(input.toLowerCase()))
    .map((c) => c.label);
}

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const commands = useCommands();
  const suggestions = buildSuggestions(input, commands);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((o) => !o);
        setInput("");
        setSelectedIdx(0);
      }
      if (e.key === "Escape" && open) {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open]);

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  const execute = useCallback(
    (query: string) => {
      const trimmed = query.trim();
      for (const cmd of commands) {
        const match = trimmed.match(cmd.pattern);
        if (match) {
          cmd.action(match);
          break;
        }
      }
      setOpen(false);
      setInput("");
    },
    [commands]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        execute(input);
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIdx((i) => (i + 1) % Math.max(suggestions.length, 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIdx((i) => (i - 1 + Math.max(suggestions.length, 1)) % Math.max(suggestions.length, 1));
      } else if (e.key === "Tab" && suggestions[selectedIdx]) {
        e.preventDefault();
        setInput(suggestions[selectedIdx]!.replace(/<[^>]+>/g, "").trim());
      }
    },
    [input, suggestions, selectedIdx, execute]
  );

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -8 }}
            transition={{ duration: 0.15 }}
            className="fixed top-[20%] left-1/2 -translate-x-1/2 z-50 w-full max-w-md"
          >
            <div className="bg-bg-elevated border border-bg-border rounded-xl shadow-2xl overflow-hidden">
              <div className="flex items-center gap-3 px-4 py-3 border-b border-bg-border">
                <Search size={14} className="text-text-subtle shrink-0" />
                <input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => {
                    setInput(e.target.value);
                    setSelectedIdx(0);
                  }}
                  onKeyDown={handleKeyDown}
                  placeholder="Type a command..."
                  className="flex-1 bg-transparent font-mono text-sm text-text placeholder:text-text-subtle focus:outline-none"
                />
                <kbd className="font-mono text-[10px] text-text-subtle border border-bg-border rounded px-1.5 py-0.5">
                  esc
                </kbd>
              </div>
              {suggestions.length > 0 && (
                <div className="max-h-64 overflow-y-auto py-1">
                  {suggestions.map((label, i) => (
                    <button
                      key={label}
                      onClick={() => execute(label.replace(/<[^>]+>/g, "").trim())}
                      className={`w-full px-4 py-2 text-left font-mono text-xs transition-colors ${
                        i === selectedIdx
                          ? "bg-accent/10 text-accent"
                          : "text-text-muted hover:bg-bg-surface hover:text-text"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              )}
              <div className="px-4 py-2 border-t border-bg-border flex items-center gap-4">
                <span className="font-mono text-[10px] text-text-subtle">
                  <kbd className="border border-bg-border rounded px-1">↵</kbd> execute
                </span>
                <span className="font-mono text-[10px] text-text-subtle">
                  <kbd className="border border-bg-border rounded px-1">tab</kbd> complete
                </span>
                <span className="font-mono text-[10px] text-text-subtle">
                  <kbd className="border border-bg-border rounded px-1">↑↓</kbd> navigate
                </span>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
