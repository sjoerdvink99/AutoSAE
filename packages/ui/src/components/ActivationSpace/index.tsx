import { useState } from "react";
import { Atom, Activity, Clock } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "../../stores/useStore";
import { useGeometry } from "../../hooks/useGeometry";
import { ConceptSpaceCanvas } from "./ConceptSpaceCanvas";
import { LiveActivationView } from "./LiveActivationView";
import { HistoryView } from "./HistoryView";
import { cn } from "../../lib/cn";

type Tab = "steer" | "monitor" | "history";

const TABS: { id: Tab; label: string; icon: typeof Atom }[] = [
  { id: "steer", label: "Steer", icon: Atom },
  { id: "monitor", label: "Monitor", icon: Activity },
  { id: "history", label: "History", icon: Clock },
];

export function ActivationSpacePanel() {
  const [tab, setTab] = useState<Tab>("steer");
  const cards = useStore((s) => s.cards);
  const trajectory = useStore((s) => s.trajectory);

  useGeometry();

  return (
    <div className="flex flex-col border-t border-bg-border bg-bg min-h-0 h-full">
      <div className="flex items-center gap-0 border-b border-bg-border px-2 shrink-0">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={cn(
              "relative flex items-center gap-1.5 px-3 py-2.5 font-mono text-xs transition-colors",
              tab === id ? "text-text" : "text-text-subtle hover:text-text-muted"
            )}
          >
            <Icon size={12} />
            {label}
            {tab === id && (
              <motion.span
                layoutId="tab-indicator"
                className="absolute bottom-0 left-0 right-0 h-px bg-accent"
                transition={{ type: "spring", stiffness: 500, damping: 30 }}
              />
            )}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-3 pr-3">
          {trajectory.length > 0 && (
            <span className="font-mono text-[10px] text-text-subtle tabular-nums">
              {trajectory.length} pts
            </span>
          )}
          {cards.length > 0 && (
            <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-accent/10 text-accent border border-accent/20">
              {cards.length} {cards.length === 1 ? "concept" : "concepts"}
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-hidden p-3">
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="h-full flex flex-col"
          >
            {tab === "steer" && <ConceptSpaceCanvas />}
            {tab === "monitor" && <LiveActivationView />}
            {tab === "history" && <HistoryView />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
