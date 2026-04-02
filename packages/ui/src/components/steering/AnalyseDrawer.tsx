import { useState } from "react";
import { X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "../../stores/useStore";
import { GramMatrixHeatmap } from "./GramMatrixHeatmap";
import { ActivationHistogram } from "./ActivationHistogram";
import { InfluenceFlow } from "./InfluenceFlow";
import { CombineExportRow } from "./CombineExportRow";
import { AlphaTimeline } from "../MixingBoard/AlphaTimeline";
import { SteeringMagnitudeTimeline } from "../canvas/SteeringMagnitudeTimeline";
import { MonitorStrip } from "../canvas/MonitorStrip";

type Tab = "gram" | "histogram" | "influence" | "timeline" | "magnitude" | "monitor" | "combine";

const TABS: { id: Tab; label: string; minCards?: number }[] = [
  { id: "gram", label: "Gram", minCards: 2 },
  { id: "histogram", label: "Activations" },
  { id: "influence", label: "Influence", minCards: 2 },
  { id: "timeline", label: "Alpha History" },
  { id: "magnitude", label: "Magnitude" },
  { id: "monitor", label: "Monitor" },
  { id: "combine", label: "Combine", minCards: 2 },
];

export function AnalyseDrawer() {
  const [open, setOpen] = useState(false);
  const cards = useStore((s) => s.cards);
  const [activeTab, setActiveTab] = useState<Tab>("histogram");

  const visibleTabs = TABS.filter((t) => !t.minCards || cards.length >= t.minCards);

  const resolvedTab =
    visibleTabs.find((t) => t.id === activeTab) ? activeTab : (visibleTabs[0]?.id ?? "histogram");

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="analyse-drawer"
          initial={{ height: 0 }}
          animate={{ height: "34vh" }}
          exit={{ height: 0 }}
          transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
          className="border-t border-bg-border bg-bg-surface overflow-hidden flex flex-col shrink-0"
        >
          <div className="flex items-center gap-0.5 px-3 py-1.5 border-b border-bg-border shrink-0">
            {visibleTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`font-mono text-[11px] px-2.5 py-1 rounded transition-colors ${
                  resolvedTab === tab.id
                    ? "bg-accent/10 text-accent"
                    : "text-text-subtle hover:text-text"
                }`}
              >
                {tab.label}
              </button>
            ))}
            <button
              onClick={() => setOpen(false)}
              className="ml-auto p-1 text-text-subtle hover:text-text transition-colors rounded"
            >
              <X size={13} />
            </button>
          </div>

          <div className="flex-1 overflow-auto min-h-0">
            {resolvedTab === "gram" && <GramMatrixHeatmap />}
            {resolvedTab === "histogram" && <ActivationHistogram />}
            {resolvedTab === "influence" && <InfluenceFlow />}
            {resolvedTab === "timeline" && (
              <div className="p-2">
                <AlphaTimeline />
              </div>
            )}
            {resolvedTab === "magnitude" && (
              <div className="p-2">
                <SteeringMagnitudeTimeline width={800} />
              </div>
            )}
            {resolvedTab === "monitor" && <MonitorStrip />}
            {resolvedTab === "combine" && <CombineExportRow />}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
