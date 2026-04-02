import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, ChevronRight, ChevronLeft } from "lucide-react";

const STORAGE_KEY = "autosae-tour-completed";

const STEPS = [
  {
    title: "Load a Concept Card",
    body: "Click 'Load' in the left sidebar to browse pre-computed concept cards from the registry, or create your own.",
  },
  {
    title: "Adjust the Alpha Slider",
    body: "The slider controls steering strength. Positive values amplify the concept; negative values suppress it. Press 1–9 to select a card, ↑↓ to nudge.",
  },
  {
    title: "Enter a Prompt",
    body: "Type a prompt in the text area at the bottom of the generation panel. Press ⌘Enter (or Ctrl+Enter) to generate.",
  },
  {
    title: "Watch the Activations",
    body: "The Live tab shows per-concept activation as the model generates. Hover a token for a breakdown, or click a token to adjust steering inline.",
  },
  {
    title: "Explore the Concept Space",
    body: "The PCA panel shows where concepts live in the model's hidden space. Drag the current point to steer interactively, or Shift+drag to lasso a target region.",
  },
];

export function OnboardingTour() {
  const [visible, setVisible] = useState(false);
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (!localStorage.getItem(STORAGE_KEY)) {
      setVisible(true);
    }
  }, []);

  const dismiss = () => {
    localStorage.setItem(STORAGE_KEY, "1");
    setVisible(false);
  };

  const next = () => {
    if (step < STEPS.length - 1) setStep((s) => s + 1);
    else dismiss();
  };

  const prev = () => setStep((s) => Math.max(0, s - 1));

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          key="tour"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          transition={{ duration: 0.25 }}
          className="fixed bottom-6 right-6 z-50 w-80 rounded-xl border border-bg-border bg-bg-elevated shadow-2xl p-5"
        >
          <div className="flex items-start justify-between mb-3">
            <div className="flex gap-1.5">
              {STEPS.map((_, i) => (
                <span
                  key={i}
                  className="h-1 rounded-full transition-all"
                  style={{
                    width: i === step ? 16 : 6,
                    backgroundColor: i === step ? "var(--color-accent)" : "var(--color-bg-border)",
                  }}
                />
              ))}
            </div>
            <button
              onClick={dismiss}
              className="text-text-subtle hover:text-text transition-colors p-0.5"
            >
              <X size={14} />
            </button>
          </div>

          <h4 className="font-mono text-sm font-semibold text-text mb-1.5">
            {STEPS[step]?.title}
          </h4>
          <p className="font-mono text-xs text-text-muted leading-relaxed">
            {STEPS[step]?.body}
          </p>

          <div className="flex items-center justify-between mt-4">
            <button
              onClick={prev}
              disabled={step === 0}
              className="flex items-center gap-1 font-mono text-xs text-text-subtle hover:text-text disabled:opacity-30 transition-colors"
            >
              <ChevronLeft size={12} />
              Back
            </button>
            <button
              onClick={next}
              className="flex items-center gap-1 font-mono text-xs text-accent hover:text-accent/80 transition-colors"
            >
              {step === STEPS.length - 1 ? "Done" : "Next"}
              {step < STEPS.length - 1 && <ChevronRight size={12} />}
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
