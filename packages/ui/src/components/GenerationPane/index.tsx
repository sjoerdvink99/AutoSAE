import { useRef, useEffect } from "react";
import { Play, Square, Trash2, Cpu } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useStore } from "../../stores/useStore";
import { useGeneration } from "../../hooks/useGeneration";
import { cardsApi } from "../../api/cards";
import { REGISTRY_CONCEPTS, getConceptColor } from "../../lib/constants";
import { Button } from "../ui/Button";
import { ConceptDot } from "../ui/ConceptDot";
import { TokenizedOutput } from "./TokenizedOutput";

const MAX_TOKENS = 512;

function QuickStart() {
  const queryClient = useQueryClient();
  const setCards = useStore((s) => s.setCards);

  const loadMutation = useMutation({
    mutationFn: (params: { registry_concept: string; registry_model: string }) =>
      cardsApi.load(params),
    onSuccess: async () => {
      const result = await cardsApi.list();
      setCards(result);
      queryClient.invalidateQueries({ queryKey: ["cards"] });
    },
  });

  return (
    <div className="flex flex-col items-center justify-center flex-1 gap-6 p-8">
      <div className="flex flex-col items-center gap-2">
        <Cpu size={28} className="text-accent opacity-80" />
        <h3 className="font-mono text-base font-semibold text-text">Start Steering</h3>
        <p className="font-mono text-xs text-text-subtle text-center max-w-xs leading-relaxed">
          Load a concept card to begin steering model behaviour. Click one below to get started.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-2 w-full max-w-sm">
        {REGISTRY_CONCEPTS.map((concept) => {
          const color = getConceptColor(concept.name);
          return (
            <button
              key={concept.name}
              onClick={() =>
                loadMutation.mutate({
                  registry_concept: concept.name,
                  registry_model: concept.model,
                })
              }
              disabled={loadMutation.isPending}
              className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-bg-border bg-bg-surface hover:border-bg-elevated transition-colors text-left"
            >
              <ConceptDot color={color} />
              <div className="min-w-0">
                <div className="font-mono text-xs font-medium text-text truncate">
                  {concept.name}
                </div>
                <div className="font-mono text-[10px] text-text-subtle truncate">
                  {concept.description}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function GenerationPane() {
  const prompt = useStore((s) => s.prompt);
  const setPrompt = useStore((s) => s.setPrompt);
  const isGenerating = useStore((s) => s.isGenerating);
  const generationError = useStore((s) => s.generationError);
  const tokens = useStore((s) => s.tokens);
  const cards = useStore((s) => s.cards);
  const clearOutput = useStore((s) => s.clearOutput);
  const { generate, stop } = useGeneration();
  const outputRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [tokens]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim()) generate(prompt.trim());
  };

  const showQuickStart = cards.length === 0 && tokens.length === 0 && !isGenerating;
  const progressPct = Math.min((tokens.length / MAX_TOKENS) * 100, 100);

  return (
    <div className="h-full flex flex-col min-w-0 min-h-0">
      <div className="flex items-center gap-3 border-b border-bg-border px-4 py-2 shrink-0">
        <AnimatePresence>
          {tokens.length > 0 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="ml-auto"
            >
              <Button variant="ghost" size="sm" onClick={clearOutput} disabled={isGenerating}>
                <Trash2 size={12} />
                Clear
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {isGenerating && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-3 px-4 py-1.5 border-b border-bg-border shrink-0"
          >
            <div className="flex-1 h-1 bg-bg-border rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-accent rounded-full"
                animate={{ width: `${progressPct}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <span className="font-mono text-[10px] text-text-subtle tabular-nums shrink-0">
              {tokens.length} tokens
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      <div
        ref={outputRef}
        className="flex-1 overflow-y-auto px-4 py-4 font-mono text-sm leading-relaxed flex flex-col"
      >
        {showQuickStart ? (
          <QuickStart />
        ) : tokens.length === 0 && !isGenerating ? (
          <p className="text-text-subtle italic">Output will appear here...</p>
        ) : (
          <TokenizedOutput tokens={tokens} />
        )}
        {generationError && (
          <p className="mt-2 font-mono text-xs text-danger">{generationError}</p>
        )}
      </div>

      <form
        onSubmit={handleSubmit}
        className="border-t border-bg-border p-3 flex gap-3 items-end"
      >
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              if (prompt.trim() && !isGenerating) generate(prompt.trim());
            }
          }}
          placeholder="Enter a prompt... (⌘Enter to generate)"
          rows={3}
          className="flex-1 resize-none rounded-lg border border-bg-border bg-bg-elevated px-3 py-2 font-mono text-sm text-text placeholder:text-text-subtle focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/30 transition-colors shadow-inner"
        />
        <div className="flex flex-col gap-2">
          <AnimatePresence mode="wait">
            {isGenerating ? (
              <motion.div
                key="stop"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
              >
                <Button type="button" variant="danger" onClick={stop}>
                  <Square size={12} />
                  Stop
                </Button>
              </motion.div>
            ) : (
              <motion.div
                key="generate"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
              >
                <Button
                  type="submit"
                  variant="primary"
                  disabled={!prompt.trim()}
                  className={prompt.trim() ? "shadow-glow-accent" : ""}
                >
                  <Play size={12} />
                  Generate
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </form>
    </div>
  );
}
