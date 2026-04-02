import { useRef, useEffect, useState } from "react";
import { Play, Square, Trash2, Cpu, GitCompare } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useStore } from "../../stores/useStore";
import { useGeneration } from "../../hooks/useGeneration";
import { cardsApi } from "../../api/cards";
import { REGISTRY_CONCEPTS, getConceptColor } from "../../lib/constants";
import { Button } from "../ui/Button";
import { ConceptDot } from "../ui/ConceptDot";
import { DiffView } from "./DiffView";
import { UserMessage, AssistantMessage } from "./ChatMessage";

const MAX_TOKENS = 512;

function QuickStart() {
  const queryClient = useQueryClient();
  const setCards = useStore((s) => s.setCards);
  const theme = useStore((s) => s.theme);

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
    <div className="flex flex-col items-center justify-center flex-1 gap-4 p-6">
      <div className="flex flex-col items-center gap-2">
        <Cpu size={24} className="text-accent opacity-80" />
        <h3 className="font-mono text-sm font-semibold text-text">Start Steering</h3>
        <p className="font-mono text-xs text-text-subtle text-center max-w-xs leading-relaxed">
          Load a concept card to begin. Click one below to get started.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-2 w-full max-w-xs">
        {REGISTRY_CONCEPTS.map((concept) => {
          const color = getConceptColor(concept.name, theme);
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
              className="flex items-center gap-2 px-3 py-2 rounded-lg border border-bg-border bg-bg-surface hover:border-bg-elevated transition-colors text-left"
            >
              <ConceptDot color={color} />
              <div className="min-w-0">
                <div className="font-mono text-xs font-medium text-text truncate">{concept.name}</div>
                <div className="font-mono text-[10px] text-text-subtle truncate">{concept.description}</div>
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
  const conversation = useStore((s) => s.conversation);
  const clearOutput = useStore((s) => s.clearOutput);
  const clearConversation = useStore((s) => s.clearConversation);
  const { generate, stop } = useGeneration();
  const threadRef = useRef<HTMLDivElement>(null);
  const [showDiff, setShowDiff] = useState(false);

  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
    }
  }, [conversation, tokens]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !isGenerating) generate(prompt.trim());
  };

  const showQuickStart = cards.length === 0 && conversation.length === 0 && !isGenerating;
  const progressPct = Math.min((tokens.length / MAX_TOKENS) * 100, 100);
  const hasContent = conversation.length > 0 || tokens.length > 0;

  return (
    <div className="h-full flex flex-col min-w-0 min-h-0">
      {showDiff && <DiffView onClose={() => setShowDiff(false)} />}

      <div
        ref={threadRef}
        className="flex-1 overflow-y-auto px-4 py-4 flex flex-col min-h-0"
      >
        {showQuickStart ? (
          <QuickStart />
        ) : (
          <>
            {conversation.map((turn) =>
              turn.role === "user" ? (
                <UserMessage key={turn.id} content={turn.content} />
              ) : (
                <AssistantMessage key={turn.id} turn={turn} />
              ),
            )}
            {generationError && (
              <p className="font-mono text-xs text-danger mb-4">{generationError}</p>
            )}
          </>
        )}
      </div>

      <AnimatePresence>
        {isGenerating && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-3 px-4 py-1.5 border-t border-bg-border shrink-0"
          >
            <div className="flex-1 h-1 bg-bg-border rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-accent rounded-full"
                animate={{ width: `${progressPct}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <span className="font-mono text-[11px] text-text-subtle tabular-nums shrink-0">
              {tokens.length} tokens
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      <form
        onSubmit={handleSubmit}
        className="border-t border-bg-border p-3 flex gap-3 items-end shrink-0"
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
          placeholder="Message... (⌘Enter to send)"
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
                key="send"
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
                  Send
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
          {hasContent && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                clearConversation();
                clearOutput();
              }}
              disabled={isGenerating}
            >
              <Trash2 size={12} />
              New
            </Button>
          )}
          {tokens.length > 0 && cards.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowDiff(true)}
              disabled={isGenerating}
            >
              <GitCompare size={12} />
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}
