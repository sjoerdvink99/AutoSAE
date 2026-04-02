import { useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { motion, AnimatePresence } from "framer-motion";
import { X, Download } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { cardsApi } from "../../api/cards";
import { REGISTRY_CONCEPTS, getConceptColor } from "../../lib/constants";
import { Button } from "../ui/Button";
import { Input, SearchInput } from "../ui/Input";
import { ConceptDot } from "../ui/ConceptDot";
import { cn } from "../../lib/cn";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function CardLoader({ open, onClose }: Props) {
  const [customPath, setCustomPath] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const queryClient = useQueryClient();

  const loadMutation = useMutation({
    mutationFn: (params: { path?: string; registry_concept?: string; registry_model?: string }) =>
      cardsApi.load(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["cards"] });
      queryClient.invalidateQueries({ queryKey: ["geometry"] });
      onClose();
    },
  });

  const filteredConcepts = REGISTRY_CONCEPTS.filter(
    (c) =>
      !searchQuery ||
      c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      c.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Dialog.Root open={open} onOpenChange={(v) => !v && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40" />
        <AnimatePresence>
          {open && (
            <Dialog.Content asChild forceMount>
              <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none focus:outline-none">
                <motion.div
                  initial={{ opacity: 0, scale: 0.95, y: 8 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95, y: 8 }}
                  transition={{ duration: 0.2 }}
                  className="w-full max-w-md rounded-xl border border-bg-border bg-bg-surface p-6 shadow-2xl pointer-events-auto"
                >
                  <div className="flex items-center justify-between mb-4">
                    <Dialog.Title className="font-mono text-sm font-semibold text-text">
                      Load Concept Card
                    </Dialog.Title>
                    <Dialog.Close asChild>
                      <button className="text-text-subtle hover:text-text transition-colors">
                        <X size={16} />
                      </button>
                    </Dialog.Close>
                  </div>

                  <div className="mb-3">
                    <SearchInput
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search concepts..."
                      className="w-full"
                    />
                  </div>

                  <div className="mb-4">
                    <p className="font-mono text-xs text-text-muted mb-2 uppercase tracking-wider">
                      Registry
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      <AnimatePresence>
                        {filteredConcepts.map((concept) => {
                          const color = getConceptColor(concept.name);
                          return (
                            <motion.button
                              key={concept.name}
                              layout
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              exit={{ opacity: 0 }}
                              onClick={() =>
                                loadMutation.mutate({
                                  registry_concept: concept.name,
                                  registry_model: concept.model,
                                })
                              }
                              disabled={loadMutation.isPending}
                              whileHover={{ y: -2 }}
                              className={cn(
                                "flex flex-col items-start gap-0.5 rounded-lg border border-bg-border p-3 text-left transition-colors",
                                "hover:border-accent/40 hover:bg-accent-dim",
                                "disabled:opacity-40 disabled:cursor-not-allowed"
                              )}
                              style={{ boxShadow: undefined }}
                              onMouseEnter={(e) => {
                                (e.currentTarget as HTMLElement).style.boxShadow = `0 4px 16px ${color}22`;
                              }}
                              onMouseLeave={(e) => {
                                (e.currentTarget as HTMLElement).style.boxShadow = "";
                              }}
                            >
                              <div className="flex items-center gap-1.5">
                                <ConceptDot color={color} />
                                <span className="font-mono text-sm text-text">{concept.name}</span>
                              </div>
                              <span className="font-mono text-xs text-text-muted">{concept.description}</span>
                            </motion.button>
                          );
                        })}
                      </AnimatePresence>
                      {filteredConcepts.length === 0 && (
                        <p className="col-span-2 font-mono text-xs text-text-subtle text-center py-4">
                          No concepts match &ldquo;{searchQuery}&rdquo;
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="border-t border-bg-border pt-4">
                    <p className="font-mono text-xs text-text-muted mb-2 uppercase tracking-wider">
                      Custom path
                    </p>
                    <div className="flex gap-2">
                      <Input
                        value={customPath}
                        onChange={(e) => setCustomPath(e.target.value)}
                        placeholder="/path/to/card.safetensors"
                        className="flex-1"
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={!customPath.trim() || loadMutation.isPending}
                        onClick={() => loadMutation.mutate({ path: customPath.trim() })}
                      >
                        <Download size={12} />
                        Load
                      </Button>
                    </div>
                  </div>

                  {loadMutation.isError && (
                    <p className="mt-3 font-mono text-xs text-danger">
                      {(loadMutation.error as Error).message}
                    </p>
                  )}
                </motion.div>
              </div>
            </Dialog.Content>
          )}
        </AnimatePresence>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
