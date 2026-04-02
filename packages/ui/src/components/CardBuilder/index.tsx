import { useState, useEffect } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { motion, AnimatePresence } from "framer-motion";
import { X, Plus, Trash2, Loader2, CheckCircle2 } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from "recharts";
import { cardsApi } from "../../api/cards";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import type { ExtractCardResponse, LayerSweepResponse } from "../../types";

interface Props {
  open: boolean;
  onClose: () => void;
}

type Step = "configure" | "prompts" | "layer-sweep" | "extract" | "result";

interface FormState {
  concept: string;
  description: string;
  default_alpha: number;
  layer_frac: number;
  positive: string[];
  negative: string[];
}

const MIN_PAIRS = 3;

export function CardBuilder({ open, onClose }: Props) {
  const queryClient = useQueryClient();
  const [step, setStep] = useState<Step>("configure");
  const [result, setResult] = useState<ExtractCardResponse | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [sweepResult, setSweepResult] = useState<LayerSweepResponse | null>(null);
  const [form, setForm] = useState<FormState>({
    concept: "",
    description: "",
    default_alpha: 1.0,
    layer_frac: 0.6,
    positive: ["", "", ""],
    negative: ["", "", ""],
  });

  const layerSweepMutation = useMutation({
    mutationFn: () =>
      cardsApi.layerSweep(
        form.positive.filter((s) => s.trim()),
        form.negative.filter((s) => s.trim())
      ),
    onSuccess: (data) => {
      setSweepResult(data);
      setSelectedLayer(data.recommended_layer);
    },
  });

  const extractMutation = useMutation({
    mutationFn: () => {
      const numLayers = sweepResult ? sweepResult.layers.length : null;
      const layerFrac =
        selectedLayer !== null && numLayers !== null
          ? selectedLayer / numLayers
          : form.layer_frac;
      return cardsApi.extract({
        concept: form.concept.trim(),
        description: form.description.trim(),
        positive: form.positive.filter((s) => s.trim()),
        negative: form.negative.filter((s) => s.trim()),
        default_alpha: form.default_alpha,
        layer_frac: Math.max(0.01, Math.min(0.99, layerFrac)),
        auto_layer: false,
      });
    },
    onSuccess: (data) => {
      setResult(data);
      setStep("result");
      queryClient.invalidateQueries({ queryKey: ["cards"] });
    },
  });

  useEffect(() => {
    if (step === "layer-sweep" && !layerSweepMutation.isPending && !sweepResult) {
      layerSweepMutation.mutate();
    }
  }, [step]);

  const handleClose = () => {
    setStep("configure");
    setResult(null);
    setSweepResult(null);
    setSelectedLayer(null);
    setForm({
      concept: "",
      description: "",
      default_alpha: 1.0,
      layer_frac: 0.6,
      positive: ["", "", ""],
      negative: ["", "", ""],
    });
    onClose();
  };

  const updateRow = (side: "positive" | "negative", idx: number, val: string) => {
    setForm((f) => {
      const arr = [...f[side]];
      arr[idx] = val;
      return { ...f, [side]: arr };
    });
  };

  const addRow = () =>
    setForm((f) => ({ ...f, positive: [...f.positive, ""], negative: [...f.negative, ""] }));

  const removeRow = (idx: number) =>
    setForm((f) => ({
      ...f,
      positive: f.positive.filter((_, i) => i !== idx),
      negative: f.negative.filter((_, i) => i !== idx),
    }));

  const validPairs = form.positive.filter((s, i) => s.trim() && form.negative[i]?.trim()).length;
  const canExtract = form.concept.trim() && validPairs >= MIN_PAIRS;

  const stepLabel: Record<Step, string> = {
    configure: "1 / 4",
    prompts: "2 / 4",
    "layer-sweep": "3 / 4",
    extract: "4 / 4",
    result: "Done",
  };

  const effectiveLayer = selectedLayer ?? sweepResult?.recommended_layer ?? null;

  return (
    <Dialog.Root open={open} onOpenChange={(v) => !v && handleClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-2xl -translate-x-1/2 -translate-y-1/2 rounded-xl border border-bg-border bg-bg-surface shadow-2xl focus:outline-none overflow-hidden">
          <div className="flex items-center justify-between border-b border-bg-border px-6 py-4">
            <div>
              <Dialog.Title className="font-mono text-sm font-semibold text-text">
                Create Concept Card
              </Dialog.Title>
              <p className="font-mono text-xs text-text-muted mt-0.5">{stepLabel[step]}</p>
            </div>
            <Dialog.Close asChild>
              <button className="text-text-subtle hover:text-text transition-colors">
                <X size={16} />
              </button>
            </Dialog.Close>
          </div>

          <div className="p-6 max-h-[70vh] overflow-y-auto">
            <AnimatePresence mode="wait">
              {step === "configure" && (
                <motion.div
                  key="configure"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="flex flex-col gap-4"
                >
                  <div className="flex flex-col gap-1.5">
                    <label className="font-mono text-xs text-text-muted uppercase tracking-wider">
                      Concept name *
                    </label>
                    <Input
                      size="md"
                      value={form.concept}
                      onChange={(e) => setForm((f) => ({ ...f, concept: e.target.value }))}
                      placeholder="e.g. formality, sentiment, verbosity"
                    />
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="font-mono text-xs text-text-muted uppercase tracking-wider">
                      Description
                    </label>
                    <Input
                      size="md"
                      value={form.description}
                      onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
                      placeholder="Human-readable description"
                    />
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="font-mono text-xs text-text-muted uppercase tracking-wider">
                      Default alpha: {form.default_alpha.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      min={0.1}
                      max={5}
                      step={0.1}
                      value={form.default_alpha}
                      onChange={(e) =>
                        setForm((f) => ({ ...f, default_alpha: parseFloat(e.target.value) }))
                      }
                      className="accent-accent"
                    />
                  </div>
                </motion.div>
              )}

              {step === "prompts" && (
                <motion.div
                  key="prompts"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="flex flex-col gap-3"
                >
                  <p className="font-mono text-xs text-text-muted">
                    Add contrastive prompt pairs. Min {MIN_PAIRS} valid pairs required. Valid: {validPairs}/{form.positive.length}
                  </p>

                  <div className="grid grid-cols-2 gap-2 mb-1">
                    <span className="font-mono text-xs text-text-muted uppercase tracking-wider text-center">
                      Positive (has concept)
                    </span>
                    <span className="font-mono text-xs text-text-muted uppercase tracking-wider text-center">
                      Negative (lacks concept)
                    </span>
                  </div>

                  <div className="flex flex-col gap-2">
                    {form.positive.map((pos, i) => (
                      <div key={i} className="flex items-start gap-2">
                        <div className="grid grid-cols-2 gap-2 flex-1">
                          <textarea
                            rows={2}
                            value={pos}
                            onChange={(e) => updateRow("positive", i, e.target.value)}
                            placeholder="Formal, professional text..."
                            className="resize-none rounded border border-bg-border bg-bg-elevated px-2 py-1.5 font-mono text-xs text-text placeholder:text-text-subtle focus:border-accent/50 focus:outline-none"
                          />
                          <textarea
                            rows={2}
                            value={form.negative[i] ?? ""}
                            onChange={(e) => updateRow("negative", i, e.target.value)}
                            placeholder="Casual, informal text..."
                            className="resize-none rounded border border-bg-border bg-bg-elevated px-2 py-1.5 font-mono text-xs text-text placeholder:text-text-subtle focus:border-accent/50 focus:outline-none"
                          />
                        </div>
                        {form.positive.length > MIN_PAIRS && (
                          <button
                            onClick={() => removeRow(i)}
                            className="mt-1 p-1 text-text-subtle hover:text-danger transition-colors"
                          >
                            <Trash2 size={12} />
                          </button>
                        )}
                      </div>
                    ))}
                  </div>

                  <Button variant="ghost" size="sm" onClick={addRow} className="self-start">
                    <Plus size={12} />
                    Add pair
                  </Button>
                </motion.div>
              )}

              {step === "layer-sweep" && (
                <motion.div
                  key="layer-sweep"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="flex flex-col gap-4"
                >
                  <p className="font-mono text-xs text-text-muted">
                    Scanning all layers to find the optimal extraction point.
                  </p>

                  {layerSweepMutation.isPending && (
                    <div className="flex flex-col items-center gap-3 py-8">
                      <Loader2 size={28} className="text-accent animate-spin" />
                      <p className="font-mono text-xs text-text-muted">Running layer sweep…</p>
                    </div>
                  )}

                  {layerSweepMutation.isError && (
                    <div className="flex flex-col items-center gap-2 py-4">
                      <p className="font-mono text-xs text-danger">Sweep failed</p>
                      <p className="font-mono text-xs text-text-muted">
                        {(layerSweepMutation.error as Error).message}
                      </p>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => layerSweepMutation.mutate()}
                      >
                        Retry
                      </Button>
                    </div>
                  )}

                  {sweepResult && (
                    <div className="flex flex-col gap-3">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs text-text-muted">
                          Recommended layer:
                        </span>
                        <span className="font-mono text-xs text-accent font-semibold">
                          {effectiveLayer}
                        </span>
                        {effectiveLayer !== sweepResult.recommended_layer && (
                          <span className="font-mono text-[10px] text-text-subtle">
                            (recommended: {sweepResult.recommended_layer})
                          </span>
                        )}
                      </div>

                      <div style={{ height: 160 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={sweepResult.layers}
                            margin={{ top: 4, right: 4, left: -20, bottom: 0 }}
                            onClick={(d) => {
                              if (d?.activePayload?.[0]) {
                                setSelectedLayer(
                                  (d.activePayload[0].payload as { layer: number }).layer
                                );
                              }
                            }}
                          >
                            <XAxis
                              dataKey="layer"
                              tick={{ fontSize: 8, fontFamily: "monospace", fill: "var(--color-text-subtle)" }}
                              interval={Math.floor(sweepResult.layers.length / 8)}
                            />
                            <YAxis
                              tick={{ fontSize: 8, fontFamily: "monospace", fill: "var(--color-text-subtle)" }}
                              width={36}
                            />
                            <Tooltip
                              contentStyle={{
                                background: "var(--color-bg-surface)",
                                border: "1px solid var(--color-bg-border)",
                                borderRadius: 4,
                                fontSize: 10,
                                fontFamily: "monospace",
                                color: "var(--color-text)",
                              }}
                              formatter={(v: number) => [v.toFixed(3), "Fisher score"]}
                              labelFormatter={(l) => `Layer ${l}`}
                            />
                            <Bar dataKey="score" radius={[2, 2, 0, 0]} style={{ cursor: "pointer" }}>
                              {sweepResult.layers.map((entry) => (
                                <Cell
                                  key={entry.layer}
                                  fill={
                                    entry.layer === effectiveLayer
                                      ? "var(--color-accent)"
                                      : entry.layer === sweepResult.recommended_layer
                                      ? "var(--color-accent-dim)"
                                      : "var(--color-bg-elevated)"
                                  }
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      <p className="font-mono text-[10px] text-text-subtle text-center">
                        Click a bar to override layer selection
                      </p>
                    </div>
                  )}
                </motion.div>
              )}

              {step === "extract" && (
                <motion.div
                  key="extract"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col items-center gap-4 py-8"
                >
                  {extractMutation.isPending ? (
                    <>
                      <Loader2 size={32} className="text-accent animate-spin" />
                      <p className="font-mono text-sm text-text-muted">
                        Extracting concept vector for &ldquo;{form.concept}&rdquo;...
                      </p>
                      <p className="font-mono text-xs text-text-subtle text-center max-w-xs">
                        Running forward passes on {validPairs} prompt pairs. This may take a moment.
                      </p>
                    </>
                  ) : extractMutation.isError ? (
                    <>
                      <p className="font-mono text-sm text-danger">Extraction failed</p>
                      <p className="font-mono text-xs text-text-muted">
                        {(extractMutation.error as Error).message}
                      </p>
                      <Button variant="ghost" size="sm" onClick={() => setStep("layer-sweep")}>
                        Back
                      </Button>
                    </>
                  ) : null}
                </motion.div>
              )}

              {step === "result" && result && (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex flex-col gap-4"
                >
                  <div className="flex items-center gap-3">
                    <CheckCircle2 size={24} className="text-accent" />
                    <div>
                      <p className="font-mono text-sm font-semibold text-text">
                        Card extracted successfully
                      </p>
                      <p className="font-mono text-xs text-text-muted">
                        Saved to {result.path} and loaded into engine
                      </p>
                    </div>
                  </div>

                  <div className="rounded-lg border border-bg-border bg-bg-elevated p-4 font-mono text-xs text-text-muted space-y-1">
                    <div><span className="text-text">concept</span> {result.concept}</div>
                    <div><span className="text-text">model</span> {result.model_id}</div>
                    <div><span className="text-text">layer</span> {result.layer}</div>
                    <div><span className="text-text">hidden_dim</span> {result.hidden_dim}</div>
                    <div><span className="text-text">default_alpha</span> {result.default_alpha}</div>
                    {result.p_value !== null && result.p_value !== undefined && (
                      <div><span className="text-text">p_value</span> {result.p_value.toFixed(4)}</div>
                    )}
                    {result.separability_score !== null && result.separability_score !== undefined && (
                      <div><span className="text-text">fisher_score</span> {result.separability_score.toFixed(4)}</div>
                    )}
                    {result.layer_selection && (
                      <div><span className="text-text">layer_selection</span> {result.layer_selection}</div>
                    )}
                  </div>

                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="flex justify-between border-t border-bg-border px-6 py-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                if (step === "prompts") setStep("configure");
                else if (step === "layer-sweep") setStep("prompts");
                else if (step === "extract") setStep("layer-sweep");
                else handleClose();
              }}
            >
              {step === "result" ? "Close" : step === "configure" ? "Cancel" : "Back"}
            </Button>

            {step === "configure" && (
              <Button
                variant="primary"
                size="sm"
                disabled={!form.concept.trim()}
                onClick={() => setStep("prompts")}
              >
                Next
              </Button>
            )}
            {step === "prompts" && (
              <Button
                variant="primary"
                size="sm"
                disabled={!canExtract}
                onClick={() => {
                  setSweepResult(null);
                  setSelectedLayer(null);
                  setStep("layer-sweep");
                }}
              >
                Next
              </Button>
            )}
            {step === "layer-sweep" && sweepResult && (
              <Button
                variant="primary"
                size="sm"
                onClick={() => {
                  setStep("extract");
                  extractMutation.mutate();
                }}
              >
                Extract Card
              </Button>
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
