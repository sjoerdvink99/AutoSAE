import { useState, useCallback } from "react";
import { X } from "lucide-react";
import { useStore } from "../../stores/useStore";
import { Button } from "../ui/Button";
import type { TokenChunk } from "../../types";

interface Props {
  onClose: () => void;
}

function lcsAlign(a: TokenChunk[], b: TokenChunk[]): [TokenChunk | null, TokenChunk | null][] {
  const n = a.length;
  const m = b.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      dp[i]![j] =
        a[i - 1]!.token === b[j - 1]!.token
          ? (dp[i - 1]![j - 1] ?? 0) + 1
          : Math.max(dp[i - 1]![j] ?? 0, dp[i]![j - 1] ?? 0);
    }
  }

  const result: [TokenChunk | null, TokenChunk | null][] = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1]!.token === b[j - 1]!.token) {
      result.unshift([a[i - 1]!, b[j - 1]!]);
      i--;
      j--;
    } else if (j > 0 && (i === 0 || (dp[i]![j - 1] ?? 0) >= (dp[i - 1]![j] ?? 0))) {
      result.unshift([null, b[j - 1]!]);
      j--;
    } else {
      result.unshift([a[i - 1]!, null]);
      i--;
    }
  }
  return result;
}

export function DiffView({ onClose }: Props) {
  const tokens = useStore((s) => s.tokens);
  const [baseline, setBaseline] = useState<TokenChunk[] | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const runBaseline = useCallback(async () => {
    setIsRunning(true);
    const prompt = useStore.getState().prompt;
    const serverUrl = useStore.getState().serverUrl;
    try {
      const res = await fetch(`${serverUrl}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_new_tokens: Math.min(tokens.length, 256), temperature: 0.0, alphas: {} }),
      });
      if (!res.ok) return;
      const data = (await res.json()) as { tokens: { token: string; activations: Record<string, number> }[] };
      setBaseline(
        data.tokens.map((t) => ({ token: t.token, activations: t.activations, projection: null }))
      );
    } finally {
      setIsRunning(false);
    }
  }, [tokens.length]);

  const aligned = baseline ? lcsAlign(baseline, tokens) : null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-bg-elevated border border-bg-border rounded-xl shadow-2xl w-full max-w-3xl max-h-[80vh] flex flex-col overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-3 border-b border-bg-border shrink-0">
          <span className="font-mono text-sm font-semibold text-text">Before / After Comparison</span>
          <button onClick={onClose} className="ml-auto text-text-subtle hover:text-text">
            <X size={16} />
          </button>
        </div>

        {!baseline ? (
          <div className="flex flex-col items-center gap-4 p-8">
            <p className="font-mono text-xs text-text-muted text-center leading-relaxed">
              Generate a baseline (all alphas=0) with the same prompt to compare.
            </p>
            <Button variant="primary" onClick={runBaseline} disabled={isRunning}>
              {isRunning ? "Generating baseline..." : "Generate baseline"}
            </Button>
          </div>
        ) : (
          <div className="flex flex-1 min-h-0 gap-0">
            <div className="flex-1 overflow-y-auto p-4 border-r border-bg-border">
              <div className="font-mono text-[11px] text-text-subtle mb-2">Baseline (α=0)</div>
              <div className="font-mono text-sm leading-relaxed">
                {aligned?.map(([base], i) =>
                  base ? (
                    <span key={i} className="text-text-muted">{base.token}</span>
                  ) : (
                    <span key={i} className="bg-danger/20 text-danger rounded px-0.5">​</span>
                  )
                )}
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              <div className="font-mono text-[11px] text-text-subtle mb-2">Steered</div>
              <div className="font-mono text-sm leading-relaxed">
                {aligned?.map(([base, steered], i) =>
                  steered ? (
                    <span
                      key={i}
                      className={base ? "text-text" : "bg-accent/20 text-accent rounded px-0.5"}
                    >
                      {steered.token}
                    </span>
                  ) : (
                    <span key={i} className="opacity-0">​</span>
                  )
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
