import { useCallback, useEffect, useRef } from "react";
import { useStore } from "../stores/useStore";
import type { WsMessage, TokenChunk, TrajectoryPoint } from "../types";
import { MAX_NEW_TOKENS } from "../lib/constants";

const BATCH_SIZE = 8;
const BATCH_TIMEOUT_MS = 50;

function getWsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/generate`;
}

export function useGeneration() {
  const appendTokenBatch = useStore((s) => s.appendTokenBatch);
  const clearOutput = useStore((s) => s.clearOutput);
  const clearTrajectory = useStore((s) => s.clearTrajectory);
  const resetCanvasViewport = useStore((s) => s.resetCanvasViewport);
  const pushTrajectoryPoint = useStore((s) => s.pushTrajectoryPoint);
  const setGenerating = useStore((s) => s.setGenerating);
  const setGenerationError = useStore((s) => s.setGenerationError);
  const setPrompt = useStore((s) => s.setPrompt);
  const addUserTurn = useStore((s) => s.addUserTurn);
  const addAssistantTurn = useStore((s) => s.addAssistantTurn);
  const finalizeAssistantTurn = useStore((s) => s.finalizeAssistantTurn);

  const wsRef = useRef<WebSocket | null>(null);
  const batchRef = useRef<TokenChunk[]>([]);
  const trajectoryBatchRef = useRef<TrajectoryPoint[]>([]);
  const batchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tokenIndexRef = useRef<number>(0);

  const flushBatch = useCallback(() => {
    if (batchRef.current.length > 0) {
      appendTokenBatch(batchRef.current);
      batchRef.current = [];
    }
    if (trajectoryBatchRef.current.length > 0) {
      for (const p of trajectoryBatchRef.current) {
        pushTrajectoryPoint(p);
      }
      trajectoryBatchRef.current = [];
    }
    if (batchTimerRef.current) {
      clearTimeout(batchTimerRef.current);
      batchTimerRef.current = null;
    }
  }, [appendTokenBatch, pushTrajectoryPoint]);

  const generate = useCallback(
    (prompt: string, maxNewTokens = 512, temperature = 0.7) => {
      if (wsRef.current) {
        wsRef.current.close();
      }

      const allCompleted = useStore.getState().conversation.filter((t) => !t.isStreaming);
      let historyTurns = allCompleted.slice(-6);
      if (historyTurns.length > 0 && historyTurns[0].role !== "user") {
        historyTurns = historyTurns.slice(1);
      }
      const messages = [
        ...historyTurns.map((t) => ({
          role: t.role,
          content:
            t.role === "assistant" && t.content.length > 1200
              ? t.content.slice(0, 1200) + "\u2026"
              : t.content,
        })),
        { role: "user" as const, content: prompt },
      ];

      const assistantId = crypto.randomUUID();
      let tokenText = "";
      let finalized = false;

      const doFinalize = () => {
        if (!finalized) {
          finalized = true;
          finalizeAssistantTurn(assistantId, tokenText);
        }
      };

      addUserTurn(prompt);
      addAssistantTurn(assistantId);
      setPrompt("");

      clearOutput();
      clearTrajectory();
      resetCanvasViewport();
      setGenerationError(null);
      setGenerating(true);
      batchRef.current = [];
      trajectoryBatchRef.current = [];
      tokenIndexRef.current = 0;

      const ws = new WebSocket(getWsUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            type: "generate",
            messages,
            max_new_tokens: maxNewTokens,
            temperature,
          }),
        );
      };

      ws.onmessage = (event: MessageEvent<string>) => {
        const msg = JSON.parse(event.data) as WsMessage;

        if (msg.type === "token" && msg.token !== undefined) {
          tokenText += msg.token;
          const proj = msg.projection ?? null;
          batchRef.current.push({
            token: msg.token,
            activations: msg.activations ?? {},
            projection: proj,
          });
          if (proj !== null) {
            trajectoryBatchRef.current.push({
              index: tokenIndexRef.current,
              token: msg.token,
              x: proj[0],
              y: proj[1],
              activations: msg.activations ?? {},
            });
          }
          tokenIndexRef.current += 1;

          if (batchRef.current.length >= BATCH_SIZE) {
            flushBatch();
          } else if (!batchTimerRef.current) {
            batchTimerRef.current = setTimeout(flushBatch, BATCH_TIMEOUT_MS);
          }
        } else if (msg.type === "done") {
          flushBatch();
          doFinalize();
          setGenerating(false);
          ws.close();
        } else if (msg.type === "error") {
          flushBatch();
          doFinalize();
          setGenerationError(msg.message ?? "Generation failed");
          setGenerating(false);
          ws.close();
        }
      };

      ws.onerror = () => {
        flushBatch();
        setGenerationError("Connection lost");
        setGenerating(false);
      };

      ws.onclose = () => {
        flushBatch();
        doFinalize();
        setGenerating(false);
      };
    },
    [
      appendTokenBatch,
      clearOutput,
      clearTrajectory,
      resetCanvasViewport,
      setGenerating,
      setGenerationError,
      setPrompt,
      addUserTurn,
      addAssistantTurn,
      finalizeAssistantTurn,
      flushBatch,
    ],
  );

  const stop = useCallback(() => {
    flushBatch();
    wsRef.current?.close();
    wsRef.current = null;
    setGenerating(false);
  }, [setGenerating, flushBatch]);

  const tokensRef = useRef<TokenChunk[]>([]);
  const promptRef = useRef<string>("");

  const storeTokens = useStore((s) => s.tokens);
  const storePrompt = useStore((s) => s.prompt);

  useEffect(() => {
    tokensRef.current = storeTokens;
  }, [storeTokens]);

  useEffect(() => {
    promptRef.current = storePrompt;
  }, [storePrompt]);

  const regenerateFrom = useCallback(
    (tokenIndex: number) => {
      const prefixText = tokensRef.current
        .slice(0, tokenIndex)
        .map((t) => t.token)
        .join("");
      const newPrompt = promptRef.current + prefixText;
      const remaining = MAX_NEW_TOKENS - tokenIndex;
      generate(newPrompt, Math.max(remaining, 64));
    },
    [generate],
  );

  useEffect(() => {
    return () => {
      if (batchTimerRef.current) {
        clearTimeout(batchTimerRef.current);
      }
      wsRef.current?.close();
    };
  }, []);

  return { generate, stop, regenerateFrom };
}
