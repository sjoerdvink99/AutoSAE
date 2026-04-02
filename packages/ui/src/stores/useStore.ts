import { create } from "zustand";
import { CONCEPT_COLOR_PALETTE_DARK, CONCEPT_COLOR_PALETTE_LIGHT } from "../lib/constants";
import type {
  AlphaSnapshot,
  CanvasViewport,
  CardInfo,
  ConceptGeometry,
  ConversationTurn,
  EngineCapabilities,
  SteeringEvent,
  TokenChunk,
  TrajectoryPoint,
} from "../types";

const MAX_TOKENS = 2000;
const MAX_TRAJECTORY = 500;
const MAX_ALPHA_HISTORY = 50;

function assignColors(
  cards: CardInfo[],
  existing: Record<string, string>,
  theme: "dark" | "light"
): Record<string, string> {
  const palette = theme === "light" ? CONCEPT_COLOR_PALETTE_LIGHT : CONCEPT_COLOR_PALETTE_DARK;
  const result = { ...existing };
  const usedColors = new Set(Object.values(result));
  for (const card of cards) {
    if (!result[card.concept]) {
      const available = palette.filter((c) => !usedColors.has(c));
      const color = available[0] ?? palette[Object.keys(result).length % palette.length]!;
      result[card.concept] = color;
      usedColors.add(color);
    }
  }
  return result;
}

interface AppState {
  serverUrl: string;
  setServerUrl: (url: string) => void;

  capabilities: EngineCapabilities;
  setCapabilities: (caps: EngineCapabilities) => void;

  cards: CardInfo[];
  setCards: (cards: CardInfo[]) => void;
  updateCardAlpha: (concept: string, alpha: number) => void;
  removeCard: (concept: string) => void;

  conceptColors: Record<string, string>;

  prompt: string;
  setPrompt: (prompt: string) => void;

  tokens: TokenChunk[];
  isGenerating: boolean;
  generationError: string | null;
  appendToken: (chunk: TokenChunk) => void;
  appendTokenBatch: (chunks: TokenChunk[]) => void;
  clearOutput: () => void;
  setGenerating: (v: boolean) => void;
  setGenerationError: (err: string | null) => void;

  geometry: ConceptGeometry | null;
  setGeometry: (g: ConceptGeometry | null) => void;

  trajectory: TrajectoryPoint[];
  pushTrajectoryPoint: (p: TrajectoryPoint) => void;
  clearTrajectory: () => void;

  hoveredTokenIndex: number | null;
  setHoveredTokenIndex: (index: number | null) => void;

  alphaHistory: AlphaSnapshot[];
  pushAlphaSnapshot: () => void;
  restoreAlphaSnapshot: (index: number) => void;
  undoAlpha: () => void;

  selectedTokenIndex: number | null;
  setSelectedTokenIndex: (i: number | null) => void;

  selectedCardIndex: number | null;
  setSelectedCardIndex: (i: number | null) => void;

  recomputeDisplayColors: () => void;

  steeringEvents: SteeringEvent[];
  recordEvent: (event: Omit<SteeringEvent, "timestamp">) => void;
  clearSteeringEvents: () => void;
  sessionStart: number;

  canvasViewport: CanvasViewport;
  setCanvasViewport: (v: CanvasViewport) => void;
  resetCanvasViewport: () => void;

  trajectoryVisible: boolean;
  toggleTrajectory: () => void;

  gridVisible: boolean;
  toggleGrid: () => void;

  theme: "dark" | "light";
  toggleTheme: () => void;

  conversation: ConversationTurn[];
  addUserTurn: (content: string) => void;
  addAssistantTurn: (id: string) => void;
  finalizeAssistantTurn: (id: string, content: string) => void;
  clearConversation: () => void;

}

const defaultCapabilities: EngineCapabilities = {
  supports_steering: true,
  supports_extraction: true,
};

const getDefaultServerUrl = (): string =>
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

const getInitialTheme = (): "dark" | "light" => {
  const stored = localStorage.getItem("autosae-theme");
  if (stored === "dark" || stored === "light") return stored;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
};

export const useStore = create<AppState>((set) => ({
  serverUrl: getDefaultServerUrl(),
  setServerUrl: (url) => set({ serverUrl: url }),

  capabilities: defaultCapabilities,
  setCapabilities: (capabilities) => set({ capabilities }),

  cards: [],
  conceptColors: {},
  setCards: (cards) =>
    set((s) => ({
      cards,
      conceptColors: assignColors(cards, s.conceptColors, s.theme),
    })),
  updateCardAlpha: (concept, alpha) =>
    set((s) => ({
      cards: s.cards.map((c) => (c.concept === concept ? { ...c, alpha } : c)),
    })),
  removeCard: (concept) =>
    set((s) => {
      const newColors = { ...s.conceptColors };
      delete newColors[concept];
      return {
        cards: s.cards.filter((c) => c.concept !== concept),
        conceptColors: newColors,
      };
    }),

  prompt: "",
  setPrompt: (prompt) => set({ prompt }),

  tokens: [],
  isGenerating: false,
  generationError: null,
  appendToken: (chunk) =>
    set((s) => ({
      tokens: [...s.tokens, chunk].slice(-MAX_TOKENS),
    })),
  appendTokenBatch: (chunks) =>
    set((s) => ({
      tokens: [...s.tokens, ...chunks].slice(-MAX_TOKENS),
    })),
  clearOutput: () => set({ tokens: [], generationError: null }),
  setGenerating: (v) => set({ isGenerating: v }),
  setGenerationError: (err) => set({ generationError: err }),

  geometry: null,
  setGeometry: (g) => set({ geometry: g }),

  trajectory: [],
  pushTrajectoryPoint: (p) =>
    set((s) => ({
      trajectory: [...s.trajectory, p].slice(-MAX_TRAJECTORY),
    })),
  clearTrajectory: () => set({ trajectory: [] }),

  hoveredTokenIndex: null,
  setHoveredTokenIndex: (index) => set({ hoveredTokenIndex: index }),

  alphaHistory: [],
  pushAlphaSnapshot: () =>
    set((s) => ({
      alphaHistory: [
        ...s.alphaHistory,
        {
          timestamp: Date.now(),
          alphas: Object.fromEntries(s.cards.map((c) => [c.concept, c.alpha])),
        },
      ].slice(-MAX_ALPHA_HISTORY),
    })),
  restoreAlphaSnapshot: (index) =>
    set((s) => {
      const snapshot = s.alphaHistory[index];
      if (!snapshot) return {};
      return {
        cards: s.cards.map((c) =>
          c.concept in snapshot.alphas
            ? { ...c, alpha: snapshot.alphas[c.concept] ?? c.alpha }
            : c
        ),
      };
    }),
  undoAlpha: () =>
    set((s) => {
      if (s.alphaHistory.length < 2) return {};
      const snapshot = s.alphaHistory[s.alphaHistory.length - 2];
      if (!snapshot) return {};
      return {
        cards: s.cards.map((c) =>
          c.concept in snapshot.alphas
            ? { ...c, alpha: snapshot.alphas[c.concept] ?? c.alpha }
            : c
        ),
        alphaHistory: s.alphaHistory.slice(0, -1),
      };
    }),

  selectedTokenIndex: null,
  setSelectedTokenIndex: (i) => set({ selectedTokenIndex: i }),

  selectedCardIndex: null,
  setSelectedCardIndex: (i) => set({ selectedCardIndex: i }),

  recomputeDisplayColors: () =>
    set((s) => {
      const alphas = Object.fromEntries(s.cards.map((c) => [c.concept, c.alpha]));
      return {
        tokens: s.tokens.map((t) => {
          let maxIntensity = 0;
          for (const [concept, val] of Object.entries(t.activations)) {
            const alpha = alphas[concept] ?? 1;
            const eff = Math.abs(val) * Math.abs(alpha);
            if (eff > maxIntensity) maxIntensity = eff;
          }
          return { ...t, displayIntensity: maxIntensity };
        }),
      };
    }),

  steeringEvents: [],
  sessionStart: Date.now(),
  recordEvent: (event) =>
    set((s) => ({
      steeringEvents: [...s.steeringEvents, { ...event, timestamp: Date.now() }],
    })),
  clearSteeringEvents: () => set({ steeringEvents: [], sessionStart: Date.now() }),

  canvasViewport: { cx: 0, cy: 0, zoom: 1 },
  setCanvasViewport: (v) =>
    set({ canvasViewport: { ...v, zoom: Math.max(0.25, Math.min(8.0, v.zoom)) } }),
  resetCanvasViewport: () =>
    set({ canvasViewport: { cx: 0, cy: 0, zoom: 1 } }),

  trajectoryVisible: true,
  toggleTrajectory: () => set((s) => ({ trajectoryVisible: !s.trajectoryVisible })),

  gridVisible: false,
  toggleGrid: () => set((s) => ({ gridVisible: !s.gridVisible })),

  theme: getInitialTheme(),
  toggleTheme: () =>
    set((s) => {
      const next: "dark" | "light" = s.theme === "dark" ? "light" : "dark";
      localStorage.setItem("autosae-theme", next);
      document.documentElement.dataset.theme = next;
      const palette = next === "light" ? CONCEPT_COLOR_PALETTE_LIGHT : CONCEPT_COLOR_PALETTE_DARK;
      const newColors: Record<string, string> = {};
      s.cards.forEach((card, i) => {
        newColors[card.concept] = palette[i % palette.length]!;
      });
      return { theme: next, conceptColors: newColors };
    }),

  conversation: [],
  addUserTurn: (content) =>
    set((s) => ({
      conversation: [
        ...s.conversation,
        { id: crypto.randomUUID(), role: "user", content, isStreaming: false },
      ],
    })),
  addAssistantTurn: (id) =>
    set((s) => ({
      conversation: [
        ...s.conversation,
        { id, role: "assistant", content: "", isStreaming: true },
      ],
    })),
  finalizeAssistantTurn: (id, content) =>
    set((s) => ({
      conversation: s.conversation.map((t) =>
        t.id === id ? { ...t, content, isStreaming: false } : t
      ),
    })),
  clearConversation: () => set({ conversation: [] }),

}));
