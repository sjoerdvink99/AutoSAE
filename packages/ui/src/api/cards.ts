import { api } from "./client";
import type {
  CardInfo,
  ConceptGeometry,
  ConceptMapResponse,
  EngineCapabilities,
  ExtractCardRequest,
  ExtractCardResponse,
  LayerSweepResponse,
} from "../types";

export const cardsApi = {
  list: () => api.get<CardInfo[]>("/cards"),

  load: (params: { path?: string; registry_concept?: string; registry_model?: string; alpha?: number }) =>
    api.post<{ status: string }>("/cards/load", params),

  unload: (concept: string) => api.delete<void>(`/cards/${encodeURIComponent(concept)}`),

  setAlpha: (concept: string, alpha: number) =>
    api.patch<{ concept: string; alpha: number }>(
      `/cards/${encodeURIComponent(concept)}/alpha`,
      { alpha }
    ),

  batchSetAlpha: (alphas: Record<string, number>) =>
    Promise.all(
      Object.entries(alphas).map(([concept, alpha]) =>
        api.patch<{ concept: string; alpha: number }>(
          `/cards/${encodeURIComponent(concept)}/alpha`,
          { alpha }
        )
      )
    ),

  extract: (req: ExtractCardRequest) =>
    api.post<ExtractCardResponse>("/cards/extract", req),

  layerSweep: (positive: string[], negative: string[]) =>
    api.post<LayerSweepResponse>("/cards/layer-sweep", { positive, negative }),
};

export const healthApi = {
  get: () => api.get<{ status: string; version: string; capabilities: EngineCapabilities }>("/health"),
};

export const conceptMapApi = {
  get: () => api.get<ConceptMapResponse | null>("/concepts/map"),
};

export const geometryApi = {
  get: () => api.get<ConceptGeometry | null>("/geometry"),
  inverseProject: (
    delta: [number, number],
    mode: "pca" | "concept_plane",
    anchor_concepts?: [string, string]
  ) =>
    api.post<{ alpha_deltas: Record<string, number> }>("/geometry/inverse", {
      delta,
      mode,
      anchor_concepts: anchor_concepts ?? null,
    }),
};
