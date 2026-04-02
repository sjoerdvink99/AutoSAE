export interface CardInfo {
  concept: string;
  model_id: string;
  layer: number;
  hidden_dim: number;
  alpha: number;
  description: string;
  p_value: number | null;
  separability_score: number | null;
  layer_selection: string | null;
  num_positive: number | null;
  num_negative: number | null;
  bootstrap_variance: number | null;
  mean_hidden_norm: number | null;
}

export interface TokenChunk {
  token: string;
  activations: Record<string, number>;
  projection: [number, number] | null;
  displayIntensity?: number;
}

export interface WsMessage {
  type: "token" | "done" | "error";
  token?: string;
  activations?: Record<string, number>;
  projection?: [number, number] | null;
  done?: boolean;
  message?: string;
}

export interface ConceptGeometry {
  concepts: string[];
  vectors_2d: Array<[number, number]>;
  gram: number[][];
  variance_ratio: [number, number];
  projection_jacobian: number[][];
  confidence_ellipses: Record<string, number[][]> | null;
  orthogonalized: boolean;
  projection_coverage: number;
}

export interface TrajectoryPoint {
  index: number;
  token: string;
  x: number;
  y: number;
  activations: Record<string, number>;
}

export interface EngineCapabilities {
  supports_steering: boolean;
  supports_extraction: boolean;
  model_id?: string | null;
}

export interface ExtractCardRequest {
  concept: string;
  description: string;
  positive: string[];
  negative: string[];
  default_alpha: number;
  layer_frac: number;
  auto_layer?: boolean;
  use_robust_mean?: boolean;
}

export interface ExtractCardResponse {
  concept: string;
  model_id: string;
  layer: number;
  hidden_dim: number;
  default_alpha: number;
  description: string;
  path: string;
  p_value: number | null;
  separability_score: number | null;
  layer_selection: string | null;
}


export interface LayerScore {
  layer: number;
  score: number;
}

export interface LayerSweepResponse {
  layers: LayerScore[];
  recommended_layer: number;
}

export interface AlphaSnapshot {
  timestamp: number;
  alphas: Record<string, number>;
}

export interface CanvasViewport {
  cx: number;
  cy: number;
  zoom: number;
}

export interface TokenCluster {
  startIndex: number;
  endIndex: number;
  dominantConcept: string | null;
  avgActivations: Record<string, number>;
}


export interface ConversationTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  isStreaming: boolean;
}

export interface ConceptMapPoint {
  concept: string;
  x: number;
  y: number;
  loaded: boolean;
  separability_score: number | null;
  bootstrap_confidence: number | null;
  model_id: string | null;
}

export interface ConceptMapResponse {
  points: ConceptMapPoint[];
  model_id: string;
}

export type SteeringDirection = "more" | "less";

export interface SteeringEvent {
  timestamp: number;
  type: "alpha_change" | "token_click" | "space_click" | "regenerate" | "selection_steer";
  payload: Record<string, unknown>;
}
