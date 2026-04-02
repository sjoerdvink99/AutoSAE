from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoadCardRequest(BaseModel):
    path: str | None = None
    registry_concept: str | None = None
    registry_model: str | None = None
    alpha: float | None = None

    model_config = {
        "json_schema_extra": {"examples": [{"registry_concept": "formality", "alpha": 2.0}]}
    }


class UpdateAlphaRequest(BaseModel):
    alpha: float = Field(
        ..., description="Steering coefficient. Negative values reverse the concept."
    )


class ConversationMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    prompt: str = ""
    messages: list[ConversationMessage] | None = None
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.01, le=5.0)
    seed: int | None = None
    greedy: bool = False
    system_prompt: str | None = Field(default="You are a helpful assistant.")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class CardInfo(BaseModel):
    concept: str
    model_id: str
    layer: int
    hidden_dim: int
    alpha: float
    description: str
    p_value: float | None = None
    separability_score: float | None = None
    layer_selection: str | None = None
    num_positive: int | None = None
    num_negative: int | None = None
    bootstrap_variance: float | None = None
    mean_hidden_norm: float | None = None


class GenerateResponse(BaseModel):
    text: str
    activations: dict[str, float]


class WsIncomingMessage(BaseModel):
    type: str
    prompt: str = ""
    messages: list[ConversationMessage] | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int | None = None
    greedy: bool = False
    system_prompt: str | None = Field(default="You are a helpful assistant.")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class WsTokenMessage(BaseModel):
    type: str = "token"
    token: str
    activations: dict[str, float]
    projection: tuple[float, float] | None = None
    done: bool = False


class WsDoneMessage(BaseModel):
    type: str = "done"
    done: bool = True


class WsErrorMessage(BaseModel):
    type: str = "error"
    message: str



class ExtractCardRequest(BaseModel):
    concept: str
    description: str = ""
    positive: list[str] = Field(..., min_length=1)
    negative: list[str] = Field(..., min_length=1)
    default_alpha: float = Field(default=1.0)
    layer_frac: float = Field(default=0.6, ge=0.0, le=1.0)
    auto_layer: bool = True
    use_robust_mean: bool = False


class ExtractCardResponse(BaseModel):
    concept: str
    model_id: str
    layer: int
    hidden_dim: int
    default_alpha: float
    description: str
    path: str
    p_value: float | None = None
    separability_score: float | None = None
    layer_selection: str | None = None


class ConceptGeometry(BaseModel):
    concepts: list[str]
    vectors_2d: list[tuple[float, float]]
    gram: list[list[float]]
    variance_ratio: tuple[float, float]
    projection_jacobian: list[list[float]]
    confidence_ellipses: dict[str, list[list[float]]] | None = None
    orthogonalized: bool = False
    projection_coverage: float = 0.0


class InverseProjectRequest(BaseModel):
    delta: tuple[float, float]
    mode: Literal["pca", "concept_plane"] = "pca"
    anchor_concepts: tuple[str, str] | None = None
    alpha_bounds: dict[str, tuple[float, float]] | None = None
    max_step: float | None = None


class InverseProjectResponse(BaseModel):
    alpha_deltas: dict[str, float]


class LayerSweepRequest(BaseModel):
    positive: list[str] = Field(..., min_length=1)
    negative: list[str] = Field(..., min_length=1)


class LayerScore(BaseModel):
    layer: int
    score: float


class LayerSweepResponse(BaseModel):
    layers: list[LayerScore]
    recommended_layer: int



class ConceptMapPoint(BaseModel):
    concept: str
    x: float
    y: float
    loaded: bool
    separability_score: float | None = None
    bootstrap_confidence: float | None = None
    model_id: str | None = None


class ConceptMapResponse(BaseModel):
    points: list[ConceptMapPoint]
    model_id: str
