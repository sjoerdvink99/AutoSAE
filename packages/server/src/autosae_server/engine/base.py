from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GenerationChunk:
    token: str = ""
    activations: dict[str, float] = field(default_factory=dict)
    projection: tuple[float, float] | None = None
    error: str | None = None


@dataclass
class LoadedCardInfo:
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


@dataclass
class EngineCapabilities:
    supports_steering: bool
    supports_extraction: bool


class InferenceEngine(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    async def load_card(
        self,
        path: str | None,
        registry_concept: str | None,
        registry_model: str | None,
        alpha: float | None,
    ) -> None: ...

    @abstractmethod
    async def unload_card(self, concept: str) -> None: ...

    @abstractmethod
    async def set_alpha(self, concept: str, alpha: float) -> None: ...

    @abstractmethod
    async def get_card_infos(self) -> list[LoadedCardInfo]: ...

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.9,
        seed: int | None = None,
        greedy: bool = False,
        steer_prompt: bool = True,
        messages: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        repetition_penalty: float = 1.0,
    ) -> AsyncIterator[GenerationChunk]: ...

    @abstractmethod
    async def extract_card(
        self,
        concept: str,
        description: str,
        positive: list[str],
        negative: list[str],
        default_alpha: float,
        layer_frac: float,
        auto_layer: bool,
        use_robust_mean: bool,
    ) -> tuple[str, LoadedCardInfo]: ...

    @abstractmethod
    async def get_geometry(self) -> dict[str, object] | None: ...

    @abstractmethod
    async def inverse_project(
        self,
        delta: tuple[float, float],
        mode: Literal["pca", "concept_plane"],
        anchor_concepts: tuple[str, str] | None,
        alpha_bounds: dict[str, tuple[float, float]] | None,
        max_step: float | None,
    ) -> dict[str, float]: ...

    @abstractmethod
    async def layer_sweep(
        self,
        positive: list[str],
        negative: list[str],
    ) -> tuple[list[tuple[int, float]], int]: ...

    @abstractmethod
    async def get_card_path(self, concept: str) -> str | None: ...

    @abstractmethod
    async def get_concept_map(self) -> dict[str, object] | None: ...

    async def shutdown(self) -> None:  # noqa: B027
        pass
