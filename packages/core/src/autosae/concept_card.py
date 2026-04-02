from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, Field
from safetensors import safe_open
from safetensors.torch import save_file

from autosae.exceptions import ConceptCardNotFoundError

logger = logging.getLogger(__name__)

_REGISTRY_CACHE: dict[str, ConceptCard] = {}

REGISTRY_CONCEPTS = frozenset(
    {
        "formality",
        "safety",
        "reasoning",
        "creativity",
        "conciseness",
        "coding",
        "empathy",
        "certainty",
    }
)


class ConceptCardMeta(BaseModel):
    model_id: str
    layer: int
    hidden_dim: int
    default_alpha: float = 1.0
    concept: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_revision: str | None = None
    transformers_version: str | None = None
    autosae_version: str | None = None
    p_value: float | None = None
    separability_score: float | None = None
    num_positive: int | None = None
    num_negative: int | None = None
    mean_positive_norm: float | None = None
    mean_negative_norm: float | None = None
    cosine_distance: float | None = None
    bootstrap_variance: float | None = None
    bootstrap_cov_2d: list[list[float]] | None = None
    bootstrap_pca_axes: list[list[float]] | None = None
    layer_selection: Literal["auto", "manual"] | None = None
    mean_hidden_norm: float | None = None

    @property
    def model_slug(self) -> str:
        return self.model_id.split("/")[-1].lower()


class ConceptCard:
    def __init__(self, meta: ConceptCardMeta, vector: torch.Tensor) -> None:
        if vector.ndim != 1 or vector.shape[0] != meta.hidden_dim:
            raise ValueError(
                f"Expected vector of shape ({meta.hidden_dim},), got {tuple(vector.shape)}"
            )
        self.meta = meta
        self.vector = vector.float()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"vector": self.vector},
            str(path),
            metadata={"meta": self.meta.model_dump_json()},
        )

    @classmethod
    def load(cls, path: str | Path) -> ConceptCard:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ConceptCard not found at: {path}")

        tensors: dict[str, torch.Tensor] = {}
        metadata: dict[str, str] = {}

        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118  safetensors API requires .keys()
                tensors[key] = f.get_tensor(key)
            metadata = f.metadata() or {}

        if "meta" not in metadata:
            raise ValueError(f"ConceptCard missing metadata header in: {path}")

        meta = ConceptCardMeta.model_validate_json(metadata["meta"])
        return cls(meta=meta, vector=tensors["vector"])

    @classmethod
    def from_registry(cls, concept: str, model: str = "llama-3.1-8b") -> ConceptCard:
        cache_key = f"{model}/{concept}"
        if cache_key in _REGISTRY_CACHE:
            return _REGISTRY_CACHE[cache_key]

        local_path = _local_registry_path(concept, model)
        if local_path.exists():
            card = cls.load(local_path)
            _REGISTRY_CACHE[cache_key] = card
            return card

        card = _fetch_from_hub(concept, model)
        _REGISTRY_CACHE[cache_key] = card
        return card

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        concept: str,
        model: str,
        token: str | None = None,
    ) -> ConceptCard:
        cache_key = f"{repo_id}/{model}/{concept}"
        if cache_key in _REGISTRY_CACHE:
            return _REGISTRY_CACHE[cache_key]

        from huggingface_hub import hf_hub_download

        local_file = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model}/{concept}.safetensors",
            token=token,
        )
        card = cls.load(local_file)
        _REGISTRY_CACHE[cache_key] = card
        return card

    def push_to_hub(self, repo_id: str, token: str | None = None) -> str:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            filename = f"{self.meta.concept}.safetensors"
            local_path = Path(tmp) / filename
            self.save(local_path)

            readme_content = _build_model_card(self.meta, repo_id)
            readme_path = Path(tmp) / "README.md"
            readme_path.write_text(readme_content)

            remote_path = f"{self.meta.model_slug}/{filename}"
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
            )
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
            )

        return f"https://huggingface.co/{repo_id}"

    def __repr__(self) -> str:
        return (
            f"ConceptCard(concept={self.meta.concept!r}, "
            f"model={self.meta.model_id!r}, "
            f"layer={self.meta.layer}, "
            f"alpha={self.meta.default_alpha})"
        )


def combine_cards(
    cards: dict[str, tuple[ConceptCard, float]],
    name: str,
    description: str = "",
) -> ConceptCard:
    if not cards:
        from autosae.exceptions import ConceptNotLoadedError

        raise ConceptNotLoadedError("No cards provided to combine.")

    cards_list = list(cards.values())
    model_ids = {card.meta.model_id for card, _ in cards_list}
    if len(model_ids) > 1:
        raise ValueError(
            f"Cards must share the same model_id, got: {', '.join(sorted(model_ids))}"
        )

    layer_set = {card.meta.layer for card, _ in cards_list}
    if len(layer_set) > 1:
        raise ValueError(
            f"Cards must target the same layer; found different layers: {sorted(layer_set)}"
        )

    ref_card, _ = cards_list[0]
    v_combined = torch.zeros(ref_card.meta.hidden_dim, dtype=torch.float32)
    for card, alpha in cards_list:
        v_combined = v_combined + alpha * card.vector.float()

    norm = v_combined.norm()
    if norm < 1e-6:
        raise ValueError("Concept vectors cancel out; cannot export zero vector.")

    v_norm = v_combined / norm
    default_alpha = norm.item()

    if not description:
        parts = [f"{c} x{alpha:.2g}" for c, (_, alpha) in cards.items()]
        description = "Combination of: " + ", ".join(parts)

    meta = ConceptCardMeta(
        model_id=ref_card.meta.model_id,
        layer=ref_card.meta.layer,
        hidden_dim=ref_card.meta.hidden_dim,
        concept=name,
        default_alpha=default_alpha,
        description=description,
    )
    return ConceptCard(meta, v_norm)


def _build_model_card(meta: ConceptCardMeta, repo_id: str) -> str:
    return f"""---
tags:
  - autosae
  - activation-steering
  - concept-card
  - {meta.concept}
  - model:{meta.model_slug}
model: {meta.model_id}
---

# {meta.concept} — AutoSAE Concept Card

**Concept:** `{meta.concept}`
**Model:** `{meta.model_id}`
**Layer:** {meta.layer}
**Default alpha:** {meta.default_alpha}

{meta.description}

## Usage

```python
from autosae import ConceptCard, Steerer

card = ConceptCard.from_hub("{repo_id}", "{meta.concept}", "{meta.model_slug}")
steerer = Steerer(model_id="{meta.model_id}")
steerer.load_card(card, alpha={meta.default_alpha})
output = steerer.generate("Your prompt here")
```
"""


def registry_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def _local_registry_path(concept: str, model: str) -> Path:
    return Path(__file__).parents[4] / "registry" / model / f"{concept}.safetensors"


def _fetch_from_hub(concept: str, model: str) -> ConceptCard:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        local_file = hf_hub_download(
            repo_id="sjoerdvink/autosae",
            filename=f"{model}/{concept}.safetensors",
        )
        return ConceptCard.load(local_file)
    except (EntryNotFoundError, RepositoryNotFoundError) as exc:
        available = ", ".join(sorted(REGISTRY_CONCEPTS))
        raise ConceptCardNotFoundError(
            f"Concept card '{concept}' for model '{model}' not found on HuggingFace Hub. "
            f"Available registry concepts: {available}."
        ) from exc
    except Exception as exc:
        logger.warning(f"Unexpected error fetching from Hub: {exc}")
        available = ", ".join(sorted(REGISTRY_CONCEPTS))
        raise ConceptCardNotFoundError(
            f"Failed to fetch concept card '{concept}' for model '{model}'. "
            f"Check network connection. Available registry concepts: {available}."
        ) from exc
