from __future__ import annotations

from dataclasses import dataclass, field

from autosae.concept_card import ConceptCard


@dataclass
class HubCardResult:
    repo_id: str
    concept: str
    model_id: str
    description: str = ""
    downloads: int = 0
    tags: list[str] = field(default_factory=list)


def search_hub(query: str = "", limit: int = 20) -> list[HubCardResult]:
    from huggingface_hub import HfApi

    api = HfApi()
    models = api.list_models(
        filter="autosae",
        search=query or None,
        limit=limit,
        sort="downloads",
    )

    results: list[HubCardResult] = []
    for model in models:
        tags = list(model.tags or [])
        concept = next(
            (t for t in tags if t not in ("autosae", "activation-steering", "concept-card")), ""
        )
        model_id = next((t.removeprefix("model:") for t in tags if t.startswith("model:")), "")
        results.append(
            HubCardResult(
                repo_id=model.id,
                concept=concept,
                model_id=model_id,
                description=getattr(model, "description", "") or "",
                downloads=getattr(model, "downloads", 0) or 0,
                tags=tags,
            )
        )
    return results


def download_card(repo_id: str, concept: str, model: str) -> ConceptCard:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{model}/{concept}.safetensors",
    )
    return ConceptCard.load(local_path)
