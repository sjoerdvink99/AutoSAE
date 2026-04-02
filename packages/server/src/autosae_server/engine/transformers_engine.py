from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from autosae import ConceptCard, Steerer
from autosae.concept_card import registry_slug
from autosae.dataset import ContrastiveDataset
from autosae.extractor import Extractor
from autosae_server.engine.base import (
    EngineCapabilities,
    GenerationChunk,
    InferenceEngine,
    LoadedCardInfo,
)


class TransformersEngine(InferenceEngine):
    def __init__(self, model_id: str, device: str = "auto", load_in_4bit: bool = False) -> None:
        self._steerer = Steerer(model_id=model_id, device=device, load_in_4bit=load_in_4bit)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autosae-gen")
        self._cards_dir = Path("./cards")
        self._cards_dir.mkdir(exist_ok=True)

    @property
    def model_id(self) -> str:
        return self._steerer.model_id

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_steering=True,
            supports_extraction=True,
        )

    async def load_card(
        self,
        path: str | None,
        registry_concept: str | None,
        registry_model: str | None,
        alpha: float | None,
    ) -> None:
        resolved_model = registry_model or registry_slug(self._steerer.model_id)
        loop = asyncio.get_running_loop()
        card = await loop.run_in_executor(
            self._executor,
            lambda: self._load_card_sync(path, registry_concept, resolved_model),
        )
        self._steerer.load_card(card, alpha=alpha)

    def _load_card_sync(
        self, path: str | None, registry_concept: str | None, registry_model: str
    ) -> ConceptCard:
        if path is not None:
            return ConceptCard.load(path)
        if registry_concept is not None:
            return ConceptCard.from_registry(registry_concept, model=registry_model)
        raise ValueError("Either 'path' or 'registry_concept' must be provided.")

    async def unload_card(self, concept: str) -> None:
        self._steerer.unload_card(concept)

    async def set_alpha(self, concept: str, alpha: float) -> None:
        self._steerer.set_alpha(concept, alpha)

    async def get_card_infos(self) -> list[LoadedCardInfo]:
        return [
            LoadedCardInfo(
                concept=card.meta.concept,
                model_id=card.meta.model_id,
                layer=card.meta.layer,
                hidden_dim=card.meta.hidden_dim,
                alpha=alpha,
                description=card.meta.description,
                p_value=card.meta.p_value,
                separability_score=card.meta.separability_score,
                layer_selection=card.meta.layer_selection,
                num_positive=card.meta.num_positive,
                num_negative=card.meta.num_negative,
                bootstrap_variance=card.meta.bootstrap_variance,
                mean_hidden_norm=card.meta.mean_hidden_norm,
            )
            for card, alpha in self._steerer.loaded_cards().values()
        ]

    async def generate_stream(
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
    ) -> AsyncIterator[GenerationChunk]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[GenerationChunk | None] = asyncio.Queue(maxsize=512)

        def _run_generation() -> None:
            try:
                for token, activations, projection in self._steerer.generate_stream(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    greedy=greedy,
                    steer_prompt=steer_prompt,
                    messages=messages,
                    system_prompt=system_prompt,
                    repetition_penalty=repetition_penalty,
                ):
                    chunk = GenerationChunk(
                        token=token,
                        activations=activations,
                        projection=projection,
                    )
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, GenerationChunk(error=str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        self._executor.submit(_run_generation)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    async def extract_card(
        self,
        concept: str,
        description: str,
        positive: list[str],
        negative: list[str],
        default_alpha: float,
        layer_frac: float,
        auto_layer: bool = False,
        use_robust_mean: bool = False,
    ) -> tuple[str, LoadedCardInfo]:
        loop = asyncio.get_running_loop()
        card = await loop.run_in_executor(
            self._executor,
            lambda: self._extract_card_sync(
                concept,
                description,
                positive,
                negative,
                default_alpha,
                layer_frac,
                auto_layer,
                use_robust_mean,
            ),
        )
        save_path = self._cards_dir / f"{concept}.safetensors"
        await loop.run_in_executor(self._executor, card.save, save_path)
        self._steerer.load_card(card)
        info = LoadedCardInfo(
            concept=card.meta.concept,
            model_id=card.meta.model_id,
            layer=card.meta.layer,
            hidden_dim=card.meta.hidden_dim,
            alpha=card.meta.default_alpha,
            description=card.meta.description,
            p_value=card.meta.p_value,
            separability_score=card.meta.separability_score,
            layer_selection=card.meta.layer_selection,
            num_positive=card.meta.num_positive,
            num_negative=card.meta.num_negative,
            bootstrap_variance=card.meta.bootstrap_variance,
            mean_hidden_norm=card.meta.mean_hidden_norm,
        )
        return str(save_path), info

    def _extract_card_sync(
        self,
        concept: str,
        description: str,
        positive: list[str],
        negative: list[str],
        default_alpha: float,
        layer_frac: float,
        auto_layer: bool,
        use_robust_mean: bool,
    ) -> ConceptCard:
        model, tokenizer = self._steerer._ensure_loaded()
        dataset = ContrastiveDataset(positive=positive, negative=negative)
        extractor = Extractor(
            model_id=self._steerer.model_id,
            layer_frac=layer_frac,
            model=model,
            tokenizer=tokenizer,
            auto_layer=auto_layer,
            use_robust_mean=use_robust_mean,
        )
        return extractor.extract(
            dataset=dataset,
            concept=concept,
            description=description,
            default_alpha=default_alpha,
        )

    async def get_geometry(self) -> dict[str, object] | None:
        import torch

        space = self._steerer.concept_space()
        if space is None:
            return None
        proj = space.projection
        cards = self._steerer.loaded_cards()
        confidence_ellipses: dict[str, list[list[float]]] | None = None
        ellipse_data: dict[str, list[list[float]]] = {}
        for concept, (card, _) in cards.items():
            cov = card.meta.bootstrap_cov_2d
            axes = card.meta.bootstrap_pca_axes
            if cov is not None and axes is not None:
                local_cov = torch.tensor(cov, dtype=torch.float32)
                local_axes = torch.tensor(axes, dtype=torch.float32)
                transformed = space.transform_bootstrap_cov(local_cov, local_axes)
                ellipse_data[concept] = transformed.tolist()
            elif cov is not None:
                ellipse_data[concept] = cov
        if ellipse_data:
            confidence_ellipses = ellipse_data
        return {
            "concepts": proj.concepts,
            "vectors_2d": [(float(v[0].item()), float(v[1].item())) for v in proj.vectors_2d],
            "gram": proj.gram.tolist(),
            "variance_ratio": (
                float(proj.variance_ratio[0].item()),
                float(proj.variance_ratio[1].item()),
            ),
            "projection_jacobian": proj.jacobian.tolist(),
            "confidence_ellipses": confidence_ellipses,
            "orthogonalized": True,
            "projection_coverage": proj.projection_coverage,
        }

    async def inverse_project(
        self,
        delta: tuple[float, float],
        mode: Literal["pca", "concept_plane"],
        anchor_concepts: tuple[str, str] | None,
        alpha_bounds: dict[str, tuple[float, float]] | None = None,
        max_step: float | None = None,
    ) -> dict[str, float]:
        space = self._steerer.concept_space()
        if space is None:
            return {}
        if mode == "concept_plane" and anchor_concepts is not None:
            c1, c2 = anchor_concepts
            concepts = space.concepts
            result = {c: 0.0 for c in concepts}
            if c1 in result:
                result[c1] = delta[0]
            if c2 in result:
                result[c2] = delta[1]
            return result
        return space.inverse_project(
            delta, current_alphas=space.alphas, alpha_bounds=alpha_bounds, max_step=max_step
        )

    async def layer_sweep(
        self,
        positive: list[str],
        negative: list[str],
    ) -> tuple[list[tuple[int, float]], int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._layer_sweep_sync(positive, negative),
        )

    def _layer_sweep_sync(
        self, positive: list[str], negative: list[str]
    ) -> tuple[list[tuple[int, float]], int]:
        model, tokenizer = self._steerer._ensure_loaded()
        dataset = ContrastiveDataset(positive=positive, negative=negative)
        extractor = Extractor(
            model_id=self._steerer.model_id,
            model=model,
            tokenizer=tokenizer,
            auto_layer=True,
        )
        scores = extractor.layer_sweep(dataset)
        best = max(scores, key=lambda x: x[1])[0]
        return scores, best

    async def get_concept_map(self) -> dict[str, object] | None:
        import torch

        from autosae import ConceptCard
        from autosae.concept_card import registry_slug

        loaded_cards = self._steerer.loaded_cards()
        model_slug = registry_slug(self._steerer.model_id)
        loaded_set = set(loaded_cards.keys())
        all_concepts: list[str] = []
        all_vectors: list[torch.Tensor] = []
        all_metas: list[dict[str, object]] = []

        for concept, (card, _) in loaded_cards.items():
            all_concepts.append(concept)
            all_vectors.append(card.vector)
            all_metas.append(
                {
                    "loaded": True,
                    "separability_score": card.meta.separability_score,
                    "bootstrap_confidence": card.meta.bootstrap_variance,
                }
            )

        registry_concepts = [
            "formality",
            "safety",
            "reasoning",
            "creativity",
            "conciseness",
            "coding",
            "empathy",
            "certainty",
        ]
        for concept in registry_concepts:
            if concept in loaded_set:
                continue
            try:
                reg_card = ConceptCard.from_registry(concept, model=model_slug)
                all_concepts.append(concept)
                all_vectors.append(reg_card.vector)
                all_metas.append(
                    {
                        "loaded": False,
                        "separability_score": reg_card.meta.separability_score,
                        "bootstrap_confidence": reg_card.meta.bootstrap_variance,
                    }
                )
            except Exception:
                pass

        if not all_vectors:
            return None

        V = torch.stack(all_vectors)
        V_norm = V / (V.norm(dim=-1, keepdim=True) + 1e-9)

        if V_norm.shape[0] == 1:
            coords: list[list[float]] = [[1.0, 0.0]]
        else:
            _, _, Vt = torch.linalg.svd(V_norm, full_matrices=False)
            proj = torch.stack([V_norm @ Vt[0], V_norm @ Vt[1]], dim=-1)
            coords = proj.tolist()

        points = []
        for i, concept in enumerate(all_concepts):
            meta = all_metas[i]
            coord = coords[i]
            points.append(
                {
                    "concept": concept,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "loaded": bool(meta["loaded"]),
                    "separability_score": meta.get("separability_score"),
                    "bootstrap_confidence": meta.get("bootstrap_confidence"),
                    "model_id": self._steerer.model_id,
                }
            )

        return {"points": points, "model_id": self._steerer.model_id}

    async def get_card_path(self, concept: str) -> str | None:
        path = self._cards_dir / f"{concept}.safetensors"
        if path.exists():
            return str(path)
        cards = self._steerer.loaded_cards()
        if concept not in cards:
            return None
        card, _ = cards[concept]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, card.save, path)
        return str(path)

    async def shutdown(self) -> None:
        self._steerer.unload()
        self._executor.shutdown(wait=False)
