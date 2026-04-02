from __future__ import annotations

import logging
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from autosae._hooks import get_model_layers, last_token_pool, mean_pool
from autosae.concept_card import ConceptCard, ConceptCardMeta
from autosae.dataset import ContrastiveDataset

logger = logging.getLogger(__name__)

MIN_RECOMMENDED_PAIRS = 10


class Extractor:
    def __init__(
        self,
        model_id: str,
        layer_frac: float = 0.6,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        auto_layer: bool = True,
        use_robust_mean: bool = False,
        pool_mode: Literal["mean", "last_token"] = "last_token",
    ) -> None:
        if not auto_layer and not 0.0 < layer_frac < 1.0:
            raise ValueError(f"layer_frac must be in (0, 1), got {layer_frac}")
        self.model_id = model_id
        self.layer_frac = layer_frac
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.auto_layer = auto_layer
        self.use_robust_mean = use_robust_mean
        self.pool_mode = pool_mode
        self._model: PreTrainedModel | None = model
        self._tokenizer: PreTrainedTokenizerBase | None = tokenizer
        self._external_model = model is not None

    def _ensure_loaded(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if self._model is None or self._tokenizer is None:
            quant_config = None
            if self.load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            elif self.load_in_8bit:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map=self.device,
                dtype=torch.float16 if not (self.load_in_4bit or self.load_in_8bit) else None,
            )
            self._model.eval()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._model, self._tokenizer

    @staticmethod
    def _permutation_test(
        pos: torch.Tensor, neg: torch.Tensor, n_permutations: int = 1000
    ) -> float:
        observed = (pos.mean(0) - neg.mean(0)).norm()
        combined = torch.cat([pos, neg], dim=0)
        n_pos = pos.shape[0]
        count = 0
        for _ in range(n_permutations):
            perm = combined[torch.randperm(combined.shape[0])]
            perm_pos = perm[:n_pos]
            perm_neg = perm[n_pos:]
            stat = (perm_pos.mean(0) - perm_neg.mean(0)).norm()
            if stat >= observed:
                count += 1
        return (count + 1) / (n_permutations + 1)

    @staticmethod
    def _bootstrap_confidence(
        pos: torch.Tensor, neg: torch.Tensor, n_bootstrap: int = 1000
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        directions: list[torch.Tensor] = []
        for _ in range(n_bootstrap):
            pos_idx = torch.randint(0, pos.shape[0], (pos.shape[0],))
            neg_idx = torch.randint(0, neg.shape[0], (neg.shape[0],))
            v_b = F.normalize(pos[pos_idx].mean(0) - neg[neg_idx].mean(0), dim=-1)
            directions.append(v_b)
        dirs = torch.stack(directions)
        mean_direction = F.normalize(dirs.mean(0), dim=-1)
        centered = dirs - dirs.mean(0)
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        pca_axes = Vh[:2]
        coords = centered @ pca_axes.T
        cov_2d = (coords.T @ coords) / max(n_bootstrap - 1, 1)
        return mean_direction, cov_2d, pca_axes

    @staticmethod
    def _robust_mean(
        hiddens: torch.Tensor,
        delta_factor: float = 1.345,
        max_iter: int = 50,
        tolerance: float = 1e-6,
    ) -> torch.Tensor:
        import warnings

        mu = hiddens.mean(0)
        converged = False
        for _ in range(max_iter):
            residuals = hiddens - mu.unsqueeze(0)
            norms = residuals.norm(dim=-1)
            median_norm = norms.median()
            s = (median_norm / 0.6745).clamp(min=1e-12)
            c = delta_factor * s
            w = torch.where(norms <= c, torch.ones_like(norms), c / norms.clamp(min=1e-12))
            w_sum = w.sum().clamp(min=1e-12)
            mu_new = (hiddens * w.unsqueeze(-1)).sum(0) / w_sum
            if (mu_new - mu).norm() < tolerance * mu.norm().clamp(min=1e-12):
                mu = mu_new
                converged = True
                break
            mu = mu_new
        if not converged:
            warnings.warn(
                f"Robust mean did not converge after {max_iter} iterations. "
                "Consider increasing max_iter or checking your data distribution.",
                UserWarning,
                stacklevel=2,
            )
        return mu

    @staticmethod
    def _fisher_discriminant(pos: torch.Tensor, neg: torch.Tensor) -> float:
        d = F.normalize(pos.mean(0) - neg.mean(0), dim=-1)
        proj_pos = pos @ d
        proj_neg = neg @ d
        num = (proj_pos.mean() - proj_neg.mean()) ** 2
        denom = proj_pos.var() + proj_neg.var()
        return float((num / denom.clamp(min=1e-12)).item())

    def _collect_hiddens_all_layers(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: ContrastiveDataset,
        batch_size: int,
        num_layers: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        layers = get_model_layers(model)
        per_layer_pos: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        per_layer_neg: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        stores: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        handles = []

        for i, layer in enumerate(layers):

            def make_hook(idx: int) -> Any:
                def hook(module: nn.Module, input: tuple[torch.Tensor, ...], output: Any) -> None:
                    hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
                    stores[idx].append(hidden.detach().cpu())

                return hook

            handles.append(layer.register_forward_hook(make_hook(i)))

        try:
            for side, prompts in [("pos", dataset.positive), ("neg", dataset.negative)]:
                for i in range(0, len(prompts), batch_size):
                    batch = prompts[i : i + batch_size]
                    for s in stores:
                        s.clear()
                    encoded = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    input_ids = encoded["input_ids"].to(model.device)
                    attention_mask = encoded["attention_mask"].to(model.device)
                    model(input_ids=input_ids, attention_mask=attention_mask)
                    pool_fn = last_token_pool if self.pool_mode == "last_token" else mean_pool
                    for layer_idx in range(num_layers):
                        if stores[layer_idx]:
                            hidden = stores[layer_idx][0]
                            pooled = pool_fn(hidden, attention_mask.cpu())
                            if side == "pos":
                                per_layer_pos[layer_idx].append(pooled)
                            else:
                                per_layer_neg[layer_idx].append(pooled)
        finally:
            for handle in handles:
                handle.remove()

        pos_tensors = [torch.cat(per_layer_pos[i], dim=0) for i in range(num_layers)]
        neg_tensors = [torch.cat(per_layer_neg[i], dim=0) for i in range(num_layers)]
        return pos_tensors, neg_tensors

    def _find_best_layer(self, dataset: ContrastiveDataset, batch_size: int) -> int:
        model, tokenizer = self._ensure_loaded()
        layers = get_model_layers(model)
        num_layers = len(layers)
        pos_hiddens, neg_hiddens = self._collect_hiddens_all_layers(
            model, tokenizer, dataset, batch_size, num_layers
        )
        fisher_scores = [
            self._fisher_discriminant(pos_hiddens[i], neg_hiddens[i]) for i in range(num_layers)
        ]
        p_values = [
            self._permutation_test(pos_hiddens[i], neg_hiddens[i]) for i in range(num_layers)
        ]
        significant = [i for i in range(num_layers) if p_values[i] < 0.05]
        candidates = significant if significant else list(range(num_layers))
        return int(max(candidates, key=lambda i: fisher_scores[i]))

    def layer_sweep(
        self, dataset: ContrastiveDataset, batch_size: int = 8
    ) -> list[tuple[int, float]]:
        model, tokenizer = self._ensure_loaded()
        layers = get_model_layers(model)
        num_layers = len(layers)
        pos_hiddens, neg_hiddens = self._collect_hiddens_all_layers(
            model, tokenizer, dataset, batch_size, num_layers
        )
        return [
            (i, self._fisher_discriminant(pos_hiddens[i], neg_hiddens[i]))
            for i in range(num_layers)
        ]

    @torch.no_grad()
    def extract(
        self,
        dataset: ContrastiveDataset,
        concept: str,
        batch_size: int = 8,
        description: str = "",
        default_alpha: float = 1.0,
    ) -> ConceptCard:
        if len(dataset) < MIN_RECOMMENDED_PAIRS:
            logger.warning(
                f"Dataset has only {len(dataset)} pairs. "
                f"Recommend at least {MIN_RECOMMENDED_PAIRS} for reliable concept extraction."
            )

        model, tokenizer = self._ensure_loaded()
        layers = get_model_layers(model)

        layer_selection: Literal["auto", "manual"]
        if self.auto_layer:
            target_idx = self._find_best_layer(dataset, batch_size)
            layer_selection = "auto"
        else:
            target_idx = min(int(len(layers) * self.layer_frac), len(layers) - 1)
            layer_selection = "manual"

        capture_store: list[torch.Tensor] = []

        def _capture_hook(module: nn.Module, input: tuple[torch.Tensor, ...], output: Any) -> None:
            hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
            capture_store.append(hidden.detach().cpu())

        handle = layers[target_idx].register_forward_hook(_capture_hook)

        try:
            pos_hiddens = self._collect_hiddens(
                model, tokenizer, dataset.positive, batch_size, capture_store
            )
            neg_hiddens = self._collect_hiddens(
                model, tokenizer, dataset.negative, batch_size, capture_store
            )
        finally:
            handle.remove()

        if self.use_robust_mean:
            pos_mean = self._robust_mean(pos_hiddens)
            neg_mean = self._robust_mean(neg_hiddens)
        else:
            pos_mean = pos_hiddens.mean(0)
            neg_mean = neg_hiddens.mean(0)

        v = F.normalize(pos_mean - neg_mean, dim=-1)

        mean_hidden_norm = float(
            ((pos_hiddens.norm(dim=-1).mean() + neg_hiddens.norm(dim=-1).mean()) / 2).item()
        )

        p_value = self._permutation_test(pos_hiddens, neg_hiddens)
        separability = self._fisher_discriminant(pos_hiddens, neg_hiddens)
        _, cov_2d, pca_axes = self._bootstrap_confidence(pos_hiddens, neg_hiddens)
        cosine_dist = float(
            1.0
            - torch.nn.functional.cosine_similarity(
                pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)
            ).item()
        )
        bootstrap_var = float(cov_2d.diagonal().sum().item())

        return ConceptCard(
            meta=ConceptCardMeta(
                model_id=self.model_id,
                layer=target_idx,
                hidden_dim=int(v.shape[0]),
                default_alpha=default_alpha,
                concept=concept,
                description=description,
                p_value=p_value,
                separability_score=separability,
                num_positive=len(dataset.positive),
                num_negative=len(dataset.negative),
                mean_positive_norm=float(pos_mean.norm().item()),
                mean_negative_norm=float(neg_mean.norm().item()),
                cosine_distance=cosine_dist,
                bootstrap_variance=bootstrap_var,
                bootstrap_cov_2d=cov_2d.tolist(),
                bootstrap_pca_axes=pca_axes.tolist(),
                layer_selection=layer_selection,
                mean_hidden_norm=mean_hidden_norm,
            ),
            vector=v,
        )

    def _collect_hiddens(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompts: list[str],
        batch_size: int,
        store: list[torch.Tensor],
    ) -> torch.Tensor:
        pooled_list: list[torch.Tensor] = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            store.clear()

            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            model(input_ids=input_ids, attention_mask=attention_mask)

            if not store:
                raise RuntimeError(
                    "Forward hook did not capture any activations. "
                    "This may indicate an unsupported model architecture."
                )
            hidden = store[0]
            pool_fn = last_token_pool if self.pool_mode == "last_token" else mean_pool
            pooled = pool_fn(hidden, attention_mask.cpu())
            pooled_list.append(pooled)

        return torch.cat(pooled_list, dim=0)

    def unload(self) -> None:
        if not self._external_model:
            self._model = None
            self._tokenizer = None
            torch.cuda.empty_cache()
