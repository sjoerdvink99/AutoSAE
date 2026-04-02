from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F

from autosae.concept_card import ConceptCard


@dataclass(frozen=True)
class ConceptProjection:
    concepts: list[str]
    vectors_2d: torch.Tensor
    axes: torch.Tensor
    gram: torch.Tensor
    variance_ratio: torch.Tensor
    jacobian: torch.Tensor
    projection_coverage: float


class ConceptSpace:
    def __init__(
        self,
        cards: dict[str, tuple[ConceptCard, float]],
        orthogonalize: bool = True,
    ) -> None:
        if len(cards) < 1:
            raise ValueError("ConceptSpace requires at least one loaded card.")

        self._concepts = list(cards.keys())
        self._alphas = {c: alpha for c, (_, alpha) in cards.items()}

        raw = torch.stack([card.vector for card, _ in cards.values()])
        self._V_normalized = F.normalize(raw, dim=-1)
        self._gram = self._V_normalized @ self._V_normalized.T

        V_for_svd = self._modified_gram_schmidt(self._V_normalized) if orthogonalize else self._V_normalized

        if len(cards) >= 2:
            U, S, Vh = torch.linalg.svd(V_for_svd, full_matrices=False)
            rank = int((S > 1e-6).sum().item())
            if rank < 2:
                warnings.warn(
                    f"ConceptSpace SVD rank={rank} < 2: concept vectors are nearly parallel. "
                    "2D projections may be degenerate.",
                    UserWarning,
                    stacklevel=2,
                )
            self._axes = Vh[:2]
            self._vectors_2d = self._V_normalized @ self._axes.T
            var = S**2
            self._variance_ratio = var[:2] / var.sum().clamp(min=1e-12)
        else:
            torch.manual_seed(42)
            v = self._V_normalized[0]
            rand = torch.randn_like(v)
            orth = rand - torch.dot(rand, v) * v
            orth = F.normalize(orth, dim=-1)
            self._axes = torch.stack([v, orth])
            self._vectors_2d = torch.zeros(1, 2)
            self._vectors_2d[0, 0] = 1.0
            self._variance_ratio = torch.tensor([1.0, 0.0])

        self._J = self._axes @ self._V_normalized.T
        self._projection_coverage = float(self._variance_ratio.sum().clamp(max=1.0).item())

    @staticmethod
    def _modified_gram_schmidt(V: torch.Tensor) -> torch.Tensor:
        Q = V.clone()
        for i in range(Q.shape[0]):
            for j in range(i):
                Q[i] = Q[i] - torch.dot(Q[i], Q[j]) * Q[j]
            norm = Q[i].norm()
            if norm > 1e-10:
                Q[i] = Q[i] / norm
        return Q

    @property
    def projection(self) -> ConceptProjection:
        return ConceptProjection(
            concepts=self._concepts,
            vectors_2d=self._vectors_2d,
            axes=self._axes,
            gram=self._gram,
            variance_ratio=self._variance_ratio,
            jacobian=self._J,
            projection_coverage=self._projection_coverage,
        )

    def project(self, h: torch.Tensor) -> tuple[float, float]:
        h_f = h.float().to(self._axes.device)
        coords = h_f @ self._axes.T
        x = float(coords[0].item())
        y = float(coords[1].item()) if coords.shape[0] > 1 else 0.0
        return x, y

    def concept_plane_project(self, h: torch.Tensor, c1: str, c2: str) -> tuple[float, float]:
        i1, i2 = self._concepts.index(c1), self._concepts.index(c2)
        h_f = h.float().to(self._V_normalized.device)
        x = torch.dot(h_f, self._V_normalized[i1]).item()
        y = torch.dot(h_f, self._V_normalized[i2]).item()
        return float(x), float(y)

    def inverse_project(
        self,
        delta: tuple[float, float],
        current_alphas: dict[str, float] | None = None,
        alpha_bounds: dict[str, tuple[float, float]] | None = None,
        max_step: float | None = None,
    ) -> dict[str, float]:
        dp = torch.tensor([delta[0], delta[1]], dtype=torch.float32, device=self._J.device)

        effective_bounds: dict[str, tuple[float, float]] | None = None
        if alpha_bounds is not None:
            effective_bounds = {}
            for concept, (lo, hi) in alpha_bounds.items():
                if current_alphas is not None and concept in current_alphas:
                    cur = current_alphas[concept]
                    effective_bounds[concept] = (lo - cur, hi - cur)
                else:
                    effective_bounds[concept] = (lo, hi)

        solution, *_ = torch.linalg.lstsq(self._J, dp.unsqueeze(-1))
        d_alpha = solution.squeeze(-1).clone()

        if effective_bounds is not None:
            for i, concept in enumerate(self._concepts):
                if concept in effective_bounds:
                    lo, hi = effective_bounds[concept]
                    d_alpha[i] = d_alpha[i].clamp(lo, hi)

        if max_step is not None:
            inf_norm = d_alpha.abs().max()
            if inf_norm > max_step:
                d_alpha = d_alpha * (max_step / inf_norm)

        return {c: float(d_alpha[i].item()) for i, c in enumerate(self._concepts)}

    def transform_bootstrap_cov(
        self, local_cov: torch.Tensor, local_axes: torch.Tensor
    ) -> torch.Tensor:
        R = self._axes @ local_axes.T
        return cast(torch.Tensor, R @ local_cov @ R.T)

    @property
    def concepts(self) -> list[str]:
        return self._concepts

    @property
    def alphas(self) -> dict[str, float]:
        return dict(self._alphas)
