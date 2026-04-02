from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from autosae.concept_card import ConceptCard, ConceptCardMeta
from autosae.geometry import ConceptSpace


def _make_card(concept: str, vector: torch.Tensor, hidden_dim: int) -> ConceptCard:
    meta = ConceptCardMeta(
        model_id="test-model",
        layer=16,
        hidden_dim=hidden_dim,
        concept=concept,
    )
    v = vector / vector.norm()
    return ConceptCard(meta=meta, vector=v)


@pytest.fixture
def hidden_dim() -> int:
    return 64


@pytest.fixture
def two_cards(hidden_dim: int) -> dict[str, tuple[ConceptCard, float]]:
    torch.manual_seed(0)
    v1 = torch.randn(hidden_dim)
    v2 = torch.randn(hidden_dim)
    c1 = _make_card("formality", v1, hidden_dim)
    c2 = _make_card("creativity", v2, hidden_dim)
    return {"formality": (c1, 1.0), "creativity": (c2, 2.0)}


@pytest.fixture
def three_cards(hidden_dim: int) -> dict[str, tuple[ConceptCard, float]]:
    torch.manual_seed(1)
    cards = {}
    for name in ("formality", "creativity", "safety"):
        v = torch.randn(hidden_dim)
        cards[name] = (_make_card(name, v, hidden_dim), 1.0)
    return cards


def test_gram_diagonal_is_one(two_cards: dict) -> None:
    space = ConceptSpace(two_cards)
    gram = space.projection.gram
    assert gram.shape == (2, 2)
    assert torch.allclose(gram.diagonal(), torch.ones(2), atol=1e-5)


def test_gram_is_symmetric(three_cards: dict) -> None:
    space = ConceptSpace(three_cards)
    gram = space.projection.gram
    assert torch.allclose(gram, gram.T, atol=1e-5)


def test_gram_values_in_range(three_cards: dict) -> None:
    space = ConceptSpace(three_cards)
    gram = space.projection.gram
    assert gram.min().item() >= -1.0 - 1e-5
    assert gram.max().item() <= 1.0 + 1e-5


def test_variance_ratio_sums_at_most_one(three_cards: dict) -> None:
    space = ConceptSpace(three_cards)
    vr = space.projection.variance_ratio
    assert vr.sum().item() <= 1.0 + 1e-5
    assert vr[0].item() >= vr[1].item() - 1e-5


def test_vectors_2d_shape(two_cards: dict) -> None:
    space = ConceptSpace(two_cards)
    vecs = space.projection.vectors_2d
    assert vecs.shape == (2, 2)


def test_project_returns_two_floats(two_cards: dict, hidden_dim: int) -> None:
    space = ConceptSpace(two_cards)
    h = torch.randn(hidden_dim)
    x, y = space.project(h)
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_inverse_project_round_trip(two_cards: dict, hidden_dim: int) -> None:
    space = ConceptSpace(two_cards, orthogonalize=False)
    torch.manual_seed(42)
    h = torch.randn(hidden_dim)
    x0, y0 = space.project(h)

    delta_alpha = space.inverse_project((0.5, 0.3))
    d_alpha = torch.tensor([delta_alpha[c] for c in space.concepts])
    concept_vecs = torch.stack([two_cards[c][0].vector for c in space.concepts])
    h_new = h + (concept_vecs.T @ d_alpha)

    x1, y1 = space.project(h_new)
    assert abs((x1 - x0) - 0.5) < 1e-4
    assert abs((y1 - y0) - 0.3) < 1e-4


def test_concept_plane_project(two_cards: dict, hidden_dim: int) -> None:
    space = ConceptSpace(two_cards)
    h = torch.randn(hidden_dim)
    x, y = space.concept_plane_project(h, "formality", "creativity")
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_single_card_space(hidden_dim: int) -> None:
    torch.manual_seed(5)
    v = torch.randn(hidden_dim)
    card = _make_card("formality", v, hidden_dim)
    space = ConceptSpace({"formality": (card, 1.0)})
    h = torch.randn(hidden_dim)
    x, y = space.project(h)
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert space._axes.shape == (2, hidden_dim)
    assert abs(torch.dot(space._axes[0], space._axes[1]).item()) < 1e-5


def test_concepts_list(two_cards: dict) -> None:
    space = ConceptSpace(two_cards)
    assert set(space.concepts) == {"formality", "creativity"}


def test_alphas(two_cards: dict) -> None:
    space = ConceptSpace(two_cards)
    assert space.alphas["formality"] == 1.0
    assert space.alphas["creativity"] == 2.0


def test_gram_reflects_actual_concept_correlations(two_cards: dict) -> None:
    space = ConceptSpace(two_cards, orthogonalize=True)
    gram = space.projection.gram
    c1 = two_cards["formality"][0].vector
    c2 = two_cards["creativity"][0].vector
    expected_cosine = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0)).item()
    assert abs(gram[0, 1].item() - expected_cosine) < 1e-5
    assert abs(gram[1, 0].item() - expected_cosine) < 1e-5


def test_orthogonalize_preserves_original_cards(two_cards: dict, hidden_dim: int) -> None:
    original_vec = two_cards["formality"][0].vector.clone()
    ConceptSpace(two_cards, orthogonalize=True)
    assert torch.allclose(two_cards["formality"][0].vector, original_vec)


def test_svd_degeneracy_warning(hidden_dim: int) -> None:
    torch.manual_seed(10)
    v = torch.randn(hidden_dim)
    v = v / v.norm()
    c1 = _make_card("formality", v, hidden_dim)
    c2 = _make_card("creativity", v.clone(), hidden_dim)
    cards: dict[str, tuple[ConceptCard, float]] = {
        "formality": (c1, 1.0),
        "creativity": (c2, 1.0),
    }
    with pytest.warns(UserWarning, match="rank"):
        ConceptSpace(cards, orthogonalize=False)


def test_inverse_project_with_bounds(two_cards: dict) -> None:
    space = ConceptSpace(two_cards)
    bounds = {c: (-0.5, 0.5) for c in space.concepts}
    result = space.inverse_project((0.3, 0.1), alpha_bounds=bounds)
    for c in space.concepts:
        assert -0.5 - 1e-4 <= result[c] <= 0.5 + 1e-4


def test_inverse_project_with_max_step(two_cards: dict) -> None:
    space = ConceptSpace(two_cards)
    max_step = 0.3
    result = space.inverse_project((1.0, 1.0), max_step=max_step)
    inf_norm = max(abs(v) for v in result.values())
    assert inf_norm <= max_step + 1e-4


def test_inverse_project_unconstrained_matches_original(two_cards: dict, hidden_dim: int) -> None:
    space = ConceptSpace(two_cards)
    torch.manual_seed(42)
    delta = (0.3, 0.2)
    result_orig = space.inverse_project(delta)
    result_unconstrained = space.inverse_project(delta, alpha_bounds=None, max_step=None)
    for c in space.concepts:
        assert abs(result_orig[c] - result_unconstrained[c]) < 1e-4


def test_inverse_project_k3_round_trip(three_cards: dict, hidden_dim: int) -> None:
    space = ConceptSpace(three_cards, orthogonalize=False)
    torch.manual_seed(42)
    h = torch.randn(hidden_dim)
    x0, y0 = space.project(h)

    delta_alpha = space.inverse_project((0.5, 0.3))
    d_alpha = torch.tensor([delta_alpha[c] for c in space.concepts])
    concept_vecs = torch.stack([three_cards[c][0].vector for c in space.concepts])
    h_new = h + (concept_vecs.T @ d_alpha)

    x1, y1 = space.project(h_new)
    assert abs((x1 - x0) - 0.5) < 1e-4
    assert abs((y1 - y0) - 0.3) < 1e-4


def test_inverse_project_convergence(two_cards: dict) -> None:
    space = ConceptSpace(two_cards)
    bounds = {c: (-2.0, 2.0) for c in space.concepts}
    delta_alpha = space.inverse_project((0.1, 0.05), alpha_bounds=bounds)
    d_alpha = torch.tensor([delta_alpha[c] for c in space.concepts])
    dp = torch.tensor([0.1, 0.05])
    residual = space._J @ d_alpha - dp
    assert residual.norm().item() < 1e-3


def test_inverse_project_absolute_bounds(two_cards: dict) -> None:
    space = ConceptSpace(two_cards, orthogonalize=False)
    current_alphas = {c: 1.0 for c in space.concepts}
    alpha_bounds = {c: (0.5, 1.5) for c in space.concepts}
    result = space.inverse_project(
        (0.5, 0.3), current_alphas=current_alphas, alpha_bounds=alpha_bounds
    )
    for c in space.concepts:
        assert -0.5 - 1e-4 <= result[c] <= 0.5 + 1e-4


def test_jacobian_correct_round_trip_with_orthogonalization(two_cards: dict, hidden_dim: int) -> None:
    space = ConceptSpace(two_cards, orthogonalize=True)
    torch.manual_seed(42)
    h = torch.randn(hidden_dim)
    x0, y0 = space.project(h)

    delta_alpha = space.inverse_project((0.5, 0.3))
    d_alpha = torch.tensor([delta_alpha[c] for c in space.concepts])
    concept_vecs = torch.stack([two_cards[c][0].vector for c in space.concepts])
    h_new = h + (concept_vecs.T @ d_alpha)

    x1, y1 = space.project(h_new)
    assert abs((x1 - x0) - 0.5) < 1e-4
    assert abs((y1 - y0) - 0.3) < 1e-4


def test_projection_coverage_in_range(three_cards: dict) -> None:
    space = ConceptSpace(three_cards)
    coverage = space.projection.projection_coverage
    assert 0.0 <= coverage <= 1.0 + 1e-5


def test_projection_coverage_k3_less_than_one(three_cards: dict) -> None:
    space = ConceptSpace(three_cards)
    assert space.projection.projection_coverage < 1.0 - 1e-5


def test_transform_bootstrap_cov(two_cards: dict, hidden_dim: int) -> None:
    space = ConceptSpace(two_cards)
    torch.manual_seed(42)
    local_cov = torch.eye(2) * 0.01
    local_axes = torch.randn(2, hidden_dim)
    local_axes = local_axes / local_axes.norm(dim=-1, keepdim=True)

    transformed = space.transform_bootstrap_cov(local_cov, local_axes)

    assert transformed.shape == (2, 2)
    eigenvalues = torch.linalg.eigvalsh(transformed)
    assert eigenvalues.min().item() >= -1e-5

    global_axes = space._axes
    same_axes_transformed = space.transform_bootstrap_cov(local_cov, global_axes)
    assert not torch.allclose(transformed, same_axes_transformed, atol=1e-3) or torch.allclose(
        local_axes, global_axes, atol=1e-3
    )
