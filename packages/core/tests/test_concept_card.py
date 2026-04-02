import tempfile
from pathlib import Path

import pytest
import torch

from autosae.concept_card import ConceptCard, ConceptCardMeta


def test_concept_card_init(concept_card: ConceptCard) -> None:
    assert concept_card.meta.concept == "formality"
    assert concept_card.vector.shape == (64,)
    assert concept_card.vector.dtype == torch.float32


def test_concept_card_wrong_shape() -> None:
    meta = ConceptCardMeta(model_id="test", layer=0, hidden_dim=64, concept="test")
    with pytest.raises(ValueError, match="Expected vector of shape"):
        ConceptCard(meta=meta, vector=torch.randn(128))


def test_concept_card_save_load_roundtrip(concept_card: ConceptCard) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "formality.safetensors"
        concept_card.save(path)
        loaded = ConceptCard.load(path)

    assert loaded.meta.concept == concept_card.meta.concept
    assert loaded.meta.model_id == concept_card.meta.model_id
    assert loaded.meta.layer == concept_card.meta.layer
    assert loaded.meta.hidden_dim == concept_card.meta.hidden_dim
    assert loaded.meta.default_alpha == concept_card.meta.default_alpha
    assert torch.allclose(loaded.vector, concept_card.vector)


def test_concept_card_load_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        ConceptCard.load("/nonexistent/path/card.safetensors")


def test_concept_card_repr(concept_card: ConceptCard) -> None:
    r = repr(concept_card)
    assert "formality" in r
    assert "test-model" in r
    assert "16" in r


def test_meta_v2_optional_fields_roundtrip() -> None:
    import tempfile
    from pathlib import Path

    meta = ConceptCardMeta(
        model_id="test-model",
        layer=16,
        hidden_dim=64,
        concept="formality",
        p_value=0.01,
        separability_score=5.3,
        num_positive=10,
        num_negative=10,
        mean_positive_norm=1.2,
        mean_negative_norm=0.9,
        cosine_distance=0.4,
        bootstrap_variance=0.003,
        bootstrap_cov_2d=[[0.01, 0.0], [0.0, 0.005]],
        layer_selection="manual",
    )
    vector = torch.randn(64)
    vector = vector / vector.norm()
    card = ConceptCard(meta=meta, vector=vector)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.safetensors"
        card.save(path)
        loaded = ConceptCard.load(path)

    assert loaded.meta.p_value == pytest.approx(0.01)
    assert loaded.meta.separability_score == pytest.approx(5.3)
    assert loaded.meta.num_positive == 10
    assert loaded.meta.num_negative == 10
    assert loaded.meta.bootstrap_cov_2d == [[0.01, 0.0], [0.0, 0.005]]
    assert loaded.meta.layer_selection == "manual"


def test_meta_v2_backward_compat() -> None:
    import tempfile
    from pathlib import Path

    meta = ConceptCardMeta(
        model_id="test-model",
        layer=8,
        hidden_dim=64,
        concept="creativity",
    )
    vector = torch.randn(64)
    vector = vector / vector.norm()
    card = ConceptCard(meta=meta, vector=vector)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old.safetensors"
        card.save(path)
        loaded = ConceptCard.load(path)

    assert loaded.meta.p_value is None
    assert loaded.meta.separability_score is None
    assert loaded.meta.bootstrap_cov_2d is None
    assert loaded.meta.layer_selection is None
