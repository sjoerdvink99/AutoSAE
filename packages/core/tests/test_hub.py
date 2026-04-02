from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from autosae.concept_card import ConceptCard, ConceptCardMeta
from autosae.hub import HubCardResult, download_card, search_hub


@pytest.fixture()
def sample_card() -> ConceptCard:
    meta = ConceptCardMeta(
        model_id="gpt2",
        layer=6,
        hidden_dim=64,
        default_alpha=1.0,
        concept="formality",
        description="Formal vs casual",
    )
    return ConceptCard(meta=meta, vector=torch.randn(64))


def test_search_hub_returns_results() -> None:
    mock_model = MagicMock()
    mock_model.id = "testuser/my-cards"
    mock_model.tags = ["autosae", "activation-steering", "concept-card", "formality"]
    mock_model.downloads = 42

    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.list_models.return_value = [mock_model]
        results = search_hub(query="formality", limit=10)

    assert len(results) == 1
    assert isinstance(results[0], HubCardResult)
    assert results[0].repo_id == "testuser/my-cards"
    assert results[0].downloads == 42


def test_search_hub_empty_query() -> None:
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.list_models.return_value = []
        results = search_hub()

    assert results == []


def test_download_card(tmp_path: Path, sample_card: ConceptCard) -> None:
    save_path = tmp_path / "formality.safetensors"
    sample_card.save(save_path)

    with patch("huggingface_hub.hf_hub_download", return_value=str(save_path)):
        card = download_card(repo_id="testuser/my-cards", concept="formality", model="gpt2")

    assert card.meta.concept == "formality"
    assert card.meta.model_id == "gpt2"


def test_push_to_hub(tmp_path: Path, sample_card: ConceptCard) -> None:
    with patch("huggingface_hub.HfApi") as MockApi:
        mock_instance = MockApi.return_value
        mock_instance.create_repo.return_value = None
        mock_instance.upload_file.return_value = None

        url = sample_card.push_to_hub(repo_id="testuser/my-cards", token="hf_test")

    assert url == "https://huggingface.co/testuser/my-cards"
    assert mock_instance.create_repo.called
    assert mock_instance.upload_file.call_count == 2


def test_push_to_hub_uploads_correct_path(tmp_path: Path, sample_card: ConceptCard) -> None:
    uploaded_paths: list[str] = []

    def capture_upload(**kwargs: object) -> None:
        uploaded_paths.append(str(kwargs.get("path_in_repo", "")))

    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.upload_file.side_effect = capture_upload
        sample_card.push_to_hub(repo_id="testuser/my-cards")

    assert any("formality.safetensors" in p for p in uploaded_paths)
    assert any("README.md" in p for p in uploaded_paths)
