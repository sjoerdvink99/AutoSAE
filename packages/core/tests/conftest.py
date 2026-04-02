import pytest
import torch

from autosae.concept_card import ConceptCard, ConceptCardMeta
from autosae.dataset import ContrastiveDataset


@pytest.fixture
def hidden_dim() -> int:
    return 64


@pytest.fixture
def concept_card(hidden_dim: int) -> ConceptCard:
    meta = ConceptCardMeta(
        model_id="test-model",
        layer=16,
        hidden_dim=hidden_dim,
        default_alpha=1.5,
        concept="formality",
        description="Formal vs casual register",
    )
    vector = torch.randn(hidden_dim)
    vector = vector / vector.norm()
    return ConceptCard(meta=meta, vector=vector)


@pytest.fixture
def contrastive_dataset() -> ContrastiveDataset:
    return ContrastiveDataset(
        positive=[
            "Please find enclosed the quarterly financial report.",
            "I am writing to formally request your assistance.",
            "The aforementioned terms are hereby agreed upon.",
        ],
        negative=[
            "Hey, check out these numbers lol",
            "can u help me out?",
            "yeah sure sounds good to me",
        ],
    )
