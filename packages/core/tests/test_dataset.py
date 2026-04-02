import json
import tempfile
from pathlib import Path

import pytest

from autosae.dataset import ContrastiveDataset


def test_dataset_init(contrastive_dataset: ContrastiveDataset) -> None:
    assert len(contrastive_dataset) == 3
    assert len(contrastive_dataset.positive) == 3
    assert len(contrastive_dataset.negative) == 3


def test_dataset_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ContrastiveDataset(positive=["a", "b"], negative=["c"])


def test_dataset_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ContrastiveDataset(positive=[], negative=[])


def test_dataset_from_dict() -> None:
    data = {"positive": ["formal text"], "negative": ["casual text"]}
    dataset = ContrastiveDataset.from_dict(data)
    assert len(dataset) == 1


def test_dataset_from_dict_missing_key() -> None:
    with pytest.raises(ValueError, match="positive"):
        ContrastiveDataset.from_dict({"positive": ["a"]})


def test_dataset_rejects_blank_strings() -> None:
    with pytest.raises(ValueError, match="blank"):
        ContrastiveDataset(positive=["", "formal text"], negative=["casual", "text"])

    with pytest.raises(ValueError, match="blank"):
        ContrastiveDataset(positive=["formal"], negative=["   "])


def test_dataset_from_json() -> None:
    data = {"positive": ["formal"], "negative": ["casual"]}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)
    try:
        dataset = ContrastiveDataset.from_json(path)
        assert len(dataset) == 1
    finally:
        path.unlink()
