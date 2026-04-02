import pytest
import torch

from autosae._hooks import get_model_layers, last_token_pool, mean_pool
from autosae.exceptions import UnsupportedArchitectureError


class _UnknownModel:
    class config:
        architectures = ["UnknownForCausalLM"]


def test_get_model_layers_unsupported_architecture() -> None:
    with pytest.raises(UnsupportedArchitectureError, match="UnknownForCausalLM"):
        get_model_layers(_UnknownModel())  # type: ignore[arg-type]


def test_mean_pool_masks_padding() -> None:
    hidden = torch.ones(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)
    result = mean_pool(hidden, mask)
    assert result.shape == (2, 8)
    assert torch.allclose(result, torch.ones(2, 8))


def test_last_token_pool() -> None:
    hidden = torch.randn(2, 5, 64)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.long)
    result = last_token_pool(hidden, mask)
    assert result.shape == (2, 64)
    assert torch.allclose(result[0], hidden[0, 2, :])
    assert torch.allclose(result[1], hidden[1, 3, :])
