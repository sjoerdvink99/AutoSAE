from unittest.mock import MagicMock, patch

import pytest
import torch

from autosae.concept_card import ConceptCard
from autosae.dataset import ContrastiveDataset
from autosae.extractor import Extractor

HIDDEN_DIM = 64
NUM_LAYERS = 32


def _make_mock_model(hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS) -> MagicMock:
    model = MagicMock()
    model.device = torch.device("cpu")
    model.config.architectures = ["LlamaForCausalLM"]

    layers = [MagicMock() for _ in range(num_layers)]
    model.model.layers = layers

    def fake_forward(**kwargs):
        batch = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        for layer in layers:
            if layer.register_forward_hook.called:
                hook_fn = layer.register_forward_hook.call_args[0][0]
                fake_hidden = torch.randn(batch, seq_len, hidden_dim)
                hook_fn(layer, (fake_hidden,), (fake_hidden,))

    model.side_effect = fake_forward
    return model


def _make_mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.pad_token = tokenizer.eos_token

    def fake_call(texts, return_tensors, padding, truncation, max_length):
        batch_size = len(texts)
        return {
            "input_ids": torch.randint(0, 100, (batch_size, 16)),
            "attention_mask": torch.ones(batch_size, 16, dtype=torch.long),
        }

    tokenizer.side_effect = fake_call
    return tokenizer


def test_extractor_init() -> None:
    ext = Extractor("meta-llama/Llama-3.1-8B", layer_frac=0.6)
    assert ext.model_id == "meta-llama/Llama-3.1-8B"
    assert ext.layer_frac == 0.6


def test_extractor_invalid_layer_frac() -> None:
    with pytest.raises(ValueError, match="layer_frac"):
        Extractor("some-model", layer_frac=1.5, auto_layer=False)

    with pytest.raises(ValueError, match="layer_frac"):
        Extractor("some-model", layer_frac=0.0, auto_layer=False)


@patch("autosae.extractor.AutoModelForCausalLM.from_pretrained")
@patch("autosae.extractor.AutoTokenizer.from_pretrained")
def test_extract_returns_concept_card(
    mock_tokenizer_cls: MagicMock,
    mock_model_cls: MagicMock,
    contrastive_dataset: ContrastiveDataset,
) -> None:
    num_layers = 32
    hidden_dim = 64
    hooks_registry: list[tuple[object, object]] = []

    layers = []
    for _ in range(num_layers):
        layer = MagicMock()
        h = MagicMock()

        def make_register(lyr: object, hook_handle: object) -> object:
            def register_forward_hook(fn: object) -> object:
                hooks_registry.append((lyr, fn))
                return hook_handle

            return register_forward_hook

        layer.register_forward_hook.side_effect = make_register(layer, h)
        layers.append(layer)

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.eval.return_value = None
    mock_model.model.layers = layers

    def fake_forward(*args: object, **kwargs: object) -> None:
        input_ids = kwargs.get("input_ids")
        assert isinstance(input_ids, torch.Tensor)
        batch, seq_len = input_ids.shape
        fake_hidden = torch.randn(batch, seq_len, hidden_dim)
        for _, hook_fn in hooks_registry:
            callable(hook_fn) and hook_fn(None, None, (fake_hidden,))  # type: ignore[operator]

    mock_model.side_effect = fake_forward
    mock_model_cls.return_value = mock_model

    def fake_tokenize(texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        batch_size = len(texts)
        return {
            "input_ids": torch.randint(0, 100, (batch_size, 16)),
            "attention_mask": torch.ones(batch_size, 16, dtype=torch.long),
        }

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.side_effect = fake_tokenize
    mock_tokenizer_cls.return_value = tokenizer

    ext = Extractor("meta-llama/Llama-3.1-8B")
    card = ext.extract(contrastive_dataset, concept="formality", description="Formal vs casual")

    assert isinstance(card, ConceptCard)
    assert card.meta.concept == "formality"
    assert card.meta.model_id == "meta-llama/Llama-3.1-8B"
    assert card.meta.hidden_dim == hidden_dim
    assert card.vector.shape == (hidden_dim,)
    assert abs(card.vector.norm().item() - 1.0) < 1e-5


def test_permutation_test_separable() -> None:
    torch.manual_seed(0)
    d = 16
    pos = torch.randn(20, d) + 5.0
    neg = torch.randn(20, d) - 5.0
    p = Extractor._permutation_test(pos, neg, n_permutations=200)
    assert p < 0.05


def test_permutation_test_identical() -> None:
    torch.manual_seed(1)
    d = 16
    data = torch.randn(20, d)
    p = Extractor._permutation_test(data, data.clone(), n_permutations=200)
    assert p > 0.05


def test_bootstrap_confidence_shape() -> None:
    torch.manual_seed(2)
    d = 16
    pos = torch.randn(10, d) + 1.0
    neg = torch.randn(10, d) - 1.0
    mean_dir, cov, pca_axes = Extractor._bootstrap_confidence(pos, neg, n_bootstrap=50)
    assert mean_dir.shape == (d,)
    assert cov.shape == (2, 2)
    assert pca_axes.shape == (2, d)


def test_bootstrap_confidence_positive_semidefinite() -> None:
    torch.manual_seed(3)
    d = 16
    pos = torch.randn(10, d) + 1.0
    neg = torch.randn(10, d) - 1.0
    _, cov, _ = Extractor._bootstrap_confidence(pos, neg, n_bootstrap=50)
    eigenvalues = torch.linalg.eigvalsh(cov)
    assert eigenvalues.min().item() >= -1e-5


def test_robust_mean_downweights_outlier() -> None:
    torch.manual_seed(4)
    d = 8
    clean = torch.randn(19, d)
    outlier = torch.randn(1, d) * 100.0
    data = torch.cat([clean, outlier], dim=0)
    robust = Extractor._robust_mean(data)
    ordinary = data.mean(0)
    clean_mean = clean.mean(0)
    assert (robust - clean_mean).norm() < (ordinary - clean_mean).norm()


def test_robust_mean_clean_matches_ordinary() -> None:
    torch.manual_seed(5)
    d = 8
    data = torch.randn(20, d)
    robust = Extractor._robust_mean(data)
    ordinary = data.mean(0)
    assert (robust - ordinary).norm() < 0.5


def test_fisher_discriminant_separable() -> None:
    torch.manual_seed(6)
    d = 16
    pos = torch.randn(20, d) + 5.0
    neg = torch.randn(20, d) - 5.0
    score = Extractor._fisher_discriminant(pos, neg)
    assert score > 10.0


def test_fisher_discriminant_overlapping() -> None:
    torch.manual_seed(7)
    d = 16
    pos = torch.randn(20, d) + 0.1
    neg = torch.randn(20, d) - 0.1
    score = Extractor._fisher_discriminant(pos, neg)
    assert score < 1.0


def test_extract_last_token_mode() -> None:
    hidden_dim = 32
    num_layers = 8
    hooks_registry: list[tuple[object, object]] = []

    layers = []
    for _ in range(num_layers):
        layer = MagicMock()
        h = MagicMock()

        def make_register(lyr: object, hook_handle: object) -> object:
            def register_forward_hook(fn: object) -> object:
                hooks_registry.append((lyr, fn))
                return hook_handle

            return register_forward_hook

        layer.register_forward_hook.side_effect = make_register(layer, h)
        layers.append(layer)

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.eval.return_value = None
    mock_model.model.layers = layers

    def fake_forward(*args: object, **kwargs: object) -> None:
        input_ids = kwargs.get("input_ids")
        assert isinstance(input_ids, torch.Tensor)
        batch, seq_len = input_ids.shape
        fake_hidden = torch.randn(batch, seq_len, hidden_dim)
        for _, hook_fn in hooks_registry:
            callable(hook_fn) and hook_fn(None, None, (fake_hidden,))  # type: ignore[operator]

    mock_model.side_effect = fake_forward

    def fake_tokenize(texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        batch_size = len(texts)
        return {
            "input_ids": torch.randint(0, 100, (batch_size, 16)),
            "attention_mask": torch.ones(batch_size, 16, dtype=torch.long),
        }

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.side_effect = fake_tokenize

    dataset = ContrastiveDataset(
        positive=["a", "b", "c", "d", "e"],
        negative=["f", "g", "h", "i", "j"],
    )
    ext = Extractor("test-model", model=mock_model, tokenizer=tokenizer, pool_mode="last_token")
    card = ext.extract(dataset, concept="test")

    assert isinstance(card, ConceptCard)
    assert card.vector.shape == (hidden_dim,)
    assert abs(card.vector.norm().item() - 1.0) < 1e-5
    assert card.meta.p_value is not None
    assert card.meta.separability_score is not None


def test_extract_populates_statistics() -> None:
    hidden_dim = 32
    num_layers = 8
    hooks_registry: list[tuple[object, object]] = []

    layers = []
    for _ in range(num_layers):
        layer = MagicMock()
        h = MagicMock()

        def make_register(lyr: object, hook_handle: object) -> object:
            def register_forward_hook(fn: object) -> object:
                hooks_registry.append((lyr, fn))
                return hook_handle

            return register_forward_hook

        layer.register_forward_hook.side_effect = make_register(layer, h)
        layers.append(layer)

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.eval.return_value = None
    mock_model.model.layers = layers

    def fake_forward(*args: object, **kwargs: object) -> None:
        input_ids = kwargs.get("input_ids")
        assert isinstance(input_ids, torch.Tensor)
        batch, seq_len = input_ids.shape
        fake_hidden = torch.randn(batch, seq_len, hidden_dim)
        for _, hook_fn in hooks_registry:
            callable(hook_fn) and hook_fn(None, None, (fake_hidden,))  # type: ignore[operator]

    mock_model.side_effect = fake_forward

    def fake_tokenize(texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        batch_size = len(texts)
        return {
            "input_ids": torch.randint(0, 100, (batch_size, 16)),
            "attention_mask": torch.ones(batch_size, 16, dtype=torch.long),
        }

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.side_effect = fake_tokenize

    dataset = ContrastiveDataset(
        positive=["a", "b", "c", "d", "e"],
        negative=["f", "g", "h", "i", "j"],
    )
    ext = Extractor("test-model", model=mock_model, tokenizer=tokenizer)
    card = ext.extract(dataset, concept="test")

    assert card.meta.p_value is not None
    assert 0.0 <= card.meta.p_value <= 1.0
    assert card.meta.separability_score is not None
    assert card.meta.separability_score >= 0.0
    assert card.meta.num_positive == 5
    assert card.meta.num_negative == 5
    assert card.meta.bootstrap_cov_2d is not None
    assert len(card.meta.bootstrap_cov_2d) == 2
    assert card.meta.layer_selection == "auto"
    assert card.meta.mean_hidden_norm is not None
    assert card.meta.mean_hidden_norm > 0
