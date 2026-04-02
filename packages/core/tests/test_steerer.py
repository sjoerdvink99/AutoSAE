from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from autosae.concept_card import ConceptCard, ConceptCardMeta, combine_cards
from autosae.exceptions import ConceptNotLoadedError, IncompatibleCardError
from autosae.steerer import Steerer

_VOCAB_SIZE = 100
_EOS_ID = 5


def test_steerer_load_card(concept_card: ConceptCard) -> None:
    steerer = Steerer("meta-llama/Llama-3.1-8B")
    steerer.load_card(concept_card, alpha=2.0)
    assert "formality" in steerer.loaded_concepts()
    assert steerer.loaded_concepts()["formality"] == 2.0


def test_steerer_load_card_default_alpha(concept_card: ConceptCard) -> None:
    steerer = Steerer("meta-llama/Llama-3.1-8B")
    steerer.load_card(concept_card)
    assert steerer.loaded_concepts()["formality"] == concept_card.meta.default_alpha


def test_steerer_set_alpha(concept_card: ConceptCard) -> None:
    steerer = Steerer("meta-llama/Llama-3.1-8B")
    steerer.load_card(concept_card, alpha=1.0)
    steerer.set_alpha("formality", 3.0)
    assert steerer.loaded_concepts()["formality"] == 3.0


def test_steerer_set_alpha_not_loaded() -> None:
    steerer = Steerer("meta-llama/Llama-3.1-8B")
    with pytest.raises(ConceptNotLoadedError):
        steerer.set_alpha("formality", 1.0)


def test_steerer_unload_card(concept_card: ConceptCard) -> None:
    steerer = Steerer("meta-llama/Llama-3.1-8B")
    steerer.load_card(concept_card)
    steerer.unload_card("formality")
    assert "formality" not in steerer.loaded_concepts()


def test_steerer_unload_card_not_loaded() -> None:
    steerer = Steerer("meta-llama/Llama-3.1-8B")
    with pytest.raises(ConceptNotLoadedError):
        steerer.unload_card("nonexistent")


def test_hook_injects_vector(hidden_dim: int, concept_card: ConceptCard) -> None:
    steerer = Steerer("meta-llama/Llama-3.1-8B")
    steerer.load_card(concept_card, alpha=1.0)

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    num_layers = 32
    layers = [MagicMock() for _ in range(num_layers)]
    mock_model.model.layers = layers

    handles = []
    registered_hooks: list[tuple[int, object]] = []

    for i, layer in enumerate(layers):

        def make_register(idx):
            def register_forward_hook(fn):
                registered_hooks.append((idx, fn))
                h = MagicMock()
                handles.append(h)
                return h

            return register_forward_hook

        layer.register_forward_hook.side_effect = make_register(i)

    steerer._model = mock_model

    steerer._register_hooks()

    target_layer = concept_card.meta.layer
    hook_entries = [(i, fn) for i, fn in registered_hooks if i == target_layer]
    assert len(hook_entries) == 1

    _, hook_fn = hook_entries[0]
    batch, seq_len = 1, 8
    hidden = torch.zeros(batch, seq_len, hidden_dim)
    result = hook_fn(None, None, (hidden,))

    v = concept_card.vector
    expected = hidden + 1.0 * v.view(1, 1, -1)
    assert torch.allclose(result[0], expected)


def _build_streaming_mocks(
    hidden_dim: int,
    num_layers: int = 32,
) -> tuple[MagicMock, MagicMock, dict[int, list[Callable]]]:
    hooks_by_layer: dict[int, list[Callable]] = {}
    layers: list[MagicMock] = []
    for i in range(num_layers):
        layer = MagicMock()

        def make_register(idx: int) -> Callable:
            def register_forward_hook(fn: Callable) -> MagicMock:
                hooks_by_layer.setdefault(idx, []).append(fn)
                return MagicMock()

            return register_forward_hook

        layer.register_forward_hook.side_effect = make_register(i)
        layers.append(layer)

    call_count = [0]

    def fake_forward(**kwargs: object) -> MagicMock:
        call_count[0] += 1
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        batch, seq_len = input_ids.shape
        fake_hidden = torch.randn(batch, seq_len, hidden_dim)
        for fns in hooks_by_layer.values():
            for fn in fns:
                fn(None, None, (fake_hidden,))  # type: ignore[operator]
        logits = torch.full((batch, seq_len, _VOCAB_SIZE), -1e9)
        if call_count[0] >= 3:
            logits[:, -1, _EOS_ID] = 100.0
        else:
            logits[:, -1, call_count[0]] = 100.0
        outputs = MagicMock()
        outputs.logits = logits
        outputs.past_key_values = MagicMock()
        return outputs

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.model.layers = layers
    mock_model.side_effect = fake_forward

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = _EOS_ID
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.chat_template = None
    mock_tokenizer.added_tokens_encoder = {}
    mock_tokenizer.side_effect = lambda *args, **kwargs: {
        "input_ids": torch.randint(1, _VOCAB_SIZE, (1, 5)),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    mock_tokenizer.decode.side_effect = lambda ids, skip_special_tokens=True: f"t{ids[0].item()}"

    return mock_model, mock_tokenizer, hooks_by_layer


def test_generate_stream_yields_tokens_and_activations(
    hidden_dim: int, concept_card: ConceptCard
) -> None:
    mock_model, mock_tokenizer, _ = _build_streaming_mocks(hidden_dim)
    steerer = Steerer("test-model")
    steerer.load_card(concept_card, alpha=1.0)
    steerer._model = mock_model
    steerer._tokenizer = mock_tokenizer

    results = list(steerer.generate_stream("test prompt", greedy=True))

    assert len(results) > 0
    for token, activations, projection in results:
        assert isinstance(token, str)
        assert isinstance(activations, dict)
        assert "formality" in activations
        assert -1.0 <= activations["formality"] <= 1.0
        assert projection is None or (isinstance(projection, tuple) and len(projection) == 2)


def test_generate_stream_stops_at_eos(hidden_dim: int, concept_card: ConceptCard) -> None:
    mock_model, mock_tokenizer, _ = _build_streaming_mocks(hidden_dim)
    steerer = Steerer("test-model")
    steerer.load_card(concept_card, alpha=1.0)
    steerer._model = mock_model
    steerer._tokenizer = mock_tokenizer

    results = list(steerer.generate_stream("test", max_new_tokens=100, greedy=True))

    assert len(results) == 2


def test_generate_non_streaming(hidden_dim: int, concept_card: ConceptCard) -> None:
    layers: list[MagicMock] = [MagicMock() for _ in range(32)]
    for layer in layers:
        layer.register_forward_hook.return_value = MagicMock()

    prompt_len = 3
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.model.layers = layers
    mock_model.generate.return_value = torch.tensor([[0, 0, 0, 42, 7]])

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = _EOS_ID
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.chat_template = None
    mock_tokenizer.added_tokens_encoder = {}
    mock_tokenizer.side_effect = lambda *args, **kwargs: {
        "input_ids": torch.zeros(1, prompt_len, dtype=torch.long),
        "attention_mask": torch.ones(1, prompt_len, dtype=torch.long),
    }
    mock_tokenizer.decode.return_value = "generated output"

    steerer = Steerer("test-model")
    steerer.load_card(concept_card, alpha=1.0)
    steerer._model = mock_model
    steerer._tokenizer = mock_tokenizer

    output = steerer.generate("test prompt", greedy=True)

    assert output == "generated output"
    decoded_ids = mock_tokenizer.decode.call_args[0][0]
    assert torch.equal(decoded_ids, torch.tensor([42, 7]))


def test_validate_card_compatibility_wrong_dim(hidden_dim: int) -> None:
    meta = ConceptCardMeta(
        model_id="test-model",
        layer=0,
        hidden_dim=hidden_dim,
        concept="formality",
    )
    card = ConceptCard(meta=meta, vector=torch.randn(hidden_dim) / torch.randn(hidden_dim).norm())
    mock_model = MagicMock()
    mock_model.config.hidden_size = hidden_dim + 128
    mock_model.model.layers = [MagicMock() for _ in range(32)]
    with pytest.raises(IncompatibleCardError, match="hidden_dim"):
        Steerer.validate_card_compatibility(card, mock_model)


def test_validate_card_compatibility_wrong_layer(hidden_dim: int) -> None:
    meta = ConceptCardMeta(
        model_id="test-model",
        layer=100,
        hidden_dim=hidden_dim,
        concept="formality",
    )
    card = ConceptCard(meta=meta, vector=torch.randn(hidden_dim) / torch.randn(hidden_dim).norm())
    mock_model = MagicMock()
    mock_model.config.hidden_size = hidden_dim
    mock_model.model.layers = [MagicMock() for _ in range(32)]
    with pytest.raises(IncompatibleCardError, match="layer"):
        Steerer.validate_card_compatibility(card, mock_model)


def test_validate_card_compatibility_valid(hidden_dim: int) -> None:
    meta = ConceptCardMeta(
        model_id="test-model",
        layer=16,
        hidden_dim=hidden_dim,
        concept="formality",
    )
    card = ConceptCard(meta=meta, vector=torch.randn(hidden_dim) / torch.randn(hidden_dim).norm())
    mock_model = MagicMock()
    mock_model.config.hidden_size = hidden_dim
    mock_model.model.layers = [MagicMock() for _ in range(32)]
    Steerer.validate_card_compatibility(card, mock_model)


def test_calibrate_baseline_shape(hidden_dim: int, concept_card: ConceptCard) -> None:
    mock_model, _, hooks_by_layer = _build_streaming_mocks(hidden_dim)

    def full_tokenize(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        texts = args[0] if args else []
        batch_size = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.randint(1, _VOCAB_SIZE, (batch_size, 5)),
            "attention_mask": torch.ones(batch_size, 5, dtype=torch.long),
        }

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = _EOS_ID
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.side_effect = full_tokenize

    steerer = Steerer("test-model")
    steerer.load_card(concept_card, alpha=1.0)
    steerer._model = mock_model
    steerer._tokenizer = mock_tokenizer
    corpus = ["hello world", "foo bar baz", "test prompt"]
    steerer.calibrate_baseline(corpus, batch_size=2)
    assert 16 in steerer._baselines
    assert steerer._baselines[16].shape == (hidden_dim,)


def test_calibrate_baseline_multi_layer(hidden_dim: int) -> None:
    mock_model, _, _ = _build_streaming_mocks(hidden_dim)

    def full_tokenize(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        texts = args[0] if args else []
        batch_size = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.randint(1, _VOCAB_SIZE, (batch_size, 5)),
            "attention_mask": torch.ones(batch_size, 5, dtype=torch.long),
        }

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = _EOS_ID
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.side_effect = full_tokenize

    meta1 = ConceptCardMeta(
        model_id="test-model", layer=8, hidden_dim=hidden_dim, concept="formality"
    )
    meta2 = ConceptCardMeta(
        model_id="test-model", layer=16, hidden_dim=hidden_dim, concept="creativity"
    )
    v = torch.randn(hidden_dim)
    v = v / v.norm()
    card1 = ConceptCard(meta=meta1, vector=v.clone())
    card2 = ConceptCard(meta=meta2, vector=v.clone())

    steerer = Steerer("test-model")
    steerer.load_card(card1, alpha=1.0)
    steerer.load_card(card2, alpha=1.0)
    steerer._model = mock_model
    steerer._tokenizer = mock_tokenizer
    corpus = ["hello world", "foo bar"]
    steerer.calibrate_baseline(corpus, batch_size=2)
    assert 8 in steerer._baselines
    assert 16 in steerer._baselines
    assert steerer._baselines[8].shape == (hidden_dim,)
    assert steerer._baselines[16].shape == (hidden_dim,)


def test_generate_stream_cleanup(hidden_dim: int, concept_card: ConceptCard) -> None:
    mock_model, mock_tokenizer, _ = _build_streaming_mocks(hidden_dim)
    steerer = Steerer("test-model")
    steerer.load_card(concept_card, alpha=1.0)
    steerer._model = mock_model
    steerer._tokenizer = mock_tokenizer
    list(steerer.generate_stream("test", max_new_tokens=10, greedy=True))
    assert steerer._last_hidden is None
    assert steerer._activation_store == {}


def test_hook_pre_steer_sim_zero_for_zero_input(hidden_dim: int) -> None:
    meta = ConceptCardMeta(
        model_id="test-model",
        layer=16,
        hidden_dim=hidden_dim,
        default_alpha=1.0,
        concept="formality",
    )
    v = torch.zeros(hidden_dim)
    v[0] = 1.0
    card = ConceptCard(meta=meta, vector=v)

    layers: list[MagicMock] = []
    registered_hook_fn = [None]

    for i in range(32):
        layer = MagicMock()

        def make_register(idx: int) -> object:
            def register_forward_hook(fn: object) -> MagicMock:
                if idx == 16:
                    registered_hook_fn[0] = fn
                return MagicMock()

            return register_forward_hook

        layer.register_forward_hook.side_effect = make_register(i)
        layers.append(layer)

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.model.layers = layers

    steerer = Steerer("test-model")
    steerer.load_card(card, alpha=2.0)
    steerer._model = mock_model
    steerer._register_hooks()

    hook_fn = registered_hook_fn[0]
    assert hook_fn is not None

    hidden = torch.zeros(1, 4, hidden_dim)
    hook_fn(None, None, (hidden,))

    assert abs(steerer._activation_store["formality"]) < 1e-5


def test_hook_pre_steer_sim_not_inflated(hidden_dim: int) -> None:
    meta = ConceptCardMeta(
        model_id="test-model",
        layer=16,
        hidden_dim=hidden_dim,
        default_alpha=1.0,
        concept="formality",
    )
    v = torch.zeros(hidden_dim)
    v[0] = 1.0
    card = ConceptCard(meta=meta, vector=v)

    layers: list[MagicMock] = []
    registered_hook_fn = [None]

    for i in range(32):
        layer = MagicMock()

        def make_register(idx: int) -> object:
            def register_forward_hook(fn: object) -> MagicMock:
                if idx == 16:
                    registered_hook_fn[0] = fn
                return MagicMock()

            return register_forward_hook

        layer.register_forward_hook.side_effect = make_register(i)
        layers.append(layer)

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.model.layers = layers

    steerer = Steerer("test-model")
    steerer.load_card(card, alpha=2.0)
    steerer._model = mock_model
    steerer._register_hooks()

    hook_fn = registered_hook_fn[0]
    assert hook_fn is not None

    torch.manual_seed(7)
    hidden = torch.randn(1, 4, hidden_dim)
    hook_fn(None, None, (hidden,))

    expected_sim = (
        F.cosine_similarity(hidden[:, -1, :].float(), v.float().unsqueeze(0), dim=-1).mean().item()
    )
    assert abs(steerer._activation_store["formality"] - expected_sim) < 1e-5


def test_baseline_subtraction_affects_sim(hidden_dim: int) -> None:
    meta = ConceptCardMeta(
        model_id="test-model",
        layer=16,
        hidden_dim=hidden_dim,
        default_alpha=1.0,
        concept="formality",
    )
    v = torch.zeros(hidden_dim)
    v[0] = 1.0
    card = ConceptCard(meta=meta, vector=v)

    layers: list[MagicMock] = []
    registered_hook_fn = [None]

    for i in range(32):
        layer = MagicMock()

        def make_register(idx: int) -> object:
            def register_forward_hook(fn: object) -> MagicMock:
                if idx == 16:
                    registered_hook_fn[0] = fn
                return MagicMock()

            return register_forward_hook

        layer.register_forward_hook.side_effect = make_register(i)
        layers.append(layer)

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.model.layers = layers

    steerer = Steerer("test-model")
    steerer.load_card(card, alpha=1.0)
    steerer._model = mock_model
    steerer._register_hooks()

    hidden = torch.ones(1, 4, hidden_dim) * 0.5
    registered_hook_fn[0](None, None, (hidden,))
    sim_no_baseline = steerer._activation_store["formality"]

    baseline = torch.zeros(hidden_dim)
    baseline[0] = 0.4
    steerer._baselines[16] = baseline
    registered_hook_fn[0](None, None, (hidden,))
    sim_with_baseline = steerer._activation_store["formality"]

    assert abs(sim_no_baseline - sim_with_baseline) > 1e-5


def test_hook_groups_two_concepts_at_same_layer(hidden_dim: int) -> None:
    meta1 = ConceptCardMeta(
        model_id="test-model",
        layer=16,
        hidden_dim=hidden_dim,
        default_alpha=1.0,
        concept="formality",
    )
    meta2 = ConceptCardMeta(
        model_id="test-model",
        layer=16,
        hidden_dim=hidden_dim,
        default_alpha=1.0,
        concept="certainty",
    )
    v1 = torch.zeros(hidden_dim)
    v1[0] = 1.0
    v2 = torch.zeros(hidden_dim)
    v2[1] = 1.0
    card1 = ConceptCard(meta=meta1, vector=v1)
    card2 = ConceptCard(meta=meta2, vector=v2)

    hooks_registered: list[int] = []
    layers: list[MagicMock] = []
    for i in range(32):
        layer = MagicMock()

        def make_register(idx: int) -> Callable:
            def register_forward_hook(fn: Callable) -> MagicMock:
                hooks_registered.append(idx)
                return MagicMock()

            return register_forward_hook

        layer.register_forward_hook.side_effect = make_register(i)
        layers.append(layer)

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.model.layers = layers

    steerer = Steerer("test-model")
    steerer.load_card(card1, alpha=1.0)
    steerer.load_card(card2, alpha=1.0)
    steerer._model = mock_model
    steerer._register_hooks()

    assert hooks_registered.count(16) == 1

    hook_fn = layers[16].register_forward_hook.call_args[0][0]
    batch, seq_len = 1, 4
    hidden = torch.zeros(batch, seq_len, hidden_dim)
    result = hook_fn(None, None, (hidden,))

    expected = hidden + v1.view(1, 1, -1) + v2.view(1, 1, -1)
    assert torch.allclose(result[0], expected)


def test_steer_prompt_false_skips_prompt(hidden_dim: int) -> None:
    meta = ConceptCardMeta(
        model_id="test-model",
        layer=16,
        hidden_dim=hidden_dim,
        default_alpha=1.0,
        concept="formality",
    )
    v = torch.zeros(hidden_dim)
    v[0] = 1.0
    card = ConceptCard(meta=meta, vector=v)

    layers: list[MagicMock] = []
    registered_hook_fn = [None]

    for i in range(32):
        layer = MagicMock()

        def make_register(idx: int) -> object:
            def register_forward_hook(fn: object) -> MagicMock:
                if idx == 16:
                    registered_hook_fn[0] = fn
                return MagicMock()

            return register_forward_hook

        layer.register_forward_hook.side_effect = make_register(i)
        layers.append(layer)

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.model.layers = layers

    steerer = Steerer("test-model")
    steerer.load_card(card, alpha=2.0)
    steerer._model = mock_model
    steerer._register_hooks()

    hook_fn = registered_hook_fn[0]
    assert hook_fn is not None

    steerer._steer_prompt = False
    hidden_prompt = torch.zeros(1, 4, hidden_dim)
    result_prompt = hook_fn(None, None, (hidden_prompt,))
    assert torch.allclose(result_prompt[0], hidden_prompt)

    hidden_single = torch.zeros(1, 1, hidden_dim)
    result_single = hook_fn(None, None, (hidden_single,))
    expected = hidden_single + 2.0 * v.view(1, 1, -1)
    assert torch.allclose(result_single[0], expected)


def test_combine_cards_single_layer(hidden_dim: int) -> None:
    meta1 = ConceptCardMeta(model_id="test-model", layer=16, hidden_dim=hidden_dim, concept="formality")
    meta2 = ConceptCardMeta(model_id="test-model", layer=16, hidden_dim=hidden_dim, concept="safety")
    torch.manual_seed(0)
    v1 = torch.randn(hidden_dim)
    v1 = v1 / v1.norm()
    v2 = torch.randn(hidden_dim)
    v2 = v2 / v2.norm()
    card1 = ConceptCard(meta=meta1, vector=v1)
    card2 = ConceptCard(meta=meta2, vector=v2)

    steerer = Steerer("test-model")
    steerer.load_card(card1, alpha=1.5)
    steerer.load_card(card2, alpha=-0.5)

    combined = combine_cards(steerer.loaded_cards(), "blend")

    assert combined.meta.concept == "blend"
    assert combined.meta.layer == 16
    assert abs(combined.vector.norm().item() - 1.0) < 1e-5

    v_raw = 1.5 * v1 + (-0.5) * v2
    expected_alpha = v_raw.norm().item()
    assert abs(combined.meta.default_alpha - expected_alpha) < 1e-4


def test_combine_cards_multi_layer_raises(hidden_dim: int) -> None:
    meta1 = ConceptCardMeta(model_id="test-model", layer=16, hidden_dim=hidden_dim, concept="formality")
    meta2 = ConceptCardMeta(model_id="test-model", layer=24, hidden_dim=hidden_dim, concept="safety")
    v = torch.randn(hidden_dim)
    v = v / v.norm()
    card1 = ConceptCard(meta=meta1, vector=v.clone())
    card2 = ConceptCard(meta=meta2, vector=v.clone())

    steerer = Steerer("test-model")
    steerer.load_card(card1, alpha=1.0)
    steerer.load_card(card2, alpha=1.0)

    with pytest.raises(ValueError, match="different layers"):
        combine_cards(steerer.loaded_cards(), "blend")


def test_combine_cards_zero_vector_raises(hidden_dim: int) -> None:
    meta1 = ConceptCardMeta(model_id="test-model", layer=16, hidden_dim=hidden_dim, concept="formality")
    meta2 = ConceptCardMeta(model_id="test-model", layer=16, hidden_dim=hidden_dim, concept="formality_neg")
    v = torch.randn(hidden_dim)
    v = v / v.norm()
    card1 = ConceptCard(meta=meta1, vector=v.clone())
    card2 = ConceptCard(meta=meta2, vector=v.clone())

    steerer = Steerer("test-model")
    steerer.load_card(card1, alpha=1.0)
    steerer.load_card(card2, alpha=-1.0)

    with pytest.raises(ValueError, match="cancel out"):
        combine_cards(steerer.loaded_cards(), "blend")


def test_load_card_model_mismatch_warns(
    concept_card: ConceptCard, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    steerer = Steerer("meta-llama/Llama-3.1-8B-Instruct")
    with caplog.at_level(logging.WARNING, logger="autosae.steerer"):
        steerer.load_card(concept_card)
    assert any("test-model" in r.message and "Llama-3.1-8B-Instruct" in r.message for r in caplog.records)


def test_generate_stream_mps_seed_fallback(hidden_dim: int, concept_card: ConceptCard) -> None:
    mock_model, mock_tokenizer, _ = _build_streaming_mocks(hidden_dim)
    steerer = Steerer("test-model")
    steerer.load_card(concept_card, alpha=1.0)
    steerer._model = mock_model
    steerer._tokenizer = mock_tokenizer

    cpu_gen = torch.Generator(device="cpu")
    attempts: list[object] = []

    def mock_generator(*args: object, **kwargs: object) -> torch.Generator:
        attempts.append(kwargs.get("device") or (args[0] if args else None))
        if len(attempts) == 1:
            raise RuntimeError("Generator not supported on this device")
        return cpu_gen

    with patch("torch.Generator", side_effect=mock_generator):
        results = list(steerer.generate_stream("test", seed=42, greedy=True))

    assert len(results) > 0
    assert len(attempts) == 2
