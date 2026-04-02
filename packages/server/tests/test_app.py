from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from fastapi.testclient import TestClient

from autosae_server.app import create_app
from autosae_server.engine.base import (
    EngineCapabilities,
    GenerationChunk,
    InferenceEngine,
    LoadedCardInfo,
)
from autosae_server.state import set_engine

_DEFAULT_CAPABILITIES = EngineCapabilities(
    supports_steering=True,
    supports_extraction=True,
)


def _make_engine(
    card_infos: list[LoadedCardInfo] | None = None,
) -> InferenceEngine:
    engine = MagicMock(spec=InferenceEngine)
    type(engine).capabilities = PropertyMock(return_value=_DEFAULT_CAPABILITIES)
    engine.get_card_infos = AsyncMock(return_value=card_infos or [])
    engine.load_card = AsyncMock()
    engine.unload_card = AsyncMock()
    engine.set_alpha = AsyncMock()
    engine.extract_card = AsyncMock()
    engine.get_card_path = AsyncMock(return_value=None)
    engine.get_concept_map = AsyncMock(return_value=None)
    return engine


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def engine() -> InferenceEngine:
    return _make_engine()


@pytest.fixture(autouse=True)
def inject_engine(engine: InferenceEngine) -> None:
    set_engine(engine)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_list_cards_empty(client: TestClient) -> None:
    resp = client.get("/cards")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_cards_with_cards(client: TestClient, engine: InferenceEngine) -> None:
    engine.get_card_infos = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            LoadedCardInfo(
                concept="formality",
                model_id="meta-llama/Llama-3.1-8B",
                layer=19,
                hidden_dim=4096,
                alpha=2.0,
                description="Formal ↔ casual register",
            )
        ]
    )
    resp = client.get("/cards")
    assert resp.status_code == 200
    cards = resp.json()
    assert len(cards) == 1
    assert cards[0]["concept"] == "formality"
    assert cards[0]["alpha"] == 2.0


def test_load_card_registry(client: TestClient) -> None:
    resp = client.post("/cards/load", json={"registry_concept": "formality"})
    assert resp.status_code == 201
    assert resp.json() == {"status": "loaded"}


def test_load_card_path(client: TestClient) -> None:
    resp = client.post("/cards/load", json={"path": "/tmp/card.safetensors"})
    assert resp.status_code == 201


def test_load_card_missing_params(client: TestClient) -> None:
    resp = client.post("/cards/load", json={})
    assert resp.status_code == 422


def test_load_card_not_found(client: TestClient, engine: InferenceEngine) -> None:
    engine.load_card = AsyncMock(side_effect=FileNotFoundError("not found"))  # type: ignore[method-assign]
    resp = client.post("/cards/load", json={"path": "/nonexistent.safetensors"})
    assert resp.status_code == 404


def test_load_card_registry_not_found(client: TestClient, engine: InferenceEngine) -> None:
    from autosae.exceptions import ConceptCardNotFoundError

    engine.load_card = AsyncMock(side_effect=ConceptCardNotFoundError("not in hub"))  # type: ignore[method-assign]
    resp = client.post("/cards/load", json={"registry_concept": "nonexistent"})
    assert resp.status_code == 404


def test_unload_card(client: TestClient) -> None:
    resp = client.delete("/cards/formality")
    assert resp.status_code == 204


def test_unload_card_not_found(client: TestClient, engine: InferenceEngine) -> None:
    from autosae.exceptions import ConceptNotLoadedError

    engine.unload_card = AsyncMock(side_effect=ConceptNotLoadedError("not loaded"))  # type: ignore[method-assign]
    resp = client.delete("/cards/nonexistent")
    assert resp.status_code == 404


def test_update_alpha(client: TestClient) -> None:
    resp = client.patch("/cards/formality/alpha", json={"alpha": 1.5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["concept"] == "formality"
    assert data["alpha"] == 1.5


def test_update_alpha_not_found(client: TestClient, engine: InferenceEngine) -> None:
    from autosae.exceptions import ConceptNotLoadedError

    engine.set_alpha = AsyncMock(side_effect=ConceptNotLoadedError("not loaded"))  # type: ignore[method-assign]
    resp = client.patch("/cards/nonexistent/alpha", json={"alpha": 1.0})
    assert resp.status_code == 404


def test_health_includes_capabilities(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "capabilities" in data
    assert data["capabilities"]["supports_steering"] is True
    assert data["capabilities"]["supports_extraction"] is True


def test_extract_card(client: TestClient, engine: InferenceEngine) -> None:
    from autosae_server.engine.base import LoadedCardInfo

    engine.extract_card = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            "./cards/formality.safetensors",
            LoadedCardInfo(
                concept="formality",
                model_id="gpt2",
                layer=6,
                hidden_dim=768,
                alpha=1.0,
                description="test",
            ),
        )
    )
    resp = client.post(
        "/cards/extract",
        json={
            "concept": "formality",
            "description": "test",
            "positive": ["formal text"],
            "negative": ["casual text"],
            "default_alpha": 1.0,
            "layer_frac": 0.6,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["concept"] == "formality"
    assert data["path"] == "./cards/formality.safetensors"



def test_download_card_endpoint(
    client: TestClient, engine: InferenceEngine, tmp_path: object
) -> None:
    import pathlib

    card_file = pathlib.Path(str(tmp_path)) / "formality.safetensors"
    card_file.write_bytes(b"fake safetensors content")
    engine.get_card_path = AsyncMock(return_value=str(card_file))  # type: ignore[method-assign]
    resp = client.get("/cards/formality/download")
    assert resp.status_code == 200
    assert "application/octet-stream" in resp.headers["content-type"]


def test_generate_blocking(client: TestClient, engine: InferenceEngine) -> None:
    async def _stream(*args: object, **kwargs: object) -> AsyncIterator[GenerationChunk]:
        yield GenerationChunk(token="Hello", activations={"formality": 0.8})
        yield GenerationChunk(token=" world", activations={"formality": 0.7})

    engine.generate_stream = _stream  # type: ignore[method-assign]

    resp = client.post(
        "/generate", json={"prompt": "Say hello", "max_new_tokens": 10, "temperature": 1.0}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Hello world"
    assert data["activations"]["formality"] == pytest.approx(0.7)


def test_load_card_default_registry_model(client: TestClient, engine: InferenceEngine) -> None:
    type(engine).model_id = PropertyMock(return_value="meta-llama/Llama-3.1-8B-Instruct")  # type: ignore[method-assign]
    resp = client.post("/cards/load", json={"registry_concept": "formality"})
    assert resp.status_code == 201
    call_kwargs = engine.load_card.call_args  # type: ignore[union-attr]
    assert call_kwargs.kwargs["registry_model"] is None or call_kwargs.args[2] is None


def test_concept_map_empty(client: TestClient) -> None:
    resp = client.get("/concepts/map")
    assert resp.status_code == 204


def test_concept_map_with_points(client: TestClient, engine: InferenceEngine) -> None:
    engine.get_concept_map = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "points": [
                {
                    "concept": "formality",
                    "x": 0.8,
                    "y": 0.2,
                    "loaded": True,
                    "separability_score": 0.84,
                    "bootstrap_confidence": 0.001,
                    "model_id": "meta-llama/Llama-3.1-8B",
                },
                {
                    "concept": "safety",
                    "x": -0.3,
                    "y": 0.7,
                    "loaded": False,
                    "separability_score": None,
                    "bootstrap_confidence": None,
                    "model_id": "meta-llama/Llama-3.1-8B",
                },
            ],
            "model_id": "meta-llama/Llama-3.1-8B",
        }
    )
    resp = client.get("/concepts/map")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "meta-llama/Llama-3.1-8B"
    assert len(data["points"]) == 2
    assert data["points"][0]["concept"] == "formality"
    assert data["points"][0]["loaded"] is True
    assert data["points"][1]["concept"] == "safety"
    assert data["points"][1]["loaded"] is False


def test_list_cards_includes_mean_hidden_norm(client: TestClient, engine: InferenceEngine) -> None:
    engine.get_card_infos = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            LoadedCardInfo(
                concept="formality",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                layer=19,
                hidden_dim=4096,
                alpha=2.0,
                description="test",
                mean_hidden_norm=42.5,
            )
        ]
    )
    resp = client.get("/cards")
    assert resp.status_code == 200
    cards = resp.json()
    assert len(cards) == 1
    assert cards[0]["mean_hidden_norm"] == pytest.approx(42.5)
