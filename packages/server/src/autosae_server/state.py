from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autosae_server.engine.base import InferenceEngine

_engine: InferenceEngine | None = None


def set_engine(engine: InferenceEngine) -> None:
    global _engine
    _engine = engine


def get_engine() -> InferenceEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialized. Start the server with --model-id.")
    return _engine
