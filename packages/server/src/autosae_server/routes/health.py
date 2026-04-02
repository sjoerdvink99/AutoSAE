from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from autosae_server.engine.base import EngineCapabilities, InferenceEngine
from autosae_server.state import get_engine

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    capabilities: EngineCapabilities


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(engine: InferenceEngine = Depends(get_engine)) -> HealthResponse:
    from autosae_server import __version__

    return HealthResponse(
        status="ok",
        version=__version__,
        capabilities=engine.capabilities,
    )
