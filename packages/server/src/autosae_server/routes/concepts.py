from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from autosae_server.engine.base import InferenceEngine
from autosae_server.schemas import ConceptMapResponse
from autosae_server.state import get_engine

router = APIRouter(prefix="/concepts", tags=["concepts"])


@router.get("/map", response_model=None)
async def get_concept_map(
    engine: InferenceEngine = Depends(get_engine),
) -> ConceptMapResponse | Response:
    raw = await engine.get_concept_map()
    if raw is None:
        return Response(status_code=204)
    return ConceptMapResponse(**raw)  # type: ignore[arg-type]
