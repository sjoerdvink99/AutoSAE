from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from autosae_server.engine.base import InferenceEngine
from autosae_server.schemas import ConceptGeometry, InverseProjectRequest, InverseProjectResponse
from autosae_server.state import get_engine

router = APIRouter(prefix="/geometry", tags=["geometry"])


@router.get("", response_model=None)
async def get_geometry(
    engine: InferenceEngine = Depends(get_engine),
) -> ConceptGeometry | Response:
    raw = await engine.get_geometry()
    if raw is None:
        return Response(status_code=204)
    return ConceptGeometry(**raw)  # type: ignore[arg-type]


@router.post("/inverse", response_model=InverseProjectResponse)
async def inverse_project(
    req: InverseProjectRequest,
    engine: InferenceEngine = Depends(get_engine),
) -> InverseProjectResponse:
    deltas = await engine.inverse_project(
        delta=req.delta,
        mode=req.mode,
        anchor_concepts=req.anchor_concepts,
        alpha_bounds=req.alpha_bounds,
        max_step=req.max_step,
    )
    return InverseProjectResponse(alpha_deltas=deltas)
