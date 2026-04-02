from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from autosae.exceptions import ConceptCardNotFoundError, ConceptNotLoadedError
from autosae_server.engine.base import InferenceEngine
from autosae_server.schemas import (
    CardInfo,
    ExtractCardRequest,
    ExtractCardResponse,
    LayerScore,
    LayerSweepRequest,
    LayerSweepResponse,
    LoadCardRequest,
    UpdateAlphaRequest,
)
from autosae_server.state import get_engine

router = APIRouter(prefix="/cards", tags=["cards"])


@router.get("", response_model=list[CardInfo])
async def list_cards(engine: InferenceEngine = Depends(get_engine)) -> list[CardInfo]:
    infos = await engine.get_card_infos()
    return [
        CardInfo(
            concept=info.concept,
            model_id=info.model_id,
            layer=info.layer,
            hidden_dim=info.hidden_dim,
            alpha=info.alpha,
            description=info.description,
            p_value=info.p_value,
            separability_score=info.separability_score,
            layer_selection=info.layer_selection,
            num_positive=info.num_positive,
            num_negative=info.num_negative,
            bootstrap_variance=info.bootstrap_variance,
            mean_hidden_norm=info.mean_hidden_norm,
        )
        for info in infos
    ]


@router.post("/load", status_code=status.HTTP_201_CREATED)
async def load_card(
    body: LoadCardRequest, engine: InferenceEngine = Depends(get_engine)
) -> dict[str, str]:
    if body.path is None and body.registry_concept is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provide either 'path' or 'registry_concept'.",
        )
    try:
        await engine.load_card(
            path=body.path,
            registry_concept=body.registry_concept,
            registry_model=body.registry_model,
            alpha=body.alpha,
        )
    except (FileNotFoundError, ValueError, ConceptCardNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {"status": "loaded"}


@router.delete("/{concept}", status_code=status.HTTP_204_NO_CONTENT)
async def unload_card(concept: str, engine: InferenceEngine = Depends(get_engine)) -> None:
    try:
        await engine.unload_card(concept)
    except ConceptNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.patch("/{concept}/alpha")
async def update_alpha(
    concept: str, body: UpdateAlphaRequest, engine: InferenceEngine = Depends(get_engine)
) -> dict[str, object]:
    try:
        await engine.set_alpha(concept, body.alpha)
    except ConceptNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {"concept": concept, "alpha": body.alpha}


@router.post("/extract", response_model=ExtractCardResponse)
async def extract_card(
    body: ExtractCardRequest, engine: InferenceEngine = Depends(get_engine)
) -> ExtractCardResponse:
    if not engine.capabilities.supports_extraction:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This backend does not support card extraction.",
        )
    try:
        path, info = await engine.extract_card(
            concept=body.concept,
            description=body.description,
            positive=body.positive,
            negative=body.negative,
            default_alpha=body.default_alpha,
            layer_frac=body.layer_frac,
            auto_layer=body.auto_layer,
            use_robust_mean=body.use_robust_mean,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    return ExtractCardResponse(
        concept=info.concept,
        model_id=info.model_id,
        layer=info.layer,
        hidden_dim=info.hidden_dim,
        default_alpha=info.alpha,
        description=info.description,
        path=path,
        p_value=info.p_value,
        separability_score=info.separability_score,
        layer_selection=info.layer_selection,
    )


@router.get("/{concept}/download")
async def download_card(
    concept: str, engine: InferenceEngine = Depends(get_engine)
) -> FileResponse:
    path = await engine.get_card_path(concept)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Card '{concept}' not found on disk.",
        )
    return FileResponse(
        path=path,
        media_type="application/octet-stream",
        filename=f"{concept}.safetensors",
    )


@router.post("/layer-sweep", response_model=LayerSweepResponse)
async def layer_sweep(
    body: LayerSweepRequest, engine: InferenceEngine = Depends(get_engine)
) -> LayerSweepResponse:
    if not engine.capabilities.supports_extraction:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This backend does not support layer sweep.",
        )
    try:
        scores, recommended = await engine.layer_sweep(
            positive=body.positive,
            negative=body.negative,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    return LayerSweepResponse(
        layers=[LayerScore(layer=i, score=s) for i, s in scores],
        recommended_layer=recommended,
    )
