from __future__ import annotations

import contextlib

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from autosae_server.engine.base import InferenceEngine
from autosae_server.schemas import (
    GenerateRequest,
    GenerateResponse,
    WsDoneMessage,
    WsErrorMessage,
    WsIncomingMessage,
    WsTokenMessage,
)
from autosae_server.state import get_engine

router = APIRouter(tags=["generate"])


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    body: GenerateRequest, engine: InferenceEngine = Depends(get_engine)
) -> GenerateResponse:
    text_parts: list[str] = []
    final_activations: dict[str, float] = {}

    messages = (
        [{"role": m.role, "content": m.content} for m in body.messages] if body.messages else None
    )
    async for chunk in engine.generate_stream(
        prompt=body.prompt,
        messages=messages,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        seed=body.seed,
        greedy=body.greedy,
        steer_prompt=False,
        system_prompt=body.system_prompt,
        repetition_penalty=body.repetition_penalty,
    ):
        if chunk.error is not None:
            raise RuntimeError(chunk.error)
        text_parts.append(chunk.token)
        final_activations = chunk.activations

    return GenerateResponse(text="".join(text_parts), activations=final_activations)


@router.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket) -> None:
    engine = get_engine()
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = WsIncomingMessage.model_validate_json(raw)
            except ValidationError as exc:
                await websocket.send_text(
                    WsErrorMessage(message=f"Invalid message: {exc}").model_dump_json()
                )
                continue

            if msg.type != "generate":
                await websocket.send_text(
                    WsErrorMessage(message=f"Unknown message type: {msg.type!r}").model_dump_json()
                )
                continue

            try:
                ws_messages = (
                    [{"role": m.role, "content": m.content} for m in msg.messages]
                    if msg.messages
                    else None
                )
                async for chunk in engine.generate_stream(
                    prompt=msg.prompt,
                    messages=ws_messages,
                    max_new_tokens=msg.max_new_tokens,
                    temperature=msg.temperature,
                    top_p=msg.top_p,
                    seed=msg.seed,
                    greedy=msg.greedy,
                    steer_prompt=False,
                    system_prompt=msg.system_prompt,
                    repetition_penalty=msg.repetition_penalty,
                ):
                    if chunk.error is not None:
                        await websocket.send_text(
                            WsErrorMessage(message=chunk.error).model_dump_json()
                        )
                        break
                    await websocket.send_text(
                        WsTokenMessage(
                            token=chunk.token,
                            activations=chunk.activations,
                            projection=chunk.projection,
                        ).model_dump_json()
                    )
                else:
                    await websocket.send_text(WsDoneMessage().model_dump_json())
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                with contextlib.suppress(Exception):
                    await websocket.send_text(WsErrorMessage(message=str(exc)).model_dump_json())

    except WebSocketDisconnect:
        pass
