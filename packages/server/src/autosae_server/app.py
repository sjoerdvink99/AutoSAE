from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autosae_server.routes.cards import router as cards_router
from autosae_server.routes.concepts import router as concepts_router
from autosae_server.routes.generate import router as generate_router
from autosae_server.routes.geometry import router as geometry_router
from autosae_server.routes.health import router as health_router
from autosae_server.state import get_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield
    try:
        engine = get_engine()
        await engine.shutdown()
    except RuntimeError:
        pass


def create_app(cors_origins: list[str] | None = None) -> FastAPI:
    app = FastAPI(
        title="AutoSAE",
        description="Activation steering inference server",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(cards_router)
    app.include_router(concepts_router)
    app.include_router(generate_router)
    app.include_router(geometry_router)

    return app
