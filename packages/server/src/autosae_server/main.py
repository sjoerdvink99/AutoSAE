from __future__ import annotations

import argparse

import uvicorn

from autosae_server.app import create_app
from autosae_server.engine.transformers_engine import TransformersEngine
from autosae_server.state import set_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoSAE inference server")
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--cors-origin", action="append", dest="cors_origins", metavar="ORIGIN")
    args = parser.parse_args()

    engine = TransformersEngine(
        model_id=args.model_id,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )
    set_engine(engine)

    app = create_app(cors_origins=args.cors_origins)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
