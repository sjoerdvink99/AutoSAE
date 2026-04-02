# Contributing to AutoSAE

## Setup

```bash
git clone https://github.com/autosae/autosae
cd autosae
uv sync
```

## Before submitting a PR

All of the following must pass:

```bash
uv run pytest                     # all tests
uv run ruff check packages/       # lint
uv run ruff format packages/      # format
uv run mypy packages/core/src packages/server/src  # typecheck
```

If you're touching the UI:

```bash
cd packages/ui
bun run typecheck
```

## High-impact areas

- **Expand the registry** — extract and PR cards for Llama-3.2, Mistral, Phi, etc. See [Extracting cards](#extracting-cards) below.
- **vLLM adapter** — native `ModelRunner` subclass in `packages/server/engine/` for production throughput
- **Dynamic scaling** — context-dependent alpha via cosine similarity (DSAS method)
- **Colab notebook** — zero-setup demo for free-tier GPU
- **New architecture support** — extend `_hooks.py` for any model not yet covered by `get_model_layers()`

## Extracting cards

To contribute a new concept card to the registry:

```python
from autosae import ContrastiveDataset, Extractor

dataset = ContrastiveDataset(
    positive=["...", "..."],  # at least 20 pairs recommended
    negative=["...", "..."],
)

extractor = Extractor("meta-llama/Llama-3.1-8B", layer_frac=0.6, load_in_4bit=True)
card = extractor.extract(dataset, concept="your-concept", default_alpha=1.5, description="...")
card.save("registry/llama-3.1-8b/your-concept.safetensors")
```

Then open a PR with:
- The `.safetensors` file under `registry/{model-slug}/`
- A row added to the registry table in `README.md`
- The extraction script saved under `examples/extract_{concept}.py`

## Code conventions

- `from __future__ import annotations` at the top of every Python file
- No comments — code should be self-documenting
- Type everything; `mypy --strict` must pass
- Functional React components only; no class components
- Python 3.12+ features are fine

## Project structure

```
packages/core/src/autosae/     # Python SDK (pip install autosae)
packages/server/src/autosae_server/  # FastAPI server (pip install autosae-server)
packages/ui/src/               # React dashboard (bun dev)
registry/                      # pre-computed .safetensors concept cards
examples/                      # standalone usage scripts
```
