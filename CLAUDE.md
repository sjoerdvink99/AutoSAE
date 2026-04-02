# AutoSAE — Agent Instructions

## What this project is

AutoSAE is an open-source developer toolkit for **activation steering** of large language models. The core idea: instead of changing what you say to an LLM via prompt engineering, you directly change what mathematical concepts are active in the model's hidden states. This is done by:

1. **Extracting** a concept vector `v ∈ ℝ^d` from contrastive prompt pairs (e.g. formal vs casual text) using PyTorch forward hooks on the residual stream.
2. **Packaging** that vector as a self-describing `.safetensors` file called a **Concept Card**.
3. **Injecting** it at inference time: `h′ = h + α·v` at the target transformer layer.

The goal is to get thousands of GitHub stars by being the definitive, production-ready Python library for this technique. Every decision should optimise for: **developer experience**, **correctness**, and **clean, readable code**.

## Repository layout

```
autosae/
├── packages/
│   ├── core/            # pip install autosae  (Python SDK)
│   ├── server/          # pip install autosae-server  (FastAPI inference server)
│   └── ui/              # bun dev  (React 19 dashboard)
├── registry/            # pre-computed .safetensors concept cards
├── examples/
├── .github/workflows/   # CI (ruff, mypy, pytest) + publish on tag
└── pyproject.toml       # uv workspace root
```

## Package responsibilities

### `packages/core` → `autosae`

The public-facing Python SDK. This is what most users will `pip install`.

| File | Purpose |
|---|---|
| `concept_card.py` | `ConceptCard` + `ConceptCardMeta` — the portable `.safetensors` format with embedded metadata. Handles save/load/registry download/push. |
| `dataset.py` | `ContrastiveDataset` — paired positive/negative prompt lists with JSON I/O. |
| `extractor.py` | `Extractor` — registers `forward_hook` on residual stream, pools hidden states (default: last-token), computes `v = norm(mean_pos - mean_neg)`. With `auto_layer=True` (default) selects the best layer via Fisher discriminant + permutation test. Supports 4-bit/8-bit via bitsandbytes. |
| `steerer.py` | `Steerer` — loads concept cards, registers injection hooks, exposes `set_alpha()` for live updates. `generate_stream()` is a manual KV-cache loop that yields 3-tuples `(token, activations, projection)`. `combine_cards()` merges loaded cards into a single exportable card. |
| `geometry.py` | `ConceptSpace` — owns all geometry math: orthogonalisation (modified Gram-Schmidt), SVD/PCA, Gram matrix, forward `project()`, constrained `inverse_project()`. |
| `_hooks.py` | Architecture-agnostic `get_model_layers()`, `mean_pool()`, `last_token_pool()`. Supports LLaMA, Mistral, Qwen, Gemma, GPT-2, GPT-NeoX, OPT. |
| `exceptions.py` | Typed exception hierarchy: `AutoSAEError`, `UnsupportedArchitectureError`, `ConceptCardNotFoundError`, `ConceptNotLoadedError`. |

**Key invariants:**
- `Steerer._cards` is a `dict[str, tuple[ConceptCard, float]]`. Hooks are registered once per unique target layer. The hook reads alpha via **fresh dict lookup** on every forward pass — this is what makes `set_alpha()` O(1) without re-registering hooks.
- The hook computes the **pre-steer cosine similarity** of the last-token hidden state (optionally baseline-subtracted via `_baselines: dict[int, Tensor]`) against each concept vector, stores it in `_activation_store`, then injects `delta = Σ alpha_i * v_i` into all token positions.
- After steering, the hook saves `_last_hidden` (the steered last-token hidden state). `generate_stream()` reads it after each token to call `ConceptSpace.project()` and yield the 2D canvas coordinate as `projection: tuple[float, float] | None`.

### `packages/server` → `autosae-server`

FastAPI app with CORS, served via `uvicorn`. Key routes:
- `GET /health` — version check
- `GET /cards` / `POST /cards/load` / `DELETE /cards/{concept}` / `PATCH /cards/{concept}/alpha`
- `POST /cards/extract` — extract a new card; `POST /cards/combine` — merge loaded cards; `POST /cards/layer-sweep`
- `GET /geometry` — concept space (204 when < 1 card loaded); `POST /geometry/inverse` — canvas click → alpha deltas
- `GET /hub/search` / `POST /hub/push` / `POST /hub/download`
- `POST /generate` — blocking; `WS /ws/generate` — streaming: `{type: "token", token, activations, projection}` per token, then `{type: "done"}`

`TransformersEngine` bridges sync PyTorch inference into async FastAPI via a single-worker `ThreadPoolExecutor` + `asyncio.Queue`. The generation loop yields `GenerationChunk(token, activations, projection)` objects that the WS route serialises into the token messages above.

### `packages/ui` — React dashboard

- **Zustand** for client state (cards, tokens, isGenerating, canvasViewport, trajectory)
- **TanStack Query** for REST mutations (load card, update alpha, geometry)
- **WebSocket** managed in `useGeneration` hook — opens per-generation, closes on done/error
- **D3** for math only (scales, line generators, drag); DOM rendered via JSX
- Dark neon aesthetic: `#080808` bg, `#00e676` accent

## Commands

```bash
# Python
uv sync                           # install all workspace packages
uv run pytest                     # run all tests (102 tests)
uv run ruff check packages/       # lint
uv run ruff format packages/      # format
uv run mypy packages/core/src packages/server/src  # typecheck

# Inference server
autosae-server --model-id meta-llama/Llama-3.1-8B --load-in-4bit --port 8000

# UI
cd packages/ui
bun install
bun dev           # dev server on :3000, proxies /api and /ws to :8000
bun run build     # production build
bun run typecheck # tsc --noEmit
```

## Code conventions

- **No comments in code.** Code should be self-documenting.
- `from __future__ import annotations` at the top of every Python file.
- Type everything. `mypy --strict` must pass.
- `ruff` for lint + format. Config in root `pyproject.toml`.
- No `TCH` rules (too aggressive for runtime imports). `B008` suppressed (FastAPI `Depends` pattern).
- Python 3.12+ features are fine (`datetime.UTC`, etc.).
- React components: functional only, no class components. Hooks for state.
- No comments in TypeScript either. Types tell the story.

## What to prioritise

**Do:**
- Keep the public Python API minimal and ergonomic — `Steerer`, `Extractor`, `ConceptCard`, `ContrastiveDataset`
- Ensure `uv run pytest` always passes before committing
- Maintain the `from_registry()` caching behaviour — cards are cached in `_REGISTRY_CACHE` after first load
- Keep hook registration logic in `steerer.py:_register_hooks()` — it's the hot path

**Don't:**
- Add `bitsandbytes` to the base `autosae` dependencies — it's in `[quantized]` optional extras
- Re-register hooks on `set_alpha()` — the hook looks up alpha from `self._cards` on every forward pass
- Import `autosae` inside `autosae_server` at module level beyond what's needed — keep startup fast
- Break the `.safetensors` metadata schema — cards in the registry depend on `ConceptCardMeta` being stable

## Testing approach

Tests are in `packages/core/tests/` (90 tests) and `packages/server/tests/` (12 tests). All tests use mocks — no real models are loaded.

- `test_concept_card.py` — save/load roundtrip, shape validation, repr
- `test_dataset.py` — construction, validation, JSON I/O
- `test_extractor.py` — mocked model + tokenizer; verifies vector shape and unit norm
- `test_steerer.py` — alpha CRUD, hook injection math verified with zero-input tensors
- `test_geometry.py` — Gram diagonal=1, symmetry, inverse round-trip math invariants
- `test_app.py` — server routes; generate mock must accept `*args, **kwargs` (route passes `seed=` kwarg)

For the extractor test: the mock tokenizer uses `.side_effect` (not `.__call__` assignment) and returns real `torch.Tensor` dicts so `.to(device)` works correctly.

## Registry

Pre-computed concept cards live in `registry/{model-slug}/{concept}.safetensors`. The same cards are mirrored on HuggingFace Hub at `sjoerdvink/autosae` for `ConceptCard.from_registry()` to download automatically.

When adding a new registry card: extract with `Extractor`, save with `card.save("registry/{model}/{concept}.safetensors")`, commit, and sync to HF Hub.

## Architecture decisions

**Why `.safetensors` over pickle/npz?** Safe (no arbitrary code execution), fast memory-mapped loading, and supports string metadata in the header — perfect for embedding `ConceptCardMeta` as a JSON string without a sidecar file.

**Why manual KV-cache loop in `Steerer.generate_stream()` instead of `model.generate()`?** `model.generate()` doesn't let us read per-token hook data synchronously between tokens. The manual loop gives us `(token, activation_snapshot)` pairs without threading complexity.

**Why `ThreadPoolExecutor` in `TransformersEngine`?** PyTorch is sync. FastAPI is async. One dedicated thread runs the generation loop; `asyncio.Queue` bridges results back to the async WebSocket handler without blocking the event loop.

**Why Zustand over Redux/Context?** Minimal boilerplate, no provider hell, typed selectors. The store is small and the mutations are simple.
