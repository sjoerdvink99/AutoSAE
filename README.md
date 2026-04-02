# AutoSAE

[![PyPI](https://img.shields.io/pypi/v/autosae?color=00e676&style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/autosae)
[![Python](https://img.shields.io/badge/python-3.12+-00e676?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-00e676?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/autosae/autosae/ci.yml?style=flat-square&label=CI)](https://github.com/autosae/autosae/actions)

**AutoSAE** extracts human-interpretable **Concept Cards** from any LLM's hidden states and injects them at inference time, permanently shifting the model's cognition without modifying a single token of the prompt.

[**Quickstart**](#quickstart) · [**Dashboard**](#dashboard) · [**Registry**](#pre-computed-registry) · [**How it works**](#how-it-works) · [**Docs**](https://autosae.dev)

---

> _Dragging the **Formality** slider from −2 → +2. No prompt change. Same model. Output shifts from "hey can u reschedule" to "I am writing to formally request a rescheduling of our engagement."_

---

## The problem with prompt engineering

Prompt engineering requires careful system prompts, few-shot examples, and iterative refinement. A single adversarial instruction such as _"ignore previous instructions"_ can nullify this investment entirely.

Prompt engineering operates at the **text layer**. AutoSAE operates at the **math layer**, directly inside the transformer's residual stream, where concepts live as directions in high-dimensional activation space. There is no text to jailbreak.

---

## Quickstart

```bash
pip install "autosae[quantized]"
```

```python
from autosae import Steerer, ConceptCard

steerer = Steerer("meta-llama/Llama-3.1-8B", load_in_4bit=True)
steerer.load_card(ConceptCard.from_registry("formality"), alpha=2.0)

print(steerer.generate("Write an email about tomorrow's meeting."))
```

No system prompt. No few-shot examples. No fine-tuning. The model's activations are shifted at inference time. No weights are changed.

**Same prompt. Different alpha.**

| `alpha` | Output                                                                                        |
| ------- | --------------------------------------------------------------------------------------------- |
| `−2.0`  | _"hey can u reschedule tmrw? thx"_                                                            |
| `0.0`   | _"Can we move tomorrow's meeting? Let me know."_                                              |
| `+2.0`  | _"I am writing to formally request a rescheduling of our engagement scheduled for tomorrow."_ |

---

## Why AutoSAE?

|                              | Prompt engineering | Fine-tuning | **AutoSAE** |
| ---------------------------- | :----------------: | :---------: | :---------: |
| Token overhead per request   |        High        |    None     |  **None**   |
| Jailbreak resistant          |         ✗          |   Partial   |    **✓**    |
| Adjustable at runtime        |         ✗          |      ✗      |    **✓**    |
| Human-interpretable          |      Partial       |      ✗      |    **✓**    |
| Works on any HF transformers model |         ✓          |      ✗      |    **✓**    |
| Requires training data       |         ✗          |      ✓      |    **✗**    |

Activation steering does not instruct the model differently. It changes what concepts are active in its cognition.

---

## Pre-computed registry

AutoSAE ships a **standard library** of Concept Cards pre-extracted for `meta-llama/Llama-3.1-8B` and `Qwen/Qwen2.5-7B`. Each card is auto-downloaded from HuggingFace Hub on first use:

```python
card = ConceptCard.from_registry("safety")
card = ConceptCard.from_registry("reasoning")
card = ConceptCard.from_registry("creativity")
```

| Card          | What it controls                                             |
| ------------- | ------------------------------------------------------------ |
| `formality`   | Formal academic prose ↔ casual conversational text           |
| `safety`      | Safe, cautious responses ↔ harmful, unconstrained output     |
| `reasoning`   | Step-by-step structured logic ↔ direct intuitive answers     |
| `creativity`  | Novel, metaphorical language ↔ literal, factual language     |
| `conciseness` | Terse, minimal responses ↔ verbose, expansive responses      |
| `coding`      | Code-first output ↔ prose-first explanation                  |
| `empathy`     | Warm, emotionally aware tone ↔ clinical, detached tone       |
| `certainty`   | Confident, assertive statements ↔ hedged, uncertain language |

All cards are self-describing `.safetensors` files. Model architecture, target layer, and default alpha are embedded in the header, making them fully portable across deployments.

---

## Mix multiple concepts

Concepts compose. Load any number of Concept Cards; they inject independently into the residual stream:

```python
steerer = Steerer("meta-llama/Llama-3.1-8B", load_in_4bit=True)

steerer.load_card(ConceptCard.from_registry("formality"),   alpha=2.0)
steerer.load_card(ConceptCard.from_registry("conciseness"), alpha=1.5)
steerer.load_card(ConceptCard.from_registry("safety"),      alpha=3.0)

# Alpha updates take effect immediately; no hook re-registration occurs
steerer.set_alpha("formality", 0.8)
steerer.set_alpha("conciseness", -1.0)  # negative alpha reverses the direction
```

Negative alpha pushes the model _against_ a concept. `alpha=-2.0` on `formality` drives output toward maximally casual language.

---

## Extract your own Concept Cards

Any concept describable through contrastive examples is supported, including legal register, domain expertise, emotional tone, and brand voice:

```python
from autosae import ContrastiveDataset, Extractor

dataset = ContrastiveDataset(
    positive=[
        "The defendant contends that the aforementioned clause is null and void.",
        "Pursuant to Article 4(b), the lessor hereby indemnifies the lessee.",
    ],
    negative=[
        "They're saying that part of the contract doesn't count.",
        "Based on Article 4(b), the landlord covers the tenant.",
    ],
)

extractor = Extractor("meta-llama/Llama-3.1-8B", layer_frac=0.6, load_in_4bit=True)
card = extractor.extract(dataset, concept="legalese", default_alpha=1.5)
card.save("./legalese.safetensors")
```

Extraction takes ~2 minutes on a consumer GPU with 30 prompt pairs.

---

## Dashboard

The interactive dashboard allows operators to steer a running model in real time.

```bash
# Terminal 1: inference server
autosae-server --model-id meta-llama/Llama-3.1-8B --load-in-4bit

# Terminal 2: dashboard
cd packages/ui && bun install && bun dev
```

Open `http://localhost:3000`.

**Features:**

- **Steering panel.** One slider per Concept Card with live `PATCH` calls; the Gram-matrix heatmap visualizes concept correlations.
- **Concept space canvas.** A 2D PCA projection of loaded concept vectors; token trajectory is plotted live during generation.
- **Token output.** Tokens are coloured by concept activation magnitude; text selection adjusts alpha directly.

The dashboard uses the same REST and WebSocket API exposed by the server. Replace the bundled frontend or call it from any HTTP client.

---

## How it works

Every transformer layer reads and writes to a **residual stream**, a vector `h ∈ ℝ^d` that accumulates the model's working state. Human-interpretable concepts correspond to linear _directions_ in this space (the linear representation hypothesis).

```
Prompt tokens
      │
      ▼
 Embedding
      │
  ┌───┴────┐
  │ Layer 0 │   h₀ = h + Attn(h) + FFN(h)
  └───┬────┘
      │
  ┌───┴────┐
  │ Layer 1 │
  └───┬────┘
      │
  ┌───┴────────────────────────────────────────────┐
  │ Layer N │  h_N′ = h_N + α · v                  │ ← Concept Card injected
  └───┬────────────────────────────────────────────┘
      │           v = normalize( mean(h_pos) − mean(h_neg) )
  ┌───┴────┐
  │ Layer  │      The model continues with a shifted concept state.
  │  N+1   │      It generates text as if it "naturally" has this trait.
  └───┬────┘
      │
    Output
```

**Extraction (offline, once):**

1. Run contrastive prompt pairs through the model
2. AutoSAE scores every layer using the Fisher discriminant and picks the layer with the strongest class separation (or you can specify `layer_frac` manually)
3. Pool the last-token hidden state per prompt, compute `v = normalize(mean(h_pos) − mean(h_neg))`
4. Save as a `.safetensors` Concept Card

**Injection (every forward pass, zero overhead):**

1. Register a `forward_hook` on the target layer
2. For each token position: `h′ = h + α × v`
3. The hook runs approximately 500 µs per token at float16, negligible relative to attention computation.

This is grounded in [Representation Engineering (Zou et al., 2023)](https://arxiv.org/abs/2310.01405) and [Activation Addition (Turner et al., 2023)](https://arxiv.org/abs/2308.10248).

---

## Installation

```bash
# Core SDK only
pip install autosae

# With 4-bit / 8-bit quantization (runs 8B models on ≤12 GB VRAM)
pip install "autosae[quantized]"

# Inference server
pip install autosae-server
```

**Requirements:** Python 3.12+, PyTorch 2.3+. CPU works for extraction; a GPU (CUDA or MPS) is recommended for generation.

---

## Supported models

Anything loadable by `transformers.AutoModelForCausalLM` with a standard residual-stream layout:

| Family            | Models                                                     |
| ----------------- | ---------------------------------------------------------- |
| LLaMA / Llama-3   | `meta-llama/Llama-3.1-8B`, `Llama-3.1-70B`, `Llama-3.2-3B` |
| Mistral / Mixtral | `mistralai/Mistral-7B-v0.3`, `Mixtral-8x7B-Instruct`       |
| Qwen              | `Qwen/Qwen2.5-7B`, `Qwen2.5-72B-Instruct`                  |
| Gemma             | `google/gemma-2-9b`, `gemma-2-27b`                         |
| GPT-2 / GPT-NeoX  | `gpt2-xl`, `EleutherAI/gpt-neox-20b`                       |
| Phi               | `microsoft/phi-3-mini-4k-instruct`, `phi-3.5-mini`         |

---

## API reference

```python
# Steerer
steerer = Steerer(model_id, device="auto", load_in_4bit=False, load_in_8bit=False)
steerer.load_card(card, alpha=None)         # alpha defaults to card.meta.default_alpha
steerer.unload_card("formality")
steerer.set_alpha("formality", 2.5)         # O(1) live update, no hook re-registration
steerer.loaded_concepts()                   # → {"formality": 2.5, "safety": 1.0}
steerer.generate(prompt, max_new_tokens=512, temperature=1.0)
steerer.generate_stream(prompt)             # → Iterator[(token, {concept: score}, (x, y) | None)]
steerer.combine_cards(name, concepts=[...]) # merge loaded cards into one exportable card
steerer.unload()                            # free GPU memory

# ConceptCard
ConceptCard.from_registry("formality", model="llama-3.1-8b")
ConceptCard.load("./card.safetensors")
card.save("./card.safetensors")
card.meta   # → ConceptCardMeta(model_id, layer, hidden_dim, default_alpha, concept, description)

# Extractor
extractor = Extractor(model_id, layer_frac=0.6, load_in_4bit=False)
card = extractor.extract(dataset, concept="legalese", default_alpha=1.5, description="...")
extractor.unload()

# ContrastiveDataset
ContrastiveDataset(positive=[...], negative=[...])
ContrastiveDataset.from_json("dataset.json")   # {"positive": [...], "negative": [...]}
```

---

## Server REST + WebSocket API

```
GET    /health
GET    /cards                      list loaded cards + current alphas
POST   /cards/load                 load from registry name or local path
DELETE /cards/{concept}            unload
PATCH  /cards/{concept}/alpha      update alpha live
GET    /cards/{concept}/download   download card as .safetensors
POST   /cards/extract              extract a new card from contrastive prompts
POST   /cards/combine              merge loaded cards into one
POST   /cards/layer-sweep          score all layers by Fisher discriminant

GET    /geometry                   concept space (PCA axes, Gram matrix, 2D vectors)
POST   /geometry/inverse           map a canvas click → per-concept alpha deltas

GET    /hub/search                 search HuggingFace Hub for concept cards
POST   /hub/push                   publish a loaded card to HF Hub
POST   /hub/download               load a card directly from HF Hub

POST   /generate                   blocking generation → {text, activations}
WS     /ws/generate                streaming: {type, token, activations, projection} per token
```

---

## Repository layout

```
autosae/
├── packages/
│   ├── core/                   # pip install autosae
│   │   └── src/autosae/
│   │       ├── concept_card.py  # ConceptCard + serialization
│   │       ├── dataset.py       # ContrastiveDataset
│   │       ├── extractor.py     # forward-hook activation capture
│   │       ├── steerer.py       # injection hooks + KV-cache generation loop
│   │       └── _hooks.py        # architecture-agnostic layer discovery
│   │
│   ├── server/                 # pip install autosae-server
│   │   └── src/autosae_server/
│   │       ├── app.py           # FastAPI factory
│   │       ├── engine/          # InferenceEngine ABC + TransformersEngine
│   │       ├── routes/          # /health, /cards, /generate, WS
│   │       └── main.py          # CLI entrypoint
│   │
│   └── ui/                     # bun dev
│       └── src/
│           ├── components/
│           │   ├── steering/    # SteeringPanel, ConceptSlider, GramMatrixHeatmap
│           │   ├── canvas/      # CanvasPanel, ConceptSpaceCanvas, MonitorStrip
│           │   └── generation/  # GenerationPane, TokenizedOutput, DiffView
│           ├── hooks/           # useGeneration (WebSocket), useGeometry
│           └── stores/          # Zustand state
│
├── registry/                   # pre-computed .safetensors (auto-downloaded from HF Hub)
│   ├── llama-3.1-8b/           # ConceptCard.from_registry() caches here after first use
│   └── qwen2.5-7b/
│
└── examples/
    ├── quickstart.py
    ├── extract_custom.py
    ├── demo_gpt2.py
    └── demo_multi_concept.py
```

---

## Contributing

```bash
git clone https://github.com/autosae/autosae
cd autosae && uv sync
uv run pytest                # all tests, <1s
uv run ruff check .          # lint
uv run ruff format .         # format
```

**High-impact areas:**

- **Expand the registry.** Extract and submit cards for Llama-3.2, Mistral, Phi, and other architectures.
- **vLLM adapter.** A native `ModelRunner` subclass in `packages/server/engine/` for production-grade throughput.
- **Dynamic scaling.** Context-dependent alpha adjustment via cosine similarity (DSAS method).
- **Colab notebook.** A zero-setup demo targeting free-tier GPU runtimes.
- **New architecture support.** Extend `_hooks.py` to cover architectures not yet listed.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## Research

AutoSAE implements the **contrastive mean difference** vector extraction method (CAA) from:

| Paper                                                                                                           | Contribution                                    |
| --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| [Representation Engineering](https://arxiv.org/abs/2310.01405) (Zou et al., 2023)                               | Contrastive activation addition framework       |
| [Activation Addition](https://arxiv.org/abs/2308.10248) (Turner et al., 2023)                                   | Steering vectors in residual stream             |
| [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (Elhage et al., 2022) | Linear representation hypothesis                |
| [Scaling and evaluating SAEs](https://arxiv.org/abs/2406.04093) (Gao et al., 2024)                              | Sparse autoencoder interpretability (Anthropic) |

---

MIT License · Built with PyTorch, FastAPI, React 19

**If AutoSAE is useful to your work, a GitHub star is appreciated.**
