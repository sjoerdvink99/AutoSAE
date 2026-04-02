# AutoSAE

**Steer any LLM's cognition in real-time — no prompt engineering required.**

[![PyPI](https://img.shields.io/pypi/v/autosae?color=00e676&style=flat-square)](https://pypi.org/project/autosae)
[![License](https://img.shields.io/badge/license-MIT-00e676?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/autosae/autosae/ci.yml?style=flat-square)](https://github.com/autosae/autosae/actions)

> *AutoSAE extracts human-interpretable concept vectors from LLM hidden states and injects them at inference time — shifting model behaviour without touching a single token of your prompt.*

<!-- GIF demo here -->

## The 3-line quickstart

```python
from autosae import Steerer, ConceptCard

steerer = Steerer("meta-llama/Llama-3.1-8B", load_in_4bit=True)
steerer.load_card(ConceptCard.from_registry("formality"), alpha=2.0)
print(steerer.generate("Summarize our Q3 earnings results."))
```

That's it. No system prompts, no few-shot examples. The model's internal representations are directly altered.

## Why AutoSAE?

| Approach | Token overhead | Jailbreak resistant | Runtime adjustable | Human-interpretable |
|---|:---:|:---:|:---:|:---:|
| Prompt engineering | High | No | No | Partial |
| Fine-tuning | None | Partial | No | No |
| **AutoSAE** | **None** | **Yes** | **Yes** | **Yes** |

Activation steering bypasses the instruction-following layer entirely — it operates at the mathematical level of the model's residual stream, making it robust against adversarial prompts that circumvent text-based guardrails.

## How it works

```
Input prompt
     │
     ▼
┌────────────┐
│  Layer 0   │
├────────────┤
│  Layer 1   │
├────────────┤       ┌────────────────────┐
│  Layer N   │◄──────│  + alpha × v       │  ← Concept Card injected here
├────────────┤       └────────────────────┘
│  ...       │         v = normalize( mean(h_pos) − mean(h_neg) )
└────────────┘
     │
     ▼
  Output
```

A **Concept Card** is a unit-norm vector `v ∈ ℝ^d` that encodes the direction in activation space between two contrastive sets of prompts (e.g. "formal" vs "casual"). At inference time, `alpha × v` is added to every position in the residual stream at the target layer.

## Pre-computed registry

AutoSAE ships with a standard library of cards for `meta-llama/Llama-3.1-8B` and `Qwen/Qwen2.5-7B`:

| Concept | Direction |
|---|---|
| `formality` | Formal ↔ casual register |
| `safety` | Safe ↔ harmful intent |
| `reasoning` | Structured ↔ intuitive |
| `creativity` | Creative ↔ literal |
| `conciseness` | Terse ↔ verbose |
| `coding` | Code-focused ↔ prose-focused |
| `empathy` | Empathetic ↔ detached |
| `certainty` | Confident ↔ hedging |

```python
card = ConceptCard.from_registry("safety", model="llama-3.1-8b")
```

## Extract your own concepts

```python
from autosae import ContrastiveDataset, Extractor

dataset = ContrastiveDataset(
    positive=["Pursuant to section 4.2...", "I am writing to formally..."],
    negative=["Hey so like...", "Can u help me with..."],
)

extractor = Extractor("meta-llama/Llama-3.1-8B", layer_frac=0.6)
card = extractor.extract(dataset, concept="formality")
card.save("./formality.safetensors")
```

Concept Cards are self-describing `.safetensors` files — architecture, target layer, and metadata are embedded in the file header.

## Stack multiple concepts

```python
steerer = Steerer("meta-llama/Llama-3.1-8B")
steerer.load_card(ConceptCard.from_registry("formality"), alpha=2.0)
steerer.load_card(ConceptCard.from_registry("conciseness"), alpha=1.5)
steerer.load_card(ConceptCard.from_registry("safety"), alpha=3.0)

# Adjust in real-time without reloading
steerer.set_alpha("formality", 0.5)
```

## Interactive Dashboard

The HITL dashboard runs alongside the inference server. Load concept cards, drag sliders to adjust `alpha` in real-time, and watch the model's output morph as you steer.

```bash
# Terminal 1 — start inference server
autosae-server --model-id meta-llama/Llama-3.1-8B --load-in-4bit

# Terminal 2 — start dashboard
cd packages/ui && bun dev
```

Open `http://localhost:3000`.

Features:
- **Mixing board** — one slider per loaded concept, alpha range −3 → +3
- **Streaming output** — tokens with high concept activation are highlighted
- **Activation monitor** — real-time sparklines showing concept magnitude per token

## Installation

```bash
pip install autosae

# For 4-bit quantization support (recommended for ≤24 GB VRAM)
pip install "autosae[quantized]"

# For the inference server
pip install autosae-server
```

## Architecture

```
autosae/
├── packages/
│   ├── core/         # Python SDK — ConceptCard, Extractor, Steerer
│   ├── server/       # FastAPI + WebSocket inference server
│   └── ui/           # React 19 dashboard
└── registry/         # Pre-computed concept cards (.safetensors)
```

## Contributing

```bash
git clone https://github.com/autosae/autosae
uv sync
uv run pytest
```

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
