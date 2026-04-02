from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from autosae.exceptions import UnsupportedArchitectureError


def _as_module_list(obj: Any) -> list[nn.Module]:
    return cast(list[nn.Module], list(obj))


def get_model_layers(model: PreTrainedModel) -> list[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return _as_module_list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return _as_module_list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return _as_module_list(model.gpt_neox.layers)
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return _as_module_list(model.model.decoder.layers)
    arch = getattr(getattr(model, "config", None), "architectures", ["unknown"])
    raise UnsupportedArchitectureError(
        f"Cannot identify transformer layers for {arch[0]}. "
        f"Supported architectures: LLaMA, Mistral, Qwen, Gemma, GPT-2, GPT-NeoX, OPT."
    )


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_idx = attention_mask.sum(dim=1).long() - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx, :]
