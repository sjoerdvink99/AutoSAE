from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from autosae._hooks import get_model_layers
from autosae.concept_card import ConceptCard
from autosae.exceptions import ConceptNotLoadedError, IncompatibleCardError
from autosae.geometry import ConceptSpace

logger = logging.getLogger(__name__)


class Steerer:
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._cards: dict[str, tuple[ConceptCard, float]] = {}
        self._cards_lock = threading.Lock()
        self._hooks: list[RemovableHandle] = []
        self._hooks_lock = threading.Lock()
        self._hooks_dirty: bool = True
        self._activation_store: dict[str, float] = {}
        self._activation_lock = threading.Lock()
        self._last_hidden: torch.Tensor | None = None
        self._baselines: dict[int, torch.Tensor] = {}
        self._steer_prompt: bool = True

    def _ensure_loaded(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if self._model is None or self._tokenizer is None:
            cuda_available = torch.cuda.is_available()
            mps_available = torch.backends.mps.is_available()

            use_4bit = self.load_in_4bit and cuda_available
            use_8bit = self.load_in_8bit and cuda_available

            quant_config = None
            if use_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            elif use_8bit:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)

            device_map = self.device
            if device_map == "auto" and not cuda_available and mps_available:
                device_map = "mps"

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map=device_map,
                torch_dtype=torch.float16 if not (use_4bit or use_8bit) else None,
            )
            self._model.eval()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._model, self._tokenizer

    def load_card(self, card: ConceptCard, alpha: float | None = None) -> None:
        if self._model is not None:
            self.validate_card_compatibility(card, self._model)
        if card.meta.model_id != self.model_id:
            logger.warning(
                "Card '%s' was extracted from '%s' but the steerer is running '%s'. "
                "Steering direction may not transfer across model variants.",
                card.meta.concept,
                card.meta.model_id,
                self.model_id,
            )
        with self._cards_lock:
            self._cards[card.meta.concept] = (
                card,
                alpha if alpha is not None else card.meta.default_alpha,
            )
        self._hooks_dirty = True

    def unload_card(self, concept: str) -> None:
        with self._cards_lock:
            if concept not in self._cards:
                raise ConceptNotLoadedError(f"Concept '{concept}' is not loaded.")
            del self._cards[concept]
        self._hooks_dirty = True

    def set_alpha(self, concept: str, alpha: float) -> None:
        with self._cards_lock:
            if concept not in self._cards:
                raise ConceptNotLoadedError(f"Concept '{concept}' is not loaded.")
            card, _ = self._cards[concept]
            self._cards[concept] = (card, alpha)

    def loaded_concepts(self) -> dict[str, float]:
        with self._cards_lock:
            return {concept: alpha for concept, (_, alpha) in self._cards.items()}

    def loaded_cards(self) -> dict[str, tuple[ConceptCard, float]]:
        with self._cards_lock:
            return dict(self._cards)

    def concept_space(self) -> ConceptSpace | None:
        with self._cards_lock:
            if not self._cards:
                return None
            cards_snapshot = dict(self._cards)
        return ConceptSpace(cards_snapshot)

    @staticmethod
    def _format_prompt(
        tokenizer: PreTrainedTokenizerBase, prompt: str, system_prompt: str | None = None
    ) -> tuple[str, bool]:
        if getattr(tokenizer, "chat_template", None) is not None:
            messages: list[dict[str, str]] = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            formatted = cast(str, tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            return formatted, True
        return prompt, False

    @staticmethod
    def _format_messages(
        tokenizer: PreTrainedTokenizerBase,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> tuple[str, bool]:
        if getattr(tokenizer, "chat_template", None) is not None:
            msgs = list(messages)
            if system_prompt is not None and (not msgs or msgs[0].get("role") != "system"):
                msgs = [{"role": "system", "content": system_prompt}] + msgs
            return cast(str, tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)), True
        return "\n".join(f"{m['content']}" for m in messages), False

    @staticmethod
    def validate_card_compatibility(card: ConceptCard, model: PreTrainedModel) -> None:
        expected_dim = getattr(getattr(model, "config", None), "hidden_size", None)
        if expected_dim is not None and card.meta.hidden_dim != expected_dim:
            raise IncompatibleCardError(
                f"Card '{card.meta.concept}' has hidden_dim={card.meta.hidden_dim} "
                f"but model expects {expected_dim}."
            )
        layers = get_model_layers(model)
        if card.meta.layer >= len(layers):
            raise IncompatibleCardError(
                f"Card '{card.meta.concept}' targets layer={card.meta.layer} "
                f"but model only has {len(layers)} layers."
            )

    @staticmethod
    def _get_last_token(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx, :]
        return hidden[:, -1, :]

    @torch.no_grad()
    def calibrate_baseline(self, corpus: list[str], batch_size: int = 8) -> None:
        model, tokenizer = self._ensure_loaded()
        with self._cards_lock:
            if not self._cards:
                return
            target_layers = {card.meta.layer for card, _ in self._cards.values()}

        layers = get_model_layers(model)
        per_batch: dict[int, list[torch.Tensor]] = {li: [] for li in target_layers}
        accumulated: dict[int, list[torch.Tensor]] = {li: [] for li in target_layers}
        handles: list[RemovableHandle] = []

        def _make_capture(
            layer_idx: int,
        ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], Any], None]:
            def _hook(module: nn.Module, input: tuple[torch.Tensor, ...], output: Any) -> None:
                hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
                per_batch[layer_idx].append(hidden.detach().float().cpu())

            return _hook

        for li in target_layers:
            handles.append(layers[li].register_forward_hook(_make_capture(li)))

        try:
            for i in range(0, len(corpus), batch_size):
                batch = corpus[i : i + batch_size]
                for li in target_layers:
                    per_batch[li].clear()
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)
                model(input_ids=input_ids, attention_mask=attention_mask)
                for li in target_layers:
                    if per_batch[li]:
                        last_tokens = self._get_last_token(per_batch[li][0], attention_mask.cpu())
                        accumulated[li].append(last_tokens)
        finally:
            for handle in handles:
                handle.remove()

        self._baselines = {
            li: torch.cat(accumulated[li], dim=0).mean(0) for li in target_layers if accumulated[li]
        }

    def _register_hooks(self) -> None:
        with self._hooks_lock:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()

            with self._cards_lock:
                cards_snapshot = dict(self._cards)

            if not cards_snapshot or self._model is None:
                return

            layers = get_model_layers(self._model)
            by_layer: dict[int, list[str]] = {}
            for concept, (card, _) in cards_snapshot.items():
                by_layer.setdefault(card.meta.layer, []).append(concept)

            for layer_idx, concepts in by_layer.items():
                hook_fn = self._make_hook(concepts, layer_idx)
                handle = layers[layer_idx].register_forward_hook(hook_fn)
                self._hooks.append(handle)

    def _make_hook(self, concepts: list[str], layer_idx: int) -> Callable[..., Any]:
        def hook(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
            output: Any,
        ) -> Any:
            is_tuple = isinstance(output, tuple)
            hidden: torch.Tensor = output[0] if is_tuple else output

            with self._cards_lock:
                cards_snapshot = {c: self._cards[c] for c in concepts if c in self._cards}

            pre_last = hidden[:, -1, :].float()
            baseline = self._baselines.get(layer_idx)
            if baseline is not None:
                pre_last = pre_last - baseline.to(pre_last.device)

            for concept, (card, _) in cards_snapshot.items():
                v = card.vector.to(hidden.device, hidden.dtype).float()
                sim = F.cosine_similarity(pre_last, v.unsqueeze(0), dim=-1).mean().item()
                with self._activation_lock:
                    self._activation_store[concept] = float(sim)

            delta = torch.zeros_like(hidden)
            for _concept, (card, alpha) in cards_snapshot.items():
                v = card.vector.to(hidden.device, hidden.dtype)
                delta = delta + alpha * v.view(1, 1, -1)

            steered = hidden + delta if self._steer_prompt or hidden.shape[1] == 1 else hidden

            self._last_hidden = steered[:, -1, :].detach().float().cpu()

            if is_tuple:
                return (steered,) + output[1:]
            return steered

        return hook

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int | None = None,
        greedy: bool = False,
        steer_prompt: bool = True,
        messages: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        repetition_penalty: float = 1.0,
    ) -> str:
        model, tokenizer = self._ensure_loaded()
        with self._activation_lock:
            self._activation_store.clear()
        if self._hooks_dirty:
            self._register_hooks()
            self._hooks_dirty = False
        self._steer_prompt = steer_prompt
        if seed is not None:
            torch.manual_seed(seed)

        formatted, used_template = (
            self._format_messages(tokenizer, messages, system_prompt=system_prompt)
            if messages is not None
            else self._format_prompt(tokenizer, prompt, system_prompt=system_prompt)
        )
        encoding = tokenizer(formatted, return_tensors="pt", add_special_tokens=not used_template)
        input_ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)
        prompt_len = input_ids.shape[1]

        do_sample = not greedy
        generate_fn: Any = model.generate
        output_ids = cast(
            torch.Tensor,
            generate_fn(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            ),
        )

        generated = output_ids[0, prompt_len:]
        return cast(str, tokenizer.decode(generated, skip_special_tokens=True))

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int | None = None,
        greedy: bool = False,
        steer_prompt: bool = True,
        messages: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        repetition_penalty: float = 1.0,
    ) -> Iterator[tuple[str, dict[str, float], tuple[float, float] | None]]:
        model, tokenizer = self._ensure_loaded()
        with self._activation_lock:
            self._activation_store.clear()
        self._last_hidden = None
        self._steer_prompt = steer_prompt
        if self._hooks_dirty:
            self._register_hooks()
            self._hooks_dirty = False

        space = self.concept_space()

        gen: torch.Generator | None = None
        if seed is not None:
            try:
                gen = torch.Generator(device=model.device)
            except RuntimeError:
                gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)

        formatted, used_template = (
            self._format_messages(tokenizer, messages, system_prompt=system_prompt)
            if messages is not None
            else self._format_prompt(tokenizer, prompt, system_prompt=system_prompt)
        )
        encoding = tokenizer(formatted, return_tensors="pt", add_special_tokens=not used_template)
        input_ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)
        past_kv: Any = None
        eos_ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            eos_ids.add(tokenizer.eos_token_id)
        gen_config_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if isinstance(gen_config_eos, int):
            eos_ids.add(gen_config_eos)
        elif isinstance(gen_config_eos, list):
            eos_ids.update(x for x in gen_config_eos if isinstance(x, int))
        added_tokens = getattr(tokenizer, "added_tokens_encoder", {})
        for token_str in ("<|eot_id|>", "<|end|>", "<|im_end|>"):
            tid = added_tokens.get(token_str)
            if isinstance(tid, int):
                eos_ids.add(tid)

        do_sample = not greedy

        try:
            for _ in range(max_new_tokens):
                current_ids = input_ids if past_kv is None else input_ids[:, -1:]
                outputs = model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = outputs.past_key_values

                logits = outputs.logits[:, -1, :]
                if repetition_penalty != 1.0:
                    for token_id in set(input_ids[0].tolist()):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty
                if temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cumulative = sorted_probs.cumsum(dim=-1)
                    mask = (cumulative - sorted_probs) >= top_p
                    sorted_probs[mask] = 0.0
                    probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                next_id = (
                    torch.multinomial(probs, num_samples=1, generator=gen)
                    if do_sample
                    else logits.argmax(-1, keepdim=True)
                )

                input_ids = torch.cat([input_ids, next_id], dim=-1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                if next_id[0, 0].item() in eos_ids:
                    break

                token = cast(str, tokenizer.decode(next_id[0], skip_special_tokens=True))
                with self._activation_lock:
                    activations = dict(self._activation_store)

                projection: tuple[float, float] | None = None
                if space is not None and self._last_hidden is not None:
                    projection = space.project(self._last_hidden[0])

                yield token, activations, projection
        finally:
            self._last_hidden = None
            with self._activation_lock:
                self._activation_store.clear()

    def concept_directions(self) -> dict[str, torch.Tensor]:
        with self._cards_lock:
            return {concept: card.vector for concept, (card, _) in self._cards.items()}

    def unload(self) -> None:
        with self._hooks_lock:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()
