"""Opt-in Qwen2.5 decode adapter for dynamic INT8 KV-cache experiments.

This module intentionally targets the eager Qwen2 API from the Transformers
versions pinned by this project. It is a correctness prototype, not a general
attention backend: prefill and unsupported calls remain on the original path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Callable

import torch

from .attention import paged_attention_dynamic_int8
from .config import QuantConfig
from .paged_cache import PagedKVCache


@dataclass
class QwenDynamicKVAdapter:
    """Patch Qwen2.5 self-attention modules only for batch-one decode."""

    config: QuantConfig = field(default_factory=QuantConfig)
    enabled: bool = True
    disabled_layers: set[int] = field(default_factory=set)
    _original_forwards: dict[int, Callable[..., Any]] = field(default_factory=dict, init=False)

    def install(self, model: Any) -> None:
        """Install wrappers after checking the pinned Qwen2 module contract."""
        try:
            from transformers.models.qwen2 import modeling_qwen2
        except ImportError as error:  # pragma: no cover - requires optional dependency
            raise RuntimeError("install transformers with Qwen2 support before installing the adapter") from error
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("expected a Qwen2ForCausalLM-style model with model.layers")
        if not hasattr(modeling_qwen2, "apply_rotary_pos_emb"):
            raise RuntimeError("unsupported Transformers Qwen2 API: apply_rotary_pos_emb is missing")
        for index, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            required = ("q_proj", "k_proj", "v_proj", "o_proj", "config", "head_dim", "layer_idx")
            if attention is None or any(not hasattr(attention, name) for name in required):
                raise RuntimeError(f"unsupported Qwen2 attention contract in layer {index}")
            if index in self._original_forwards:
                continue
            original = attention.forward
            self._original_forwards[index] = original
            attention.forward = MethodType(self._make_forward(index, attention, original), attention)

    def uninstall(self, model: Any) -> None:
        """Restore original module methods."""
        layers = getattr(getattr(model, "model", None), "layers", [])
        for index, original in self._original_forwards.items():
            layers[index].self_attn.forward = original
        self._original_forwards.clear()

    def set_layer_enabled(self, layer_index: int, enabled: bool) -> None:
        """Enable or disable quantized decode for one layer."""
        if enabled:
            self.disabled_layers.discard(layer_index)
        else:
            self.disabled_layers.add(layer_index)

    def _make_forward(self, index: int, module: Any, original: Callable[..., Any]):
        adapter = self

        def forward(
            _self: Any,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            attention_mask: torch.Tensor | None = None,
            past_key_value: Any | None = None,
            cache_position: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> Any:
            if (
                not adapter.enabled
                or index in adapter.disabled_layers
                or hidden_states.shape[0] != 1
                or hidden_states.shape[1] != 1
                or past_key_value is None
                or position_embeddings is None
            ):
                return original(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    **kwargs,
                )
            # Qwen2 creates a 4D causal mask even for a non-padded one-token
            # decode and includes one future cache position. The dynamic path
            # gathers only ``cache.update``'s returned valid tokens, so that
            # extra future column must not be interpreted as user padding.
            return adapter._dynamic_decode(module, hidden_states, position_embeddings, past_key_value, cache_position)

        return forward

    def _dynamic_decode(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: Any,
        cache_position: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        """Execute the documented eager Qwen2 decode contract for batch one."""
        try:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Transformers Qwen2 eager implementation is unavailable") from error
        batch, _, _ = hidden_states.shape
        heads = module.config.num_attention_heads
        kv_heads = module.config.num_key_value_heads
        dim = module.head_dim
        query = module.q_proj(hidden_states).view(batch, 1, heads, dim).transpose(1, 2)
        key = module.k_proj(hidden_states).view(batch, 1, kv_heads, dim).transpose(1, 2)
        value = module.v_proj(hidden_states).view(batch, 1, kv_heads, dim).transpose(1, 2)
        cosine, sine = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cosine, sine)
        cache_kwargs = {"sin": sine, "cos": cosine, "cache_position": cache_position}
        key, value = cache.update(key, value, module.layer_idx, cache_kwargs)
        if key.shape[0] != 1 or key.shape[2] < 1:
            raise RuntimeError("Qwen cache update did not return [1, kv_heads, tokens, head_dim]")
        tokens = key.shape[2]
        pages = (tokens + self.config.block_size - 1) // self.config.block_size
        paged = PagedKVCache.empty(pages, self.config.block_size, kv_heads, dim, dtype=key.dtype, device=key.device)
        slots = torch.arange(tokens, device=key.device)
        paged.write(key[0].transpose(0, 1), value[0].transpose(0, 1), slots)
        table = torch.arange(pages, device=key.device, dtype=torch.long).unsqueeze(0)
        output, _ = paged_attention_dynamic_int8(
            query[:, :, 0, :], paged, table, torch.tensor([tokens], device=key.device), self.config
        )
        return module.o_proj(output[:, None].reshape(batch, 1, heads * dim)), None
