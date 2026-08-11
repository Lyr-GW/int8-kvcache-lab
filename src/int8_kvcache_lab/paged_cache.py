"""A small, explicit PagedAttention KV-cache representation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PagedKVCache:
    """KV cache with physical layout ``[blocks, block_size, 2, heads, dim]``."""

    values: torch.Tensor

    @classmethod
    def empty(
        cls,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "PagedKVCache":
        if min(num_blocks, block_size, num_kv_heads, head_dim) <= 0:
            raise ValueError("cache dimensions must be positive")
        return cls(torch.zeros((num_blocks, block_size, 2, num_kv_heads, head_dim), dtype=dtype, device=device))

    def __post_init__(self) -> None:
        if self.values.ndim != 5 or self.values.shape[2] != 2:
            raise ValueError("cache shape must be [blocks, block_size, 2, heads, dim]")
        if not (self.values.is_floating_point() or self.values.dtype == torch.int8):
            raise TypeError("cache values must be floating point or torch.int8")

    @property
    def num_blocks(self) -> int:
        return self.values.shape[0]

    @property
    def block_size(self) -> int:
        return self.values.shape[1]

    @property
    def num_kv_heads(self) -> int:
        return self.values.shape[3]

    @property
    def head_dim(self) -> int:
        return self.values.shape[4]

    @property
    def fp_bytes(self) -> int:
        return self.values.numel() * self.values.element_size()

    @property
    def int8_bytes(self) -> int:
        return self.values.numel()

    def write(self, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Write tokens into physical slots.

        Inputs use ``[tokens, kv_heads, head_dim]`` and slot IDs encode
        ``physical_block * block_size + block_offset``. Duplicate slots are
        rejected because their order would otherwise be ambiguous.
        """
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key and value must have [tokens, heads, dim] shape")
        if key.shape[1:] != (self.num_kv_heads, self.head_dim):
            raise ValueError("key/value head shape does not match cache")
        if slot_mapping.ndim != 1 or slot_mapping.numel() != key.shape[0]:
            raise ValueError("slot_mapping must have one entry per token")
        slots = slot_mapping.to(device=self.values.device, dtype=torch.long)
        if torch.unique(slots).numel() != slots.numel():
            raise ValueError("slot_mapping must not contain duplicate slots")
        if torch.any(slots < 0) or torch.any(slots >= self.num_blocks * self.block_size):
            raise ValueError("slot_mapping contains an out-of-range slot")
        blocks = torch.div(slots, self.block_size, rounding_mode="floor")
        offsets = slots.remainder(self.block_size)
        self.values[blocks, offsets, 0] = key.to(self.values.dtype)
        self.values[blocks, offsets, 1] = value.to(self.values.dtype)

    def valid_mask(self, block_tables: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """Return physical valid-token positions addressed by request pages."""
        self._validate_metadata(block_tables, seq_lens)
        mask = torch.zeros((self.num_blocks, self.block_size), dtype=torch.bool, device=self.values.device)
        for request, length in enumerate(seq_lens.tolist()):
            for token in range(length):
                block = int(block_tables[request, token // self.block_size])
                mask[block, token % self.block_size] = True
        return mask

    def gather(self, block_tables: torch.Tensor, seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather logical K/V into padded tensors and return their validity mask."""
        self._validate_metadata(block_tables, seq_lens)
        batch = block_tables.shape[0]
        max_len = int(seq_lens.max().item()) if batch else 0
        key = torch.zeros((batch, max_len, self.num_kv_heads, self.head_dim), dtype=self.values.dtype, device=self.values.device)
        value = torch.zeros_like(key)
        mask = torch.zeros((batch, max_len), dtype=torch.bool, device=self.values.device)
        for request, length in enumerate(seq_lens.tolist()):
            for token in range(length):
                block = int(block_tables[request, token // self.block_size])
                offset = token % self.block_size
                key[request, token] = self.values[block, offset, 0]
                value[request, token] = self.values[block, offset, 1]
                mask[request, token] = True
        return key, value, mask

    def _validate_metadata(self, block_tables: torch.Tensor, seq_lens: torch.Tensor) -> None:
        if block_tables.ndim != 2 or seq_lens.ndim != 1 or block_tables.shape[0] != seq_lens.numel():
            raise ValueError("block_tables must be [batch, pages] and seq_lens [batch]")
        if block_tables.dtype not in (torch.int32, torch.int64):
            raise TypeError("block_tables must contain integer physical block IDs")
        if seq_lens.dtype not in (torch.int32, torch.int64):
            raise TypeError("seq_lens must contain integers")
        if torch.any(seq_lens < 0):
            raise ValueError("seq_lens must be non-negative")
        pages_needed = torch.div(seq_lens + self.block_size - 1, self.block_size, rounding_mode="floor")
        if torch.any(pages_needed > block_tables.shape[1]):
            raise ValueError("block_tables has too few pages for a sequence")
        for request, count in enumerate(pages_needed.tolist()):
            ids = block_tables[request, :count]
            if torch.any(ids < 0) or torch.any(ids >= self.num_blocks):
                raise ValueError("block_tables contains an invalid referenced block")
