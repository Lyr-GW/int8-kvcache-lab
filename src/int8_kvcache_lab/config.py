"""Configuration types for the dynamic KV-cache experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantConfig:
    """Symmetric INT8 quantization configuration.

    The reference path quantizes each sequence's query with one scale and uses
    one K and V scale per KV head over all valid cache tokens in the batch.
    """

    block_size: int = 16
    eps: float = 1e-8
    q_granularity: str = "per_tensor"
    kv_granularity: str = "per_head"

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.q_granularity != "per_tensor":
            raise ValueError("only per_tensor query quantization is supported")
        if self.kv_granularity != "per_head":
            raise ValueError("only per_head KV quantization is supported")
