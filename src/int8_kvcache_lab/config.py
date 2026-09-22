"""Configuration types for the dynamic KV-cache experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantConfig:
    """Symmetric INT8 quantization configuration.

    Query quantization is per sequence tensor. KV quantization is either one
    scale per KV head, or one scale per ``(kv_head, head_dim)`` channel.
    ``per_channel`` reduces over tokens only, so its scale depends on the QK
    reduction axis and must be folded into Q before an INT8 GEMM.
    """

    block_size: int = 16
    eps: float = 1e-8
    q_granularity: str = "per_tensor"
    kv_granularity: str = "per_head"
    scale_statistic: str = "absmax"

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.q_granularity != "per_tensor":
            raise ValueError("only per_tensor query quantization is supported")
        if self.kv_granularity not in ("per_head", "per_channel"):
            raise ValueError("kv_granularity must be per_head or per_channel")
        if self.scale_statistic not in ("absmax", "p999"):
            raise ValueError("scale_statistic must be absmax or p999")
