"""Shared GEMM launch helpers used by body-aware code generation and runtime."""

# These table-format helpers are shared by the currently registered GEMM
# implementations. Hardware-specific imports stay within the bodies package.
from .cuda.sm90.cutlass_gemm import STREAM_TAG_SLOT
from .cuda.sm90.gemm import (
    TABLE_DESCRIPTORS_PER_GEMM,
    cute_gemm_tensors,
    epilogue_loads,
    epilogue_source,
    has_bias,
    has_fused_epilogue,
    is_paired,
    is_swap_ab,
    kernel_tile,
    split_k,
    table_gemm_shared_bytes,
)

__all__ = [
    "STREAM_TAG_SLOT",
    "TABLE_DESCRIPTORS_PER_GEMM",
    "cute_gemm_tensors",
    "epilogue_loads",
    "epilogue_source",
    "has_bias",
    "has_fused_epilogue",
    "is_paired",
    "is_swap_ab",
    "kernel_tile",
    "split_k",
    "table_gemm_shared_bytes",
]
