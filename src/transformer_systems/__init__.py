"""Analytical and empirical utilities for transformer inference systems."""

from .accounting import (
    MemoryEstimate,
    ParameterBreakdown,
    count_parameters,
    estimate_memory,
    maximum_cached_tokens,
    profile_sequence_lengths,
)
from .config import TransformerConfig
from .benchmarking import TimingSummary, benchmark_callable, summarise_timings
from .efficiency import (
    EfficiencyComparison,
    EfficiencyMeasurement,
    compare_efficiency,
)
from .attention import blockwise_attention, dense_attention, score_workspace_elements
from .kv_cache import KVCache, KVCacheStats, grouped_query_decode
from .quantization import (
    SymmetricInt8Weights,
    dequantize_weights,
    quantize_symmetric_int8,
    relative_root_mean_square_error,
    weight_only_linear,
)

__all__ = [
    "MemoryEstimate",
    "EfficiencyComparison",
    "EfficiencyMeasurement",
    "KVCache",
    "KVCacheStats",
    "ParameterBreakdown",
    "TimingSummary",
    "TransformerConfig",
    "SymmetricInt8Weights",
    "benchmark_callable",
    "blockwise_attention",
    "count_parameters",
    "compare_efficiency",
    "dense_attention",
    "dequantize_weights",
    "estimate_memory",
    "grouped_query_decode",
    "maximum_cached_tokens",
    "profile_sequence_lengths",
    "quantize_symmetric_int8",
    "relative_root_mean_square_error",
    "score_workspace_elements",
    "summarise_timings",
    "weight_only_linear",
]
