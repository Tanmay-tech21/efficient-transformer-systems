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
from .reporting import (
    create_report_snapshot,
    render_systems_report,
    validate_efficiency_matrix,
    write_report_artifacts,
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
    "create_report_snapshot",
    "dense_attention",
    "dequantize_weights",
    "estimate_memory",
    "grouped_query_decode",
    "maximum_cached_tokens",
    "profile_sequence_lengths",
    "quantize_symmetric_int8",
    "render_systems_report",
    "relative_root_mean_square_error",
    "score_workspace_elements",
    "summarise_timings",
    "validate_efficiency_matrix",
    "weight_only_linear",
    "write_report_artifacts",
]
