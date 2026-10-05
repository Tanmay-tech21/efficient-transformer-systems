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
from .attention import blockwise_attention, dense_attention, score_workspace_elements

__all__ = [
    "MemoryEstimate",
    "ParameterBreakdown",
    "TimingSummary",
    "TransformerConfig",
    "benchmark_callable",
    "blockwise_attention",
    "count_parameters",
    "dense_attention",
    "estimate_memory",
    "maximum_cached_tokens",
    "profile_sequence_lengths",
    "score_workspace_elements",
    "summarise_timings",
]
