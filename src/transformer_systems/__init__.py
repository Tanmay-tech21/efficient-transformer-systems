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
from .attention import dense_attention

__all__ = [
    "MemoryEstimate",
    "ParameterBreakdown",
    "TimingSummary",
    "TransformerConfig",
    "benchmark_callable",
    "count_parameters",
    "dense_attention",
    "estimate_memory",
    "maximum_cached_tokens",
    "profile_sequence_lengths",
    "summarise_timings",
]
