"""Transparent parameter and memory accounting for transformer systems."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .config import TransformerConfig


@dataclass(frozen=True)
class ParameterBreakdown:
    token_embeddings: int
    position_embeddings: int
    attention: int
    feed_forward: int
    normalisation: int
    output_projection: int
    total: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryEstimate:
    sequence_length: int
    batch_size: int
    dtype_bytes: int
    key_value_cache_bytes: int
    naive_attention_scores_per_layer_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def count_parameters(config: TransformerConfig) -> ParameterBreakdown:
    """Count parameters under the documented reference architecture."""
    width = config.model_width
    ff_width = config.resolved_feed_forward_width
    head_width = config.head_width
    kv_width = config.resolved_key_value_heads * head_width

    token_embeddings = config.vocabulary_size * width
    position_embeddings = (
        config.max_sequence_length * width
        if config.learned_position_embeddings
        else 0
    )
    attention_weights_per_layer = (
        width * width
        + 2 * width * kv_width
        + width * width
    )
    attention_biases_per_layer = (
        width + 2 * kv_width + width if config.use_bias else 0
    )
    attention = config.layer_count * (
        attention_weights_per_layer + attention_biases_per_layer
    )
    feed_forward_weights_per_layer = 2 * width * ff_width
    feed_forward_biases_per_layer = ff_width + width if config.use_bias else 0
    feed_forward = config.layer_count * (
        feed_forward_weights_per_layer + feed_forward_biases_per_layer
    )
    normalisation = (2 * config.layer_count + 1) * width
    output_projection = 0 if config.tied_embeddings else config.vocabulary_size * width
    total = sum(
        (
            token_embeddings,
            position_embeddings,
            attention,
            feed_forward,
            normalisation,
            output_projection,
        )
    )
    return ParameterBreakdown(
        token_embeddings=token_embeddings,
        position_embeddings=position_embeddings,
        attention=attention,
        feed_forward=feed_forward,
        normalisation=normalisation,
        output_projection=output_projection,
        total=total,
    )


def estimate_memory(
    config: TransformerConfig,
    *,
    sequence_length: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> MemoryEstimate:
    """Estimate decoder KV cache and materialised attention-score storage.

    The score estimate describes a conventional implementation that stores the
    full attention matrix. Fused attention kernels need not materialise it.
    """
    for name, value in {
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "dtype_bytes": dtype_bytes,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if sequence_length > config.max_sequence_length:
        raise ValueError("sequence_length exceeds max_sequence_length")

    kv_cache = (
        batch_size
        * config.layer_count
        * sequence_length
        * 2
        * config.resolved_key_value_heads
        * config.head_width
        * dtype_bytes
    )
    attention_scores = (
        batch_size
        * config.attention_heads
        * sequence_length
        * sequence_length
        * dtype_bytes
    )
    return MemoryEstimate(
        sequence_length=sequence_length,
        batch_size=batch_size,
        dtype_bytes=dtype_bytes,
        key_value_cache_bytes=kv_cache,
        naive_attention_scores_per_layer_bytes=attention_scores,
    )


def profile_sequence_lengths(
    config: TransformerConfig,
    sequence_lengths: Iterable[int],
    *,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> dict[str, Any]:
    """Return a JSON-compatible analytical profile over sequence lengths."""
    estimates = tuple(
        estimate_memory(
            config,
            sequence_length=length,
            batch_size=batch_size,
            dtype_bytes=dtype_bytes,
        )
        for length in sequence_lengths
    )
    if not estimates:
        raise ValueError("at least one sequence length is required")
    return {
        "status": "analytical estimate; no hardware timing performed",
        "architecture": {
            "vocabulary_size": config.vocabulary_size,
            "model_width": config.model_width,
            "layer_count": config.layer_count,
            "attention_heads": config.attention_heads,
            "key_value_heads": config.resolved_key_value_heads,
            "head_width": config.head_width,
            "feed_forward_width": config.resolved_feed_forward_width,
            "tied_embeddings": config.tied_embeddings,
            "learned_position_embeddings": config.learned_position_embeddings,
            "use_bias": config.use_bias,
        },
        "parameters": count_parameters(config).to_dict(),
        "memory": [estimate.to_dict() for estimate in estimates],
    }


def maximum_cached_tokens(
    config: TransformerConfig,
    *,
    memory_budget_bytes: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """Return the context length fitting a KV-cache-only memory budget."""
    for name, value in {
        "memory_budget_bytes": memory_budget_bytes,
        "batch_size": batch_size,
        "dtype_bytes": dtype_bytes,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    bytes_per_token = (
        batch_size
        * config.layer_count
        * 2
        * config.resolved_key_value_heads
        * config.head_width
        * dtype_bytes
    )
    return min(config.max_sequence_length, memory_budget_bytes // bytes_per_token)
