"""Validated architectural assumptions for decoder-only transformer estimates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for a dense pre-norm decoder-only transformer.

    The reference architecture uses multi-head or grouped-query attention,
    two affine projections in a GELU feed-forward block, two normalisations per
    layer, and a final normalisation. Token embeddings may be tied to the output
    projection. Rotary positions are assumed unless learned positions are enabled.
    """

    vocabulary_size: int
    model_width: int
    layer_count: int
    attention_heads: int
    max_sequence_length: int
    feed_forward_width: int | None = None
    key_value_heads: int | None = None
    tied_embeddings: bool = True
    learned_position_embeddings: bool = False
    use_bias: bool = False

    def __post_init__(self) -> None:
        integer_fields = {
            "vocabulary_size": self.vocabulary_size,
            "model_width": self.model_width,
            "layer_count": self.layer_count,
            "attention_heads": self.attention_heads,
            "max_sequence_length": self.max_sequence_length,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_width % self.attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")
        if self.feed_forward_width is not None and (
            not isinstance(self.feed_forward_width, int)
            or isinstance(self.feed_forward_width, bool)
            or self.feed_forward_width <= 0
        ):
            raise ValueError("feed_forward_width must be a positive integer")
        if self.key_value_heads is not None and (
            not isinstance(self.key_value_heads, int)
            or isinstance(self.key_value_heads, bool)
            or self.key_value_heads <= 0
        ):
            raise ValueError("key_value_heads must be a positive integer")
        if self.attention_heads % self.resolved_key_value_heads:
            raise ValueError("attention_heads must be divisible by key_value_heads")

    @property
    def head_width(self) -> int:
        return self.model_width // self.attention_heads

    @property
    def resolved_feed_forward_width(self) -> int:
        return self.feed_forward_width or 4 * self.model_width

    @property
    def resolved_key_value_heads(self) -> int:
        return self.key_value_heads or self.attention_heads
