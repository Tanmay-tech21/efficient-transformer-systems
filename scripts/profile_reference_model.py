"""Produce a reproducible analytical profile for a decoder-only reference model."""

from __future__ import annotations

import json

from transformer_systems import TransformerConfig, profile_sequence_lengths


def main() -> None:
    config = TransformerConfig(
        vocabulary_size=32_000,
        model_width=4_096,
        layer_count=32,
        attention_heads=32,
        key_value_heads=8,
        max_sequence_length=8_192,
        feed_forward_width=11_008,
        tied_embeddings=True,
    )
    profile = profile_sequence_lengths(
        config,
        (128, 512, 2_048, 8_192),
        batch_size=1,
        dtype_bytes=2,
    )
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
