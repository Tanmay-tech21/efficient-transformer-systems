import pytest

from transformer_systems import (
    TransformerConfig,
    count_parameters,
    estimate_memory,
    maximum_cached_tokens,
    profile_sequence_lengths,
)


def tiny_config(**overrides) -> TransformerConfig:
    values = {
        "vocabulary_size": 100,
        "model_width": 16,
        "layer_count": 2,
        "attention_heads": 4,
        "max_sequence_length": 128,
        "feed_forward_width": 64,
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_dense_parameter_count_matches_hand_calculation() -> None:
    parameters = count_parameters(tiny_config())

    assert parameters.token_embeddings == 1_600
    assert parameters.attention == 2 * 4 * 16 * 16
    assert parameters.feed_forward == 2 * 2 * 16 * 64
    assert parameters.normalisation == 5 * 16
    assert parameters.output_projection == 0
    assert parameters.total == 7_824


def test_grouped_query_attention_reduces_attention_parameters() -> None:
    multi_head = count_parameters(tiny_config(key_value_heads=4))
    grouped_query = count_parameters(tiny_config(key_value_heads=1))

    assert grouped_query.attention < multi_head.attention


def test_untied_and_learned_embeddings_are_counted() -> None:
    parameters = count_parameters(
        tiny_config(tied_embeddings=False, learned_position_embeddings=True)
    )

    assert parameters.position_embeddings == 128 * 16
    assert parameters.output_projection == 100 * 16


def test_kv_cache_matches_closed_form() -> None:
    config = tiny_config(key_value_heads=1)
    estimate = estimate_memory(
        config, sequence_length=32, batch_size=3, dtype_bytes=2
    )

    assert estimate.key_value_cache_bytes == 3 * 2 * 32 * 2 * 1 * 4 * 2
    assert estimate.naive_attention_scores_per_layer_bytes == 3 * 4 * 32 * 32 * 2


def test_kv_cache_scales_linearly_with_context() -> None:
    config = tiny_config()
    short = estimate_memory(config, sequence_length=16)
    long = estimate_memory(config, sequence_length=64)

    assert long.key_value_cache_bytes == 4 * short.key_value_cache_bytes
    assert (
        long.naive_attention_scores_per_layer_bytes
        == 16 * short.naive_attention_scores_per_layer_bytes
    )


def test_maximum_cached_tokens_respects_budget_and_model_limit() -> None:
    config = tiny_config(key_value_heads=1)
    bytes_per_token = 2 * 2 * 1 * 4 * 2

    assert maximum_cached_tokens(
        config, memory_budget_bytes=bytes_per_token * 40
    ) == 40
    assert maximum_cached_tokens(
        config, memory_budget_bytes=bytes_per_token * 1_000
    ) == 128


def test_profile_labels_estimates_and_is_json_compatible() -> None:
    profile = profile_sequence_lengths(tiny_config(), (16, 32))

    assert profile["status"] == "analytical estimate; no hardware timing performed"
    assert [row["sequence_length"] for row in profile["memory"]] == [16, 32]
    assert isinstance(profile["parameters"]["total"], int)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_width": 15}, "divisible"),
        ({"key_value_heads": 3}, "key_value_heads"),
        ({"layer_count": 0}, "layer_count"),
        ({"feed_forward_width": -1}, "feed_forward_width"),
    ],
)
def test_invalid_architectures_are_rejected(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        tiny_config(**overrides)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequence_length": 0}, "sequence_length"),
        ({"sequence_length": 129}, "max_sequence_length"),
        ({"sequence_length": 16, "dtype_bytes": 0}, "dtype_bytes"),
    ],
)
def test_invalid_memory_requests_are_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_memory(tiny_config(), **kwargs)
