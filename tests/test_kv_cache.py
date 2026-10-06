import numpy as np
import pytest

from transformer_systems import KVCache, dense_attention, grouped_query_decode


def test_cache_appends_chunks_and_exposes_only_active_tokens() -> None:
    cache = KVCache(batch_size=1, key_value_heads=2, capacity=5, head_width=3)
    first = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    second = np.full((1, 2, 1, 3), 7.0, dtype=np.float32)

    cache.append(first, -first)
    cache.append(second, -second)

    assert cache.length == 3
    np.testing.assert_array_equal(cache.keys[:, :, :2], first)
    np.testing.assert_array_equal(cache.keys[:, :, 2:], second)
    np.testing.assert_array_equal(cache.values, -cache.keys)


def test_cache_stats_distinguish_active_and_allocated_storage() -> None:
    cache = KVCache(
        batch_size=2,
        key_value_heads=4,
        capacity=10,
        head_width=8,
        dtype=np.float32,
    )
    chunk = np.ones((2, 4, 3, 8), dtype=np.float32)
    cache.append(chunk, chunk)

    stats = cache.stats()
    assert stats.active_bytes == 2 * 4 * 3 * 8 * 4 * 2
    assert stats.allocated_bytes == 2 * 4 * 10 * 8 * 4 * 2


def test_failed_overflow_does_not_change_cache() -> None:
    cache = KVCache(batch_size=1, key_value_heads=1, capacity=2, head_width=2)
    token = np.ones((1, 1, 1, 2), dtype=np.float32)
    cache.append(token, token)

    with pytest.raises(ValueError, match="capacity"):
        cache.append(
            np.ones((1, 1, 2, 2), dtype=np.float32),
            np.ones((1, 1, 2, 2), dtype=np.float32),
        )

    assert cache.length == 1
    np.testing.assert_array_equal(cache.keys, token)


def test_reset_reuses_allocation() -> None:
    cache = KVCache(batch_size=1, key_value_heads=1, capacity=2, head_width=2)
    allocated = cache.stats().allocated_bytes
    token = np.ones((1, 1, 1, 2), dtype=np.float32)
    cache.append(token, token)

    cache.reset()

    assert cache.length == 0
    assert cache.keys.shape[2] == 0
    assert cache.stats().allocated_bytes == allocated


@pytest.mark.parametrize("key_value_heads", [1, 2, 4])
def test_grouped_query_decode_matches_explicit_kv_repetition(key_value_heads) -> None:
    generator = np.random.default_rng(18)
    query_heads = 4
    group_size = query_heads // key_value_heads
    query = generator.standard_normal((2, query_heads, 5), dtype=np.float32)
    key = generator.standard_normal((2, key_value_heads, 7, 5), dtype=np.float32)
    value = generator.standard_normal((2, key_value_heads, 7, 5), dtype=np.float32)
    repeated_key = np.repeat(key, group_size, axis=1)
    repeated_value = np.repeat(value, group_size, axis=1)

    expected = dense_attention(query[:, :, None, :], repeated_key, repeated_value)[:, :, 0]
    actual = grouped_query_decode(query, key, value)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_grouped_query_cache_uses_fewer_bytes() -> None:
    multi_head = KVCache(batch_size=1, key_value_heads=8, capacity=32, head_width=16)
    grouped = KVCache(batch_size=1, key_value_heads=2, capacity=32, head_width=16)
    multi_chunk = np.ones((1, 8, 32, 16), dtype=np.float32)
    grouped_chunk = np.ones((1, 2, 32, 16), dtype=np.float32)
    multi_head.append(multi_chunk, multi_chunk)
    grouped.append(grouped_chunk, grouped_chunk)

    assert multi_head.stats().active_bytes == 4 * grouped.stats().active_bytes


def test_decode_rejects_non_divisible_head_counts() -> None:
    query = np.ones((1, 3, 4), dtype=np.float32)
    cache = np.ones((1, 2, 5, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="divisible"):
        grouped_query_decode(query, cache, cache)


def test_decode_rejects_empty_cache() -> None:
    query = np.ones((1, 2, 4), dtype=np.float32)
    cache = np.ones((1, 1, 0, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        grouped_query_decode(query, cache, cache)


@pytest.mark.parametrize(
    "kwargs, exception",
    [
        ({"batch_size": 0}, ValueError),
        ({"key_value_heads": False}, ValueError),
        ({"dtype": np.int32}, TypeError),
    ],
)
def test_cache_rejects_invalid_configuration(kwargs, exception) -> None:
    values = dict(batch_size=1, key_value_heads=1, capacity=2, head_width=2)
    values.update(kwargs)
    with pytest.raises(exception):
        KVCache(**values)
