import numpy as np
import pytest

from transformer_systems import (
    blockwise_attention,
    dense_attention,
    score_workspace_elements,
)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("block_size", [1, 3, 8])
def test_blockwise_attention_matches_dense_reference(causal, block_size) -> None:
    generator = np.random.default_rng(7)
    shape = (2, 3, 7, 5)
    query = generator.standard_normal(shape, dtype=np.float32)
    key = generator.standard_normal(shape, dtype=np.float32)
    value = generator.standard_normal((2, 3, 7, 4), dtype=np.float32)

    expected = dense_attention(query, key, value, causal=causal)
    actual = blockwise_attention(
        query,
        key,
        value,
        query_block_size=block_size,
        key_block_size=block_size,
        causal=causal,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_blockwise_attention_supports_cross_attention() -> None:
    generator = np.random.default_rng(11)
    query = generator.standard_normal((1, 2, 5, 4), dtype=np.float32)
    key = generator.standard_normal((1, 2, 9, 4), dtype=np.float32)
    value = generator.standard_normal((1, 2, 9, 6), dtype=np.float32)

    expected = dense_attention(query, key, value)
    actual = blockwise_attention(
        query,
        key,
        value,
        query_block_size=3,
        key_block_size=4,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_score_workspace_reflects_largest_tile() -> None:
    query = np.empty((2, 4, 100, 16), dtype=np.float32)
    key = np.empty((2, 4, 120, 16), dtype=np.float32)

    assert score_workspace_elements(query, key) == 2 * 4 * 100 * 120
    assert score_workspace_elements(
        query, key, query_block_size=32, key_block_size=48
    ) == 2 * 4 * 32 * 48


@pytest.mark.parametrize("kwargs", [{"query_block_size": 0}, {"key_block_size": False}])
def test_invalid_block_sizes_are_rejected(kwargs) -> None:
    values = {"query_block_size": 2, "key_block_size": 2}
    values.update(kwargs)
    array = np.ones((1, 1, 3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="positive integer"):
        blockwise_attention(array, array, array, **values)


def test_workspace_requires_both_block_sizes() -> None:
    array = np.ones((1, 1, 3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="both block sizes"):
        score_workspace_elements(array, array, query_block_size=2)
