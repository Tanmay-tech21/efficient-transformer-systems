"""Readable NumPy attention used as a benchmark workload and reference."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


def _validate_attention_inputs(
    query: FloatArray,
    key: FloatArray,
    value: FloatArray,
    *,
    causal: bool,
) -> None:
    """Validate the shape and dtype contract shared by attention kernels."""
    if query.ndim < 2 or key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("query, key and value must have matching ranks of at least 2")
    if query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]:
        raise ValueError("query, key and value must share batch dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head widths must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")
    if causal and query.shape[-2] != key.shape[-2]:
        raise ValueError("causal attention requires equal query and key lengths")
    if not all(np.issubdtype(array.dtype, np.floating) for array in (query, key, value)):
        raise TypeError("attention inputs must use floating-point dtypes")


def _positive_block_size(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def dense_attention(
    query: FloatArray,
    key: FloatArray,
    value: FloatArray,
    *,
    causal: bool = False,
) -> FloatArray:
    """Compute scaled dot-product attention over the penultimate axis."""
    _validate_attention_inputs(query, key, value, causal=causal)

    scale = np.sqrt(np.asarray(query.shape[-1], dtype=query.dtype))
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) / scale
    if causal:
        mask = np.triu(np.ones(scores.shape[-2:], dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, value)


def blockwise_attention(
    query: FloatArray,
    key: FloatArray,
    value: FloatArray,
    *,
    query_block_size: int = 64,
    key_block_size: int = 64,
    causal: bool = False,
) -> FloatArray:
    """Compute exact attention without materialising the full score matrix.

    Each query tile maintains a running maximum, softmax denominator and
    value-weighted numerator. Rescaling those statistics when a new key tile
    arrives preserves the global softmax while bounding score-tile storage.
    """
    _validate_attention_inputs(query, key, value, causal=causal)
    _positive_block_size("query_block_size", query_block_size)
    _positive_block_size("key_block_size", key_block_size)

    query_length = query.shape[-2]
    key_length = key.shape[-2]
    value_width = value.shape[-1]
    scale = np.sqrt(np.asarray(query.shape[-1], dtype=query.dtype))
    output = np.empty((*query.shape[:-2], query_length, value_width), dtype=query.dtype)

    for query_start in range(0, query_length, query_block_size):
        query_stop = min(query_start + query_block_size, query_length)
        query_tile = query[..., query_start:query_stop, :]
        tile_rows = query_stop - query_start
        running_max = np.full((*query.shape[:-2], tile_rows, 1), -np.inf, dtype=query.dtype)
        running_sum = np.zeros_like(running_max)
        accumulator = np.zeros(
            (*query.shape[:-2], tile_rows, value_width), dtype=query.dtype
        )

        for key_start in range(0, key_length, key_block_size):
            if causal and key_start >= query_stop:
                break
            key_stop = min(key_start + key_block_size, key_length)
            key_tile = key[..., key_start:key_stop, :]
            value_tile = value[..., key_start:key_stop, :]
            scores = np.matmul(query_tile, np.swapaxes(key_tile, -1, -2)) / scale

            if causal:
                query_positions = np.arange(query_start, query_stop)[:, None]
                key_positions = np.arange(key_start, key_stop)[None, :]
                scores = np.where(key_positions > query_positions, -np.inf, scores)

            block_max = np.max(scores, axis=-1, keepdims=True)
            new_max = np.maximum(running_max, block_max)
            previous_scale = np.exp(running_max - new_max)
            weights = np.exp(scores - new_max)
            accumulator = (
                accumulator * previous_scale + np.matmul(weights, value_tile)
            )
            running_sum = running_sum * previous_scale + np.sum(
                weights, axis=-1, keepdims=True
            )
            running_max = new_max

        output[..., query_start:query_stop, :] = accumulator / running_sum
    return output


def score_workspace_elements(
    query: FloatArray,
    key: FloatArray,
    *,
    query_block_size: int | None = None,
    key_block_size: int | None = None,
) -> int:
    """Estimate elements in the largest dense or tiled score workspace."""
    if query.ndim < 2 or key.ndim != query.ndim:
        raise ValueError("query and key must have matching ranks of at least 2")
    if query.shape[:-2] != key.shape[:-2]:
        raise ValueError("query and key must share batch dimensions")
    if (query_block_size is None) != (key_block_size is None):
        raise ValueError("both block sizes must be supplied together")

    query_rows = query.shape[-2]
    key_rows = key.shape[-2]
    if query_block_size is not None and key_block_size is not None:
        _positive_block_size("query_block_size", query_block_size)
        _positive_block_size("key_block_size", key_block_size)
        query_rows = min(query_rows, query_block_size)
        key_rows = min(key_rows, key_block_size)
    batch_elements = int(np.prod(query.shape[:-2], dtype=np.int64))
    return batch_elements * query_rows * key_rows
